import os
import argparse
import yaml
import numpy as np
import pandas as pd

import evaluate
import mlflow

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_csv_as_dataset(path: str, text_col: str, label_col: str) -> Dataset:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV mora imat kolone '{text_col}' i '{label_col}'. Nađeno: {list(df.columns)}")
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    df = df.rename(columns={text_col: "text", label_col: "emotion"})
    return Dataset.from_pandas(df, preserve_index=False)

def apply_label_map(ds: Dataset, label_map: dict) -> Dataset:
    if not label_map:
        return ds

    def _map(ex):
        raw = ex["emotion"]
        if raw not in label_map:
            raise ValueError(f"Raw labela '{raw}' nije u LABEL_MAP (config.yaml).")
        return {"emotion": label_map[raw]}

    return ds.map(_map)

def encode_labels(ds: Dataset, label2id: dict) -> Dataset:
    def _enc(ex):
        lab = ex["emotion"]
        if lab not in label2id:
            raise ValueError(f"Labela '{lab}' nije u LABELS (config.yaml).")
        return {"labels": int(label2id[lab])}
    return ds.map(_enc)

def tokenize(ds: Dataset, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True)
    return ds.map(_tok, batched=True, remove_columns=["text", "emotion"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    data_cfg = cfg["DATA"]
    csv_path = data_cfg["CSV_PATH"]
    text_col = data_cfg.get("TEXT_COL", "text")
    label_col = data_cfg.get("LABEL_COL", "emotion")
    valid_size = float(data_cfg.get("VALID_SIZE", 0.1))
    stratify = bool(data_cfg.get("STRATIFY", True))
    seed = int(data_cfg.get("SEED", 42))
    output_root = data_cfg.get("OUTPUT_ROOT", "outputs")

    set_seed(seed)

    labels = cfg["LABELS"]
    num_labels = int(cfg["NUM_LABELS"])
    if num_labels != len(labels):
        raise ValueError(f"NUM_LABELS={num_labels}, ali LABELS ima {len(labels)} elemenata.")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    ds = load_csv_as_dataset(csv_path, text_col=text_col, label_col=label_col)

    ds = apply_label_map(ds, cfg.get("LABEL_MAP", {}))
    ds = encode_labels(ds, label2id)

    # Split (stratify opcionalno)
    split_kwargs = {"test_size": valid_size, "seed": seed}
    if stratify:
        split_kwargs["stratify_by_column"] = "labels"  # HF datasets podržava stratify_by_column [web:135]
    splits = ds.train_test_split(**split_kwargs)
    train_ds, valid_ds = splits["train"], splits["test"]

    tr_cfg = cfg["TRAINING"]
    lr = float(tr_cfg["LEARNING_RATE"])
    bs = int(tr_cfg["BATCH_SIZE"])
    epochs = int(tr_cfg["NUM_EPOCHS"])
    wd = float(tr_cfg["WEIGHT_DECAY"])
    eval_strategy = tr_cfg.get("EVAL_STRATEGY", "epoch")
    save_strategy = tr_cfg.get("SAVE_STRATEGY", "epoch")
    best_metric = tr_cfg.get("BEST_MODEL_METRIC", "f1")

    # Metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        out = {}
        out.update(acc.compute(predictions=y_pred, references=y_true))
        out.update(f1.compute(predictions=y_pred, references=y_true, average="macro"))
        return out

    # MLflow
    versioning = cfg.get("VERSIONING", {})
    mlflow_enabled = bool(versioning.get("ENABLED", False))
    if mlflow_enabled:
        mlflow.set_tracking_uri(versioning.get("MLFLOW_TRACKING_URI", "./mlruns"))
        mlflow.set_experiment(versioning.get("EXPERIMENT_NAME", "emotion_classification"))

    os.makedirs(output_root, exist_ok=True)

    results = []
    for m in cfg["MODELS"]:
        model_name = m["NAME"]
        alias = m["ALIAS"]

        run = mlflow.start_run(run_name=alias) if mlflow_enabled else None
        try:
            if mlflow_enabled:
                mlflow.log_params({
                    "model_name": model_name,
                    "alias": alias,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "num_epochs": epochs,
                    "weight_decay": wd,
                    "num_labels": num_labels,
                    "seed": seed,
                    "valid_size": valid_size,
                    "stratify": stratify,
                })

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            collator = DataCollatorWithPadding(tokenizer=tokenizer)

            tok_train = tokenize(train_ds, tokenizer)
            tok_valid = tokenize(valid_ds, tokenizer)

            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )

            out_dir = os.path.join(output_root, alias)

            training_args = TrainingArguments(
                output_dir=out_dir,
                learning_rate=lr,
                per_device_train_batch_size=bs,
                per_device_eval_batch_size=bs,
                num_train_epochs=epochs,
                weight_decay=wd,
                eval_strategy=eval_strategy,  # npr "epoch" [web:136]
                save_strategy=save_strategy,
                load_best_model_at_end=True,
                metric_for_best_model=best_metric,
                report_to="none",
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tok_train,
                eval_dataset=tok_valid,
                tokenizer=tokenizer,
                data_collator=collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            metrics = trainer.evaluate()

            trainer.save_model(out_dir)
            tokenizer.save_pretrained(out_dir)

            if mlflow_enabled:
                mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
                mlflow.log_artifacts(out_dir, artifact_path=f"model_{alias}")

            results.append((alias, metrics))

        finally:
            if run is not None:
                mlflow.end_run()

    print("\n=== RESULTS ===")
    for alias, metrics in results:
        show = {k: metrics[k] for k in metrics if k in ["eval_accuracy", "eval_f1", "eval_loss"]}
        print(alias, show)

if __name__ == "__main__":
    main()
