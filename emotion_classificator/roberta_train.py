import os
import argparse
import yaml
import numpy as np
import pandas as pd
import evaluate

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback,
)

MODEL_NAME = "roberta-base"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_as_dataset(path: str, text_col: str, label_col: str) -> Dataset:
    df = pd.read_csv(path)
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
            raise ValueError(f"Raw label '{raw}' not found in LABEL_MAP.")
        return {"emotion": label_map[raw]}

    return ds.map(_map)


def encode_labels(ds: Dataset, label2id: dict) -> Dataset:
    def _enc(ex):
        lab = ex["emotion"]
        if lab not in label2id:
            raise ValueError(f"Label '{lab}' not found in LABELS.")
        return {"labels": int(label2id[lab])}

    return ds.map(_enc)


def tokenize(ds: Dataset, tokenizer):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=512)

    return ds.map(_tok, batched=True, remove_columns=["text", "emotion"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="emotion_classificator/config.yaml")
    ap.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Path to a checkpoint-* folder (or use True to resume last checkpoint in output_dir).",
    )
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
        raise ValueError(f"NUM_LABELS={num_labels}, but LABELS has {len(labels)} items.")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    ds = load_csv_as_dataset(csv_path, text_col=text_col, label_col=label_col)
    ds = apply_label_map(ds, cfg.get("LABEL_MAP", {}))
    ds = encode_labels(ds, label2id)

    split_kwargs = {"test_size": valid_size, "seed": seed}
    if stratify:
        split_kwargs["stratify_by_column"] = "labels"
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

    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        y_pred = np.argmax(logits, axis=-1)
        out = {}
        out.update(acc.compute(predictions=y_pred, references=y_true))
        out.update(f1.compute(predictions=y_pred, references=y_true, average="macro"))
        return out

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tok_train = tokenize(train_ds, tokenizer)
    tok_valid = tokenize(valid_ds, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    out_dir = os.path.join("emotion_classificator", output_root, "roberta")
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        weight_decay=wd,

        eval_strategy=eval_strategy,
        save_strategy=save_strategy,

        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,

        save_total_limit=2,   # keep best + last
        logging_strategy="epoch",
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

    )

    # Resume if requested:
    # - if string path -> resume from that checkpoint
    # - if "true"/"True" -> resume last checkpoint in output_dir
    resume = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "true":
            resume = True
        else:
            resume = args.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=resume)  # [web:313]
    metrics = trainer.evaluate()

    best_ckpt = trainer.state.best_model_checkpoint
    print("BEST CHECKPOINT:", best_ckpt)  # [web:308]

    export_dir = os.path.join(out_dir, "best_model")
    trainer.model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print("EXPORTED BEST MODEL TO:", export_dir)

    print("EVAL:", {k: metrics[k] for k in metrics if k in ["eval_accuracy", "eval_f1", "eval_loss"]})


if __name__ == "__main__":
    main()
