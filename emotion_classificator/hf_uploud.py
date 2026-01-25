from huggingface_hub import HfApi

api = HfApi()

# Uploadaj cijeli folder
api.upload_folder(
    folder_path="./urgency_classificator/models/latest",  # putanja do foldera
    repo_id="tvoj-username/urgency-classifier",
    repo_type="model"
)
