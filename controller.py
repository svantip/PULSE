from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict

# Initialize FastAPI app
app = FastAPI(title="Classification API")

# Load model and tokenizer from Hugging Face
MODEL_NAME_1 = "svantip123/urgency_classificator"
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_NAME_1)
model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_1)
model_1.eval()

# Request model


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1,
                         description="Text message to classify")

# Response model


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


@app.post("ugrncy/predict", response_model=PredictionResponse)
async def predict_urgency(request: MessageRequest):
    """
    Predict urgency level for a given message.

    Args:
        request: MessageRequest containing the message text

    Returns:
        PredictionResponse with predicted class, confidence, and probabilities
    """
    try:
        # Tokenize input
        inputs = tokenizer_1(
            request.message,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

        # Map to class labels (adjust based on your model's labels)
        # Update based on your actual labels
        label_map = {0: "low", 1: "medium", 2: "high"}
        predicted_class = label_map.get(
            predicted_idx, f"class_{predicted_idx}")

        # Create probability dictionary
        prob_dict = {
            label_map.get(i, f"class_{i}"): prob.item()
            for i, prob in enumerate(probabilities)
        }

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME_1}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
