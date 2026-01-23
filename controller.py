from slack_service import SlackTicketAnalyzer
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch
import numpy as np
import re
import os
import json
import hmac
import hashlib
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Classification API")

# Initialize Slack Ticket Analyzer
slack_analyzer = SlackTicketAnalyzer(
    urgency_model_path=None,  # Will use HuggingFace model
    emotion_model_path=None,  # Will use PEFT model
    slack_bot_token=os.getenv("SLACK_BOT_TOKEN")
)

# ==================== URGENCY MODEL ====================
# Load urgency model and tokenizer from Hugging Face
MODEL_NAME_1 = "svantip123/urgency_classificator"
tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_NAME_1)
model_1 = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_1)
model_1.eval()
# ==================== EMOTION MODEL (PEFT LoRA) ====================
# Load emotion PEFT model
MODEL_NAME_2 = "drPantagana/PULSE_emotion"

print(f"Loading PEFT emotion model from {MODEL_NAME_2}...")

# Load PEFT config
peft_config_2 = PeftConfig.from_pretrained(MODEL_NAME_2)
print(f"✓ PEFT config loaded. Base model: {peft_config_2.base_model_name_or_path}")

# Load base model for emotion (roberta-base)
base_model_2 = AutoModelForSequenceClassification.from_pretrained(
    peft_config_2.base_model_name_or_path,
    num_labels=6  # joy, sadness, anger, love, surprise, neutral
)
print(f"✓ Base model loaded: {peft_config_2.base_model_name_or_path}")

# Load PEFT LoRA adapter on top of base model
model_2 = PeftModel.from_pretrained(base_model_2, MODEL_NAME_2)
model_2.eval()
print(f"✓ PEFT LoRA adapter loaded from {MODEL_NAME_2}")

# Load tokenizer from base model
tokenizer_2 = AutoTokenizer.from_pretrained(peft_config_2.base_model_name_or_path)
print(f"✓ Tokenizer loaded")

# Emotion labels (ensure these match your training labels)
EMOTION_LABELS = ["joy", "sadness", "anger", "love", "surprise", "neutral"]

# ==================== RULE-BASED KEYWORDS ====================
# URGENCY KEYWORDS (supplementary to model's attention weights)
# NOTE: These are simple heuristics, NOT what the model actually learned.
# The model's attention weights (from extract_important_tokens) are the true explanation.
# These keywords provide additional context but may not match what the model focuses on.
URGENCY_KEYWORDS = {
    "high": [
        "urgent", "critical", "emergency", "down", "broken", "not working",
        "immediately", "asap", "production", "outage", "crash", "failed",
        "security", "breach", "data loss", "cannot access", "blocking"
    ],
    "medium": [
        "issue", "problem", "error", "bug", "slow", "delayed",
        "intermittent", "sometimes", "workaround", "affecting"
    ],
    "low": [
        "question", "request", "when possible", "help", "how to",
        "information", "documentation", "feature request"
    ]
}

# EMOTION KEYWORDS (supplementary to model's attention weights)
# NOTE: Same as urgency - these are heuristics, not what the PEFT model learned
EMOTION_KEYWORDS = {
    "joy": [
        "happy", "great", "wonderful", "excellent", "love", "amazing", 
        "fantastic", "perfect", "delighted", "thrilled", "excited", "glad"
    ],
    "sadness": [
        "sad", "disappointed", "unhappy", "terrible", "awful", "bad", 
        "depressed", "miserable", "upset", "hurt", "down", "blue"
    ],
    "anger": [
        "angry", "furious", "mad", "frustrated", "annoyed", "outraged", 
        "hate", "irritated", "rage", "pissed", "livid", "infuriated"
    ],
    "love": [
        "love", "adore", "care", "cherish", "appreciate", "grateful", 
        "thankful", "blessed", "treasure", "devoted", "fond"
    ],
    "surprise": [
        "wow", "amazing", "unexpected", "surprised", "shocking", 
        "incredible", "unbelievable", "astonishing", "stunning", "speechless"
    ],
    "neutral": [
        "ok", "fine", "alright", "normal", "regular", "standard", 
        "average", "typical", "ordinary"
    ]
}

# ==================== PYDANTIC MODELS ====================

class SlackChallenge(BaseModel):
    """Slack URL verification challenge"""
    challenge: str
    type: str = "url_verification"

class SlackMessage(BaseModel):
    """Slack message event"""
    text: str
    user: str
    channel: str
    ts: str

class SlackEvent(BaseModel):
    """Slack event wrapper"""
    type: str
    event: Optional[Dict] = None
    challenge: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Request for combined analysis"""
    text: str = Field(..., min_length=1, description="Text to analyze")
    slack_channel: Optional[str] = Field(
        None, description="Slack channel to post results")
    include_explanation: bool = Field(
        True, description="Include detailed explanations")

class AnalysisResponse(BaseModel):
    """Combined analysis response"""
    timestamp: str
    text: str
    urgency: Dict
    emotion: Dict
    priority_flag: Dict
    report: Optional[str] = None

class MessageRequest(BaseModel):
    """Request model for prediction"""
    message: str = Field(..., min_length=1,
                         description="Text message to classify")

class PredictionResponse(BaseModel):
    """Response model for prediction"""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: Optional[Dict] = None

# ==================== HELPER FUNCTIONS ====================

def extract_important_tokens(text: str, inputs: Dict, model, tokenizer, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Extract important tokens using MODEL'S ATTENTION WEIGHTS.
    This is the TRUE model-based explainability - shows what the model actually focused on.

    Args:
        text: Input text
        inputs: Tokenized inputs
        model: The model (can be standard or PEFT)
        tokenizer: The tokenizer
        top_k: Number of top tokens to return

    Returns:
        List of (token, importance_score) tuples based on attention mechanism
    """
    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            attentions = outputs.attentions

        # Get attention from last layer, average across heads
        # Shape: [batch, heads, seq_len, seq_len]
        last_layer_attention = attentions[-1][0]  # First batch item

        # Average across attention heads
        avg_attention = last_layer_attention.mean(dim=0)  # [seq_len, seq_len]

        # Get attention TO the CLS token (first token) FROM all other tokens
        cls_attention = avg_attention[0, :].cpu().numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Pair tokens with their attention scores
        token_importance = []
        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            # Skip special tokens and subword markers
            if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', '<mask>']:
                # Clean up subword tokens (works for both BERT and RoBERTa tokenizers)
                clean_token = token.replace('##', '').replace('Ġ', '')
                if clean_token.strip():
                    token_importance.append((clean_token, float(score)))

        # Sort by importance and return top k
        token_importance.sort(key=lambda x: x[1], reverse=True)
        return token_importance[:top_k]

    except Exception as e:
        print(f"Error extracting tokens: {e}")
        return []

def find_keyword_hints(text: str, keywords_dict: Dict, predicted_class: str = None) -> List[str]:
    """
    Find rule-based keyword hints in the text (SUPPLEMENTARY ONLY).
    NOTE: This is NOT what the model used - just simple pattern matching for additional context.
    The model's attention weights are the real explanation.

    Args:
        text: Input text
        keywords_dict: Dictionary of category -> keywords
        predicted_class: Predicted class (optional, for filtering)

    Returns:
        List of found keyword hints (not model-based)
    """
    text_lower = text.lower()
    found_indicators = []

    # Check all keyword categories
    for category, keywords in keywords_dict.items():
        for keyword in keywords:
            if keyword in text_lower:
                found_indicators.append(f"{keyword} ({category})")

    return found_indicators

def generate_urgency_explanation(text: str, predicted_class: str, confidence: float,
                                 important_tokens: List[Tuple[str, float]],
                                 keyword_hints: List[str]) -> Dict:
    """
    Generate a human-readable explanation for urgency prediction.
    Combines MODEL-BASED explainability (attention weights) with optional rule-based hints.

    Args:
        text: Input text
        predicted_class: Predicted urgency level
        confidence: Prediction confidence
        important_tokens: Important tokens from MODEL'S attention weights (primary explanation)
        keyword_hints: Rule-based keyword hints (supplementary only)

    Returns:
        Explanation dictionary with model-based and rule-based insights clearly separated
    """
    reasoning_parts = []

    # Model confidence (PRIMARY)
    if confidence > 0.8:
        reasoning_parts.append(
            f"The model is highly confident ({confidence*100:.1f}%) in this classification.")
    elif confidence > 0.6:
        reasoning_parts.append(
            f"The model has moderate confidence ({confidence*100:.1f}%) in this classification.")
    else:
        reasoning_parts.append(
            f"The model has low confidence ({confidence*100:.1f}%). The text may be ambiguous.")

    # Model's attention-based tokens (PRIMARY EXPLANATION)
    if important_tokens:
        top_tokens = [token for token, _ in important_tokens[:5]]
        reasoning_parts.append(
            f"The model focused most on: {', '.join(top_tokens)}.")

    # Rule-based keyword hints (SUPPLEMENTARY - may not match model's focus)
    if keyword_hints:
        high_hints = [h for h in keyword_hints if "(high)" in h]
        medium_hints = [h for h in keyword_hints if "(medium)" in h]
        low_hints = [h for h in keyword_hints if "(low)" in h]

        hint_parts = []
        if high_hints:
            keywords = ", ".join([h.split(" (")[0] for h in high_hints[:3]])
            hint_parts.append(f"high urgency: {keywords}")
        if medium_hints:
            keywords = ", ".join([h.split(" (")[0] for h in medium_hints[:3]])
            hint_parts.append(f"medium urgency: {keywords}")
        if low_hints:
            keywords = ", ".join([h.split(" (")[0] for h in low_hints[:3]])
            hint_parts.append(f"low urgency: {keywords}")

        if hint_parts:
            reasoning_parts.append(
                f"Rule-based hints found ({'; '.join(hint_parts)}).")

    # Text characteristics
    word_count = len(text.split())
    if word_count < 10:
        reasoning_parts.append(
            "The text is quite short, which may affect classification accuracy.")

    return {
        # PRIMARY: What the model actually used
        "model_attention_tokens": important_tokens,
        # SUPPLEMENTARY: Simple pattern matching
        "rule_based_keyword_hints": keyword_hints,
        "reasoning": " ".join(reasoning_parts),
        "text_length": word_count
    }

def generate_emotion_explanation(text: str, predicted_class: str, confidence: float,
                                important_tokens: List[Tuple[str, float]],
                                emotion_indicators: List[str]) -> Dict:
    """
    Generate a human-readable explanation for emotion prediction.
    Combines MODEL-BASED explainability (PEFT LoRA attention) with optional rule-based hints.

    Args:
        text: Input text
        predicted_class: Predicted emotion
        confidence: Prediction confidence
        important_tokens: Important tokens from PEFT MODEL'S attention weights (primary)
        emotion_indicators: Rule-based keyword hints (supplementary only)

    Returns:
        Explanation dictionary with model-based and rule-based insights clearly separated
    """
    reasoning_parts = []

    # Model confidence (PRIMARY)
    if confidence > 0.8:
        reasoning_parts.append(
            f"High confidence ({confidence*100:.1f}%) in {predicted_class} classification.")
    elif confidence > 0.6:
        reasoning_parts.append(
            f"Moderate confidence ({confidence*100:.1f}%) in {predicted_class} classification.")
    else:
        reasoning_parts.append(
            f"Low confidence ({confidence*100:.1f}%). Text may be emotionally ambiguous.")

    # Model's attention-based tokens (PRIMARY EXPLANATION from PEFT LoRA)
    if important_tokens:
        top_tokens = [token for token, _ in important_tokens[:5]]
        reasoning_parts.append(
            f"Model focused on: {', '.join(top_tokens)}.")

    # Rule-based emotion keywords (SUPPLEMENTARY - may not match model's focus)
    if emotion_indicators:
        keywords = ", ".join([ind.split(" (")[0] for ind in emotion_indicators[:3]])
        reasoning_parts.append(
            f"Found emotion keywords: {keywords}.")

    # Text characteristics
    word_count = len(text.split())
    if word_count < 10:
        reasoning_parts.append(
            "Short text may limit emotion detection accuracy.")

    return {
        # PRIMARY: What the PEFT model actually used
        "model_attention_tokens": important_tokens,
        # SUPPLEMENTARY: Simple pattern matching
        "emotion_indicators": emotion_indicators,
        "reasoning": " ".join(reasoning_parts),
        "text_length": word_count,
        "model_type": "PEFT LoRA"
    }

# ==================== API ENDPOINTS ====================

@app.post("/urgency/predict", response_model=PredictionResponse)
async def predict_urgency(request: MessageRequest):
    """
    Predict urgency level for a given message with explainability.

    Args:
        request: MessageRequest containing the message text

    Returns:
        PredictionResponse with predicted class, confidence, probabilities, and explanation
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
            outputs = model_1(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

        # Map to class labels
        label_map = {0: "low", 1: "medium", 2: "high"}
        predicted_class = label_map.get(
            predicted_idx, f"class_{predicted_idx}")

        # Create probability dictionary
        prob_dict = {
            label_map.get(i, f"class_{i}"): prob.item()
            for i, prob in enumerate(probabilities)
        }

        # Generate explainability
        # PRIMARY: Extract tokens based on model's attention weights
        model_attention_tokens = extract_important_tokens(
            request.message, inputs, model_1, tokenizer_1, top_k=10
        )

        # SUPPLEMENTARY: Simple rule-based keyword matching (not model-based)
        keyword_hints = find_keyword_hints(
            request.message, URGENCY_KEYWORDS, predicted_class
        )

        # Combine model-based and rule-based explanations
        explanation = generate_urgency_explanation(
            request.message,
            predicted_class,
            confidence,
            model_attention_tokens,
            keyword_hints
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/emotion/predict", response_model=PredictionResponse)
async def predict_emotion(request: MessageRequest):
    """
    Predict emotion for a given message with explainability using PEFT LoRA model.
    
    This endpoint uses a fine-tuned RoBERTa model with LoRA adapters for emotion classification.
    The model was trained to detect: joy, sadness, anger, love, surprise, and neutral emotions.

    Args:
        request: MessageRequest containing the message text

    Returns:
        PredictionResponse with predicted emotion, confidence, probabilities, and explanation
    """
    try:
        # Tokenize input using RoBERTa tokenizer
        inputs = tokenizer_2(
            request.message,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        # Make prediction using PEFT LoRA model
        with torch.no_grad():
            outputs = model_2(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        # Get predicted class
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        predicted_emotion = EMOTION_LABELS[predicted_idx]

        # Create probability dictionary
        prob_dict = {
            label: prob.item()
            for label, prob in zip(EMOTION_LABELS, probabilities)
        }

        # Generate explainability
        # PRIMARY: Extract tokens based on PEFT model's attention weights
        model_attention_tokens = extract_important_tokens(
            request.message, inputs, model_2, tokenizer_2, top_k=10
        )

        # SUPPLEMENTARY: Simple rule-based emotion keyword matching (not model-based)
        emotion_indicators = find_keyword_hints(
            request.message, EMOTION_KEYWORDS, predicted_emotion
        )

        # Combine PEFT model-based and rule-based explanations
        explanation = generate_emotion_explanation(
            request.message,
            predicted_emotion,
            confidence,
            model_attention_tokens,
            emotion_indicators
        )

        return PredictionResponse(
            predicted_class=predicted_emotion,
            confidence=confidence,
            probabilities=prob_dict,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Emotion prediction error: {str(e)}")

@app.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_signature: Optional[str] = Header(None),
    x_slack_request_timestamp: Optional[str] = Header(None)
):
    """
    Slack Events API endpoint
    Handles incoming Slack events including messages

    To set this up:
    1. Create a Slack App at https://api.slack.com/apps
    2. Enable Event Subscriptions
    3. Set Request URL to: http://your-server:8000/slack/events
    4. Subscribe to bot events: message.channels, message.groups, message.im
    5. Install app to workspace and copy Bot Token to .env as SLACK_BOT_TOKEN
    """
    body_bytes = await request.body()
    body = json.loads(body_bytes.decode('utf-8'))

    # Verify Slack signature (optional but recommended for production)
    if os.getenv("SLACK_SIGNING_SECRET") and x_slack_signature and x_slack_request_timestamp:
        if not verify_slack_signature(
            body_bytes,
            x_slack_signature,
            x_slack_request_timestamp,
            os.getenv("SLACK_SIGNING_SECRET")
        ):
            raise HTTPException(status_code=403, detail="Invalid signature")

    # Handle URL verification challenge
    if body.get("type") == "url_verification":
        return {"challenge": body.get("challenge")}

    # Handle events
    if body.get("type") == "event_callback":
        event = body.get("event", {})

        # Ignore bot messages to prevent loops
        if event.get("bot_id"):
            return {"status": "ignored_bot_message"}

        # Handle message events
        if event.get("type") == "message" and event.get("text"):
            text = event.get("text")
            channel = event.get("channel")

            # Analyze the message
            analysis = slack_analyzer.analyze_ticket(
                text, include_explanation=False)

            # Post results to a designated support channel
            support_channel = os.getenv("SLACK_SUPPORT_CHANNEL", channel)

            if slack_analyzer.slack_client:
                # Post formatted analysis
                slack_analyzer.post_to_slack(support_channel, analysis)

            return {"status": "processed", "analysis": analysis}

    return {"status": "ok"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ticket(request: AnalysisRequest):
    """
    Analyze a ticket/message for both urgency and emotion.
    Optionally post results to Slack.

    This endpoint combines both classifiers and returns a comprehensive analysis.
    Use this for manual API calls or webhook integrations.

    Args:
        request: AnalysisRequest with text and optional Slack channel

    Returns:
        AnalysisResponse with combined urgency and emotion analysis
    """
    try:
        # Perform analysis
        analysis = slack_analyzer.analyze_ticket(
            request.text,
            include_explanation=request.include_explanation
        )

        # Generate markdown report
        markdown_report = slack_analyzer.format_markdown_report(analysis)
        analysis["report"] = markdown_report

        # Optionally post to Slack
        if request.slack_channel and slack_analyzer.slack_client:
            slack_analyzer.post_to_slack(request.slack_channel, analysis)

        return AnalysisResponse(**analysis)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )

def verify_slack_signature(
    body: bytes,
    signature: str,
    timestamp: str,
    signing_secret: str
) -> bool:
    """
    Verify that requests are coming from Slack

    Args:
        body: Request body bytes
        signature: X-Slack-Signature header
        timestamp: X-Slack-Request-Timestamp header
        signing_secret: Your Slack app's signing secret

    Returns:
        True if signature is valid
    """
    # Prevent replay attacks
    import time
    if abs(time.time() - int(timestamp)) > 60 * 5:
        return False

    # Compute expected signature
    sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
    expected_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()

    # Compare signatures
    return hmac.compare_digest(expected_signature, signature)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "urgency_model": MODEL_NAME_1,
        "emotion_model": MODEL_NAME_2,
        "emotion_model_type": "PEFT LoRA (RoBERTa-base + adapter)",
        "slack_enabled": slack_analyzer.slack_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
