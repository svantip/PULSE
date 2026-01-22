# 🚀 Quick Start Guide - Urgency Classification UI

## Step 1: Install Dependencies

```bash
pip install gradio requests
```

## Step 2: Start the Services

### Option A: Use the startup script (Easiest)

```bash
chmod +x start_services.sh
./start_services.sh
```

### Option B: Manual start (Two terminals)

**Terminal 1 - API Server:**

```bash
python controller.py
```

**Terminal 2 - Gradio UI:**

```bash
python app.py
```

## Step 3: Access the UI

Open your browser and go to:

- **Gradio UI**: http://localhost:7860
- **API Docs**: http://localhost:8000/docs

## Step 4: Test It!

1. Enter text in the input box
2. Click "🚀 Classify"
3. See the results with:
   - Urgency level (LOW/MEDIUM/HIGH)
   - Confidence scores
   - Important keywords
   - Explanation

## What You Get

### ✅ Prediction

- Urgency level classification
- Confidence percentage
- Visual probability bars

### 🔍 Explainability

- **Important Tokens**: Which words influenced the decision
- **Urgency Indicators**: Specific keywords found
- **Reasoning**: Why the model made this classification

## Example Usage

**Input:**

```
System is completely down and users cannot access the application.
This is affecting production.
```

**Output:**

- 🔴 **HIGH URGENCY** (95% confidence)
- Important words: `system`, `down`, `cannot`, `production`
- Indicators: down (high), cannot access (high), production (high)
- Reasoning: "The model is highly confident (95.0%) in this classification. High urgency keywords detected: down, cannot access, production."

## Troubleshooting

**Problem**: UI shows "Cannot connect to API"
**Solution**: Make sure `controller.py` is running in another terminal

**Problem**: Port already in use
**Solution**:

- For API: Edit `controller.py`, change `port=8000` to another port
- For UI: Edit `app.py`, change `server_port=7860` to another port

**Problem**: Module not found
**Solution**: `pip install -r requirements.txt`

## Testing the API Directly

```bash
# Using curl
curl -X POST "http://localhost:8000/urgency/predict" \
  -H "Content-Type: application/json" \
  -d '{"message": "Production database is down"}'

# Or run the test script
python test_api.py
```

---

That's it! You're ready to classify urgency with explainability! 🎉
