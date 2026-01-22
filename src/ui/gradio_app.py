import gradio as gr
import requests
import json
from typing import Dict, Tuple

# Configuration
URGENCY_API_URL = "http://localhost:8000/urgency/predict"
EMOTION_API_URL = "http://localhost:8000/emotion/predict"


def classify_emotion(text: str) -> Tuple[str, str, str]:
    """
    Call the emotion classification endpoint and return results with explainability.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (prediction, confidence scores, explanation)
    """
    if not text or not text.strip():
        return "⚠️ Please enter some text", "", ""

    try:
        # Make API request
        response = requests.post(
            EMOTION_API_URL,
            json={"message": text},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # Format prediction
            pred_class = result['predicted_class'].upper()
            confidence = result['confidence'] * 100

            # Emoji mapping for emotions
            emoji_map = {
                "HAPPY": "😊",
                "SAD": "😢",
                "ANGRY": "😠",
                "NEUTRAL": "😐",
                "FRUSTRATED": "😤",
                "SATISFIED": "😌"
            }

            prediction_text = f"{emoji_map.get(pred_class, '🎭')} **{pred_class}**\n\nConfidence: {confidence:.1f}%"

            # Format probabilities
            prob_text = "### All Emotion Probabilities:\n\n"
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                prob_text += f"**{label.upper()}**: {prob*100:.1f}% {bar}\n\n"

            # Format explanation
            explanation = result.get('explanation', {})
            if explanation:
                explain_text = "### 🔍 Explanation\n\n"

                # Important tokens
                if 'important_tokens' in explanation:
                    explain_text += "**Key Words (by importance):**\n\n"
                    for token, score in explanation['important_tokens'][:10]:
                        explain_text += f"- `{token}`: {score:.3f}\n"
                    explain_text += "\n"

                # Emotion indicators
                if 'emotion_indicators' in explanation:
                    indicators = explanation['emotion_indicators']
                    if indicators:
                        explain_text += "**Emotion Indicators Found:**\n\n"
                        for indicator in indicators[:5]:
                            explain_text += f"- {indicator}\n"
                        explain_text += "\n"

                # Model reasoning
                if 'reasoning' in explanation:
                    explain_text += f"**Model Reasoning:**\n\n{explanation['reasoning']}\n"
            else:
                explain_text = "No explanation available"

            return prediction_text, prob_text, explain_text

        else:
            error_msg = f"❌ Error {response.status_code}: {response.text}"
            return error_msg, "", ""

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running on http://localhost:8000", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


def classify_urgency(text: str) -> Tuple[str, str, str]:
    """
    Call the urgency classification endpoint and return results with explainability.

    Args:
        text: Input text to classify

    Returns:
        Tuple of (prediction, confidence scores, explanation)
    """
    if not text or not text.strip():
        return "⚠️ Please enter some text", "", ""

    try:
        # Make API request
        response = requests.post(
            URGENCY_API_URL,
            json={"message": text},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # Format prediction
            pred_class = result['predicted_class'].upper()
            confidence = result['confidence'] * 100

            # Emoji mapping
            emoji_map = {
                "LOW": "🟢",
                "MEDIUM": "🟡",
                "HIGH": "🔴"
            }

            prediction_text = f"{emoji_map.get(pred_class, '❓')} **{pred_class} URGENCY**\n\nConfidence: {confidence:.1f}%"

            # Format probabilities
            prob_text = "### All Class Probabilities:\n\n"
            for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                prob_text += f"**{label.upper()}**: {prob*100:.1f}% {bar}\n\n"

            # Format explanation
            explanation = result.get('explanation', {})
            if explanation:
                explain_text = "### 🔍 Explanation\n\n"

                # Model attention tokens (PRIMARY - what model actually used)
                if 'model_attention_tokens' in explanation:
                    explain_text += "**🧠 Model's Focus (Attention Weights):**\n\n"
                    explain_text += "_These are the words the model actually focused on when making the decision._\n\n"
                    for token, score in explanation['model_attention_tokens'][:10]:
                        explain_text += f"- `{token}`: {score:.3f}\n"
                    explain_text += "\n"
                # Fallback for old format
                elif 'important_tokens' in explanation:
                    explain_text += "**Key Words (by importance):**\n\n"
                    for token, score in explanation['important_tokens'][:10]:
                        explain_text += f"- `{token}`: {score:.3f}\n"
                    explain_text += "\n"

                # Rule-based keyword hints (SUPPLEMENTARY)
                if 'rule_based_keyword_hints' in explanation:
                    hints = explanation['rule_based_keyword_hints']
                    if hints:
                        explain_text += "**📋 Rule-Based Hints (Supplementary):**\n\n"
                        explain_text += "_Simple keyword matching - may differ from what the model focused on._\n\n"
                        for hint in hints[:5]:
                            explain_text += f"- {hint}\n"
                        explain_text += "\n"
                # Fallback for old format
                elif 'urgency_indicators' in explanation:
                    indicators = explanation['urgency_indicators']
                    if indicators:
                        explain_text += "**Urgency Indicators Found:**\n\n"
                        for indicator in indicators[:5]:
                            explain_text += f"- {indicator}\n"
                        explain_text += "\n"

                # Model reasoning
                if 'reasoning' in explanation:
                    explain_text += f"**💡 Summary:**\n\n{explanation['reasoning']}\n"
            else:
                explain_text = "No explanation available"

            return prediction_text, prob_text, explain_text

        else:
            error_msg = f"❌ Error {response.status_code}: {response.text}"
            return error_msg, "", ""

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Make sure the server is running on http://localhost:8000", "", ""
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""


# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎯 Text Classification System
        
        Classify support tickets and messages for **urgency** and **emotion**.
        Get detailed explanations and insights powered by transformer models.
        """
    )

    with gr.Tabs():
        # Urgency Classification Tab
        with gr.TabItem("🎯 Urgency Classification"):
            gr.Markdown(
                """
                ### Classify Urgency Level
                Enter a support ticket or message to classify its urgency level.
                - **LOW**: Questions, requests, minor issues
                - **MEDIUM**: Partial functionality issues, workarounds available  
                - **HIGH**: Production issues, system down, security incidents
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    urgency_text_input = gr.Textbox(
                        label="Enter Text to Classify",
                        placeholder="Type your support ticket or message here...",
                        lines=8,
                        max_lines=15
                    )

                    with gr.Row():
                        urgency_clear_btn = gr.Button(
                            "🗑️ Clear", variant="secondary")
                        urgency_submit_btn = gr.Button(
                            "🚀 Classify Urgency", variant="primary")

                    # Example inputs
                    gr.Examples(
                        examples=[
                            ["URGENT: Production database is down! All users are affected. Need immediate assistance!"],
                            ["Critical security breach detected in payment system. Customer data may be compromised."],
                            ["Application crashed and won't restart. Blocking all work for the development team."],
                            ["Website loading very slowly for some users. Error appears intermittently when uploading files."],
                            ["Email notifications are delayed by about 30 minutes. Users can still work normally."],
                            ["Some reports are showing incorrect data. Workaround available by refreshing twice."],
                            ["Can you help me reset my password when you have time?"],
                            ["I would like to request access to the analytics dashboard. No rush."],
                            ["Could someone provide documentation on how to export data? Thanks!"],
                            ["Feature request: It would be nice to have dark mode in the future."],
                        ],
                        inputs=urgency_text_input,
                        label="📝 Example Tickets"
                    )

                with gr.Column(scale=1):
                    urgency_prediction_output = gr.Markdown(label="Prediction")
                    urgency_probabilities_output = gr.Markdown(
                        label="Probabilities")
                    urgency_explanation_output = gr.Markdown(
                        label="Explanation")

            # Event handlers for urgency
            urgency_submit_btn.click(
                fn=classify_urgency,
                inputs=urgency_text_input,
                outputs=[urgency_prediction_output,
                         urgency_probabilities_output, urgency_explanation_output]
            )

            urgency_clear_btn.click(
                fn=lambda: ("", "", "", ""),
                outputs=[urgency_text_input, urgency_prediction_output,
                         urgency_probabilities_output, urgency_explanation_output]
            )

        # Emotion Classification Tab
        with gr.TabItem("🎭 Emotion Classification"):
            gr.Markdown(
                """
                ### Classify Emotional Tone
                Analyze the emotional tone and sentiment of customer messages.
                Detects emotions like: Happy, Sad, Angry, Frustrated, Satisfied, Neutral
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    emotion_text_input = gr.Textbox(
                        label="Enter Text to Classify",
                        placeholder="Type your message here...",
                        lines=8,
                        max_lines=15
                    )

                    with gr.Row():
                        emotion_clear_btn = gr.Button(
                            "🗑️ Clear", variant="secondary")
                        emotion_submit_btn = gr.Button(
                            "🎭 Classify Emotion", variant="primary")

                    # Example inputs
                    gr.Examples(
                        examples=[
                            ["Thank you so much! This really helped solve my problem. Great service!"],
                            ["Your team is amazing! Fixed my issue in minutes. Couldn't be happier!"],
                            ["Perfect! Everything is working now. Really appreciate the quick response."],
                            ["I've been waiting for 3 days and still no response. This is unacceptable."],
                            ["Extremely disappointed with the service. This is the third time the same issue occurred."],
                            ["This is ridiculous! How can such a basic feature not work properly?!"],
                            ["I've tried everything you suggested but nothing works. Getting really frustrated here."],
                            ["Still having the same problem after following all the steps. This is taking too long."],
                            ["Could you please provide information about the new features?"],
                            ["I have a question about the billing. Can someone clarify this?"],
                            ["The system is down again. Really tired of dealing with this every week."],
                            ["Thanks, I guess it's working now. Took longer than expected though."],
                        ],
                        inputs=emotion_text_input,
                        label="💬 Example Messages"
                    )

                with gr.Column(scale=1):
                    emotion_prediction_output = gr.Markdown(label="Prediction")
                    emotion_probabilities_output = gr.Markdown(
                        label="Probabilities")
                    emotion_explanation_output = gr.Markdown(
                        label="Explanation")

            # Event handlers for emotion
            emotion_submit_btn.click(
                fn=classify_emotion,
                inputs=emotion_text_input,
                outputs=[emotion_prediction_output,
                         emotion_probabilities_output, emotion_explanation_output]
            )

            emotion_clear_btn.click(
                fn=lambda: ("", "", "", ""),
                outputs=[emotion_text_input, emotion_prediction_output,
                         emotion_probabilities_output, emotion_explanation_output]
            )

    gr.Markdown(
        """
        ---
        💡 **Tips:**
        - More detailed text generally leads to better classification
        - The models are trained on support tickets in multiple languages
        - Switch between tabs to classify for urgency or emotion
        - Both classifiers provide explainability showing key words and reasoning
        """
    )

if __name__ == "__main__":
    print("\n🚀 Starting Classification UI...")
    print(f"📡 Urgency API: {URGENCY_API_URL}")
    print(f"📡 Emotion API: {EMOTION_API_URL}")
    print("\n⚠️  Make sure the FastAPI server is running!")
    print("   Run: python controller.py\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
