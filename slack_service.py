"""
Slack Integration Service
Combines urgency and emotion classifiers and formats results for Slack
"""
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from emotion_classificator.predictor import EmotionPredictor
from urgency_classificator.scripts.inference.predictor import UrgencyPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlackTicketAnalyzer:
    """
    Combines urgency and emotion analysis for Slack messages/tickets
    """

    def __init__(
        self,
        slack_bot_token: Optional[str] = None
    ):
        """
        Initialize the analyzers

        Args:
            urgency_model_path: Path to urgency model (uses HuggingFace model if None)
            emotion_model_path: Path to emotion model (uses base model if None)
            slack_bot_token: Slack bot token for posting messages
        """
        logger.info("Initializing Slack Ticket Analyzer...")

        # Initialize classifiers
        
        self.urgency_predictor = UrgencyPredictor(
        model_name="svantip123/urgency_classificator")

        
        self.emotion_predictor = EmotionPredictor()


        # Initialize Slack client if token provided
        self.slack_client = None
        if slack_bot_token:
            self.slack_client = WebClient(token=slack_bot_token)
            logger.info("✓ Slack client initialized")

    def analyze_ticket(self, text: str, include_explanation: bool = True) -> Dict[str, Any]:
        """
        Analyze a ticket/message for both urgency and emotion

        Args:
            text: The ticket/message text
            include_explanation: Whether to include detailed explanations

        Returns:
            Combined analysis results
        """
        logger.info(f"Analyzing ticket: {text[:50]}...")

        # Get predictions
        urgency_result = self.urgency_predictor.predict(
            text, explain=include_explanation)
        emotion_result = self.emotion_predictor.predict(
            text, explain=include_explanation)

        # Combine results
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "urgency": {
                "level": urgency_result["urgency"],
                "confidence": urgency_result["confidence"],
                "all_scores": urgency_result["all_scores"]
            },
            "emotion": {
                "type": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "all_scores": emotion_result["all_scores"]
            }
        }

        # Add explanations if requested
        if include_explanation:
            if "explanation" in urgency_result:
                analysis["urgency"]["explanation"] = urgency_result["explanation"]
            if "explanation" in emotion_result:
                analysis["emotion"]["explanation"] = emotion_result["explanation"]

        # Add priority flag based on combination
        analysis["priority_flag"] = self._determine_priority(
            urgency_result["urgency"],
            emotion_result["emotion"]
        )

        return analysis

    def _determine_priority(self, urgency: str, emotion: str) -> Dict[str, Any]:
        """
        Determine priority level based on urgency and emotion combination

        Args:
            urgency: Urgency level (high/medium/low)
            emotion: Emotion type

        Returns:
            Priority information
        """
        # Define escalation conditions
        escalate = False
        priority_level = "normal"

        if urgency.lower() == "high":
            priority_level = "high"
            escalate = True
        elif urgency.lower() == "medium" and emotion.lower() in ["angry", "frustrated"]:
            priority_level = "high"
            escalate = True
        elif urgency.lower() == "medium":
            priority_level = "medium"

        return {
            "level": priority_level,
            "escalate": escalate,
            "reason": f"{urgency.capitalize()} urgency with {emotion.lower()} emotion"
        }

    def format_slack_message(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format analysis results as a rich Slack message block

        Args:
            analysis: Analysis results from analyze_ticket()

        Returns:
            Slack Block Kit formatted message
        """
        # Emoji mappings
        urgency_emoji = {
            "high": "🚨",
            "medium": "⚠️",
            "low": "ℹ️"
        }

        emotion_emoji = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "frustrated": "😤",
            "neutral": "😐",
            "satisfied": "😌"
        }

        urgency = analysis["urgency"]["level"]
        emotion = analysis["emotion"]["type"]
        priority = analysis["priority_flag"]

        # Build header
        if priority["escalate"]:
            header_text = f":rotating_light: *URGENT TICKET - IMMEDIATE ATTENTION REQUIRED* :rotating_light:"
            header_color = "#ff0000"
        elif priority["level"] == "medium":
            header_text = f"⚠️ *Medium Priority Ticket*"
            header_color = "#ff9900"
        else:
            header_text = f"ℹ️ *New Ticket*"
            header_color = "#36a64f"

        # Build blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "🎫 Ticket Analysis Report",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header_text
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*{urgency_emoji.get(urgency.lower(), '❓')} Urgency Level:*\n{urgency.upper()} ({analysis['urgency']['confidence']*100:.1f}%)"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*{emotion_emoji.get(emotion.lower(), '🎭')} Customer Emotion:*\n{emotion.capitalize()} ({analysis['emotion']['confidence']*100:.1f}%)"
                    }
                ]
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📝 Message:*\n_{analysis['text']}_"
                }
            },
            {
                "type": "divider"
            }
        ]

        # Add urgency breakdown
        urgency_bars = ""
        for label, score in sorted(analysis["urgency"]["all_scores"].items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 10)
            bar = "▓" * bar_length + "░" * (10 - bar_length)
            urgency_bars += f"{label.upper()}: {bar} {score*100:.0f}%\n"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*📊 Urgency Breakdown:*\n```{urgency_bars}```"
            }
        })

        # Add emotion breakdown
        emotion_bars = ""
        for label, score in sorted(analysis["emotion"]["all_scores"].items(), key=lambda x: x[1], reverse=True):
            bar_length = int(score * 10)
            bar = "▓" * bar_length + "░" * (10 - bar_length)
            emotion_bars += f"{label.upper()}: {bar} {score*100:.0f}%\n"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*💭 Emotion Breakdown:*\n```{emotion_bars}```"
            }
        })

        # Add recommendation
        blocks.extend([
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*💡 Recommendation:*\n{priority['reason']}"
                }
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"🕐 Analyzed at {analysis['timestamp']}"
                    }
                ]
            }
        ])

        return {
            "blocks": blocks,
            # Fallback text
            "text": f"Ticket Analysis: {urgency} urgency, {emotion} emotion"
        }

    def post_to_slack(self, channel: str, analysis: Dict[str, Any]) -> Optional[Dict]:
        """
        Post analysis results to a Slack channel

        Args:
            channel: Slack channel ID or name
            analysis: Analysis results from analyze_ticket()

        Returns:
            Slack API response or None on failure
        """
        if not self.slack_client:
            logger.error(
                "Slack client not initialized. Provide slack_bot_token.")
            return None

        try:
            message = self.format_slack_message(analysis)
            response = self.slack_client.chat_postMessage(
                channel=channel,
                **message
            )
            logger.info(f"✓ Message posted to {channel}")
            return response
        except SlackApiError as e:
            logger.error(f"Error posting to Slack: {e.response['error']}")
            return None

    def format_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """
        Format analysis as a simple markdown report (for non-Slack use)

        Args:
            analysis: Analysis results from analyze_ticket()

        Returns:
            Markdown formatted string
        """
        urgency = analysis["urgency"]["level"]
        emotion = analysis["emotion"]["type"]
        priority = analysis["priority_flag"]

        report = f"""
# 🎫 Ticket Analysis Report

**Timestamp:** {analysis['timestamp']}

---

## 📋 Message
_{analysis['text']}_

---

## 🚨 Priority Assessment
- **Level:** {priority['level'].upper()}
- **Escalate:** {'YES ⚠️' if priority['escalate'] else 'NO'}
- **Reason:** {priority['reason']}

---

## 📊 Analysis Results

### Urgency Level: {urgency.upper()}
**Confidence:** {analysis['urgency']['confidence']*100:.1f}%

**All Urgency Scores:**
"""
        for label, score in sorted(analysis["urgency"]["all_scores"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            report += f"\n- {label.upper()}: {score*100:.1f}% {bar}"

        report += f"""

### Emotion Type: {emotion.upper()}
**Confidence:** {analysis['emotion']['confidence']*100:.1f}%

**All Emotion Scores:**
"""
        for label, score in sorted(analysis["emotion"]["all_scores"].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            report += f"\n- {label.upper()}: {score*100:.1f}% {bar}"

        report += "\n\n---\n"

        return report
