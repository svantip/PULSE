"""
Simple example of using the Slack Ticket Analyzer directly in Python
without needing to set up Slack webhooks.
"""
from slack_service import SlackTicketAnalyzer


def main():
    print("="*60)
    print("🎫 Slack Ticket Analyzer - Standalone Example")
    print("="*60)
    print()

    # Initialize analyzer (no Slack token needed for local analysis)
    print("📚 Loading models...")
    analyzer = SlackTicketAnalyzer()
    print("✅ Models loaded successfully!")
    print()

    # Test messages
    test_messages = [
        "URGENT: Production server is down! Customers cannot login!",
        "Thank you for the quick help! Everything works perfectly now.",
        "I'm so frustrated! This is the third time my payment has failed.",
        "Can you help me understand how to reset my password?",
        "CRITICAL: Database backup failed last night. Possible data loss!"
    ]

    # Analyze each message
    for i, message in enumerate(test_messages, 1):
        print("="*60)
        print(f"Test Case {i}/{len(test_messages)}")
        print("="*60)
        print()

        # Analyze the ticket
        analysis = analyzer.analyze_ticket(message, include_explanation=False)

        # Print results
        print(f"📝 Message:")
        print(f'   "{message}"')
        print()

        urgency = analysis['urgency']
        emotion = analysis['emotion']
        priority = analysis['priority_flag']

        urgency_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}
        emotion_emoji = {
            "happy": "😊", "sad": "😢", "angry": "😠",
            "frustrated": "😤", "neutral": "😐", "satisfied": "😌"
        }

        print(
            f"{urgency_emoji.get(urgency['level'].lower(), '❓')} URGENCY: {urgency['level'].upper()}")
        print(f"   Confidence: {urgency['confidence']*100:.1f}%")
        print()

        print(
            f"{emotion_emoji.get(emotion['type'].lower(), '🎭')} EMOTION: {emotion['type'].upper()}")
        print(f"   Confidence: {emotion['confidence']*100:.1f}%")
        print()

        if priority['escalate']:
            print(f"🚨 PRIORITY: {priority['level'].upper()} - ESCALATE!")
        else:
            print(f"✅ PRIORITY: {priority['level'].upper()}")
        print(f"💡 {priority['reason']}")
        print()

        # Show all scores
        print("📊 Detailed Scores:")
        print()
        print("   Urgency breakdown:")
        for label, score in sorted(urgency['all_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 15)
            print(f"   {label.upper():8} {bar:15} {score*100:5.1f}%")

        print()
        print("   Emotion breakdown:")
        for label, score in sorted(emotion['all_scores'].items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 15)
            print(f"   {label.upper():11} {bar:15} {score*100:5.1f}%")

        print()

        # Generate markdown report
        if i == 1:  # Only show report for first message
            print("📄 Markdown Report Preview:")
            print("-" * 60)
            report = analyzer.format_markdown_report(analysis)
            # Print first 500 chars of report
            print(report[:500])
            print("...")
            print("-" * 60)
            print()

    print("="*60)
    print("✨ Analysis Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. To integrate with Slack, see SLACK_INTEGRATION.md")
    print("2. To start the API server: python controller.py")
    print("3. To run full test suite: python test_slack_integration.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
