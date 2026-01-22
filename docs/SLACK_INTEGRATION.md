# Slack Integration Setup Guide

This guide will help you integrate your urgency and emotion classifiers with Slack to automatically analyze incoming messages.

## 🎯 Overview

The integration allows you to:

- Receive incoming Slack messages via webhook
- Automatically classify urgency level (high/medium/low)
- Automatically detect customer emotion (happy/sad/angry/frustrated/satisfied/neutral)
- Generate formatted reports for support teams
- Post analysis results to designated Slack channels

## 📋 Prerequisites

- Python 3.8+
- A Slack workspace where you have admin permissions
- The PULSE application running

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **"Create New App"**
3. Choose **"From scratch"**
4. Enter app name: `Ticket Analyzer` (or your preferred name)
5. Select your workspace
6. Click **"Create App"**

### Step 3: Configure Bot Token Scopes

1. In your app settings, go to **"OAuth & Permissions"**
2. Scroll to **"Scopes" → "Bot Token Scopes"**
3. Add the following scopes:
   - `chat:write` - Post messages
   - `channels:history` - View messages in public channels
   - `channels:read` - View basic channel info
   - `groups:history` - View messages in private channels (optional)
   - `im:history` - View messages in direct messages (optional)

### Step 4: Install App to Workspace

1. Scroll up to **"OAuth Tokens for Your Workspace"**
2. Click **"Install to Workspace"**
3. Review permissions and click **"Allow"**
4. Copy the **"Bot User OAuth Token"** (starts with `xoxb-`)

### Step 5: Get Signing Secret

1. Go to **"Basic Information"** in your app settings
2. Scroll to **"App Credentials"**
3. Copy the **"Signing Secret"**

### Step 6: Configure Environment Variables

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your credentials:
   ```bash
   SLACK_BOT_TOKEN=xoxb-your-token-here
   SLACK_SIGNING_SECRET=your-signing-secret-here
   SLACK_SUPPORT_CHANNEL=support-tickets
   ```

### Step 7: Set Up Event Subscriptions

⚠️ **Important:** Your server must be running and publicly accessible for this step.

1. In your app settings, go to **"Event Subscriptions"**
2. Toggle **"Enable Events"** to ON
3. Enter your **Request URL**:
   - Local development (using ngrok): `https://your-ngrok-url.ngrok.io/slack/events`
   - Production: `https://your-domain.com/slack/events`
4. Slack will send a verification challenge - your server will respond automatically
5. Once verified, scroll to **"Subscribe to bot events"**
6. Add these bot events:
   - `message.channels` - Messages in public channels
   - `message.groups` - Messages in private channels (optional)
   - `message.im` - Direct messages (optional)
7. Click **"Save Changes"**
8. **Reinstall your app** (Slack will prompt you)

### Step 8: Start the Server

```bash
python controller.py
```

Or use uvicorn:

```bash
uvicorn controller:app --host 0.0.0.0 --port 8000 --reload
```

### Step 9: Test the Integration

#### Option 1: Send a message in Slack

1. Invite the bot to a channel: `/invite @Ticket Analyzer`
2. Send a test message: `URGENT: Production server is down!`
3. Check your support channel for the analysis report

#### Option 2: Use the API directly

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "URGENT: Production server is down!",
    "slack_channel": "support-tickets",
    "include_explanation": true
  }'
```

## 🔧 Local Development with ngrok

If developing locally, use ngrok to expose your server:

1. Install ngrok: [https://ngrok.com/download](https://ngrok.com/download)
2. Start your server: `python controller.py`
3. In another terminal: `ngrok http 8000`
4. Copy the HTTPS URL from ngrok (e.g., `https://abc123.ngrok.io`)
5. Use this URL in Slack Event Subscriptions: `https://abc123.ngrok.io/slack/events`

## 📊 API Endpoints

### `/slack/events` (POST)

Webhook endpoint for Slack Events API. Receives and processes Slack messages automatically.

**Handled automatically by Slack.**

### `/analyze` (POST)

Manual analysis endpoint. Use this for API integrations or testing.

**Request:**

```json
{
  "text": "Customer is very angry, need immediate help!",
  "slack_channel": "support-tickets", // optional
  "include_explanation": true
}
```

**Response:**

```json
{
  "timestamp": "2026-01-22T10:30:00",
  "text": "Customer is very angry, need immediate help!",
  "urgency": {
    "level": "high",
    "confidence": 0.92,
    "all_scores": {
      "high": 0.92,
      "medium": 0.06,
      "low": 0.02
    }
  },
  "emotion": {
    "type": "angry",
    "confidence": 0.88,
    "all_scores": {
      "angry": 0.88,
      "frustrated": 0.1,
      "neutral": 0.02
    }
  },
  "priority_flag": {
    "level": "high",
    "escalate": true,
    "reason": "High urgency with angry emotion"
  },
  "report": "# Ticket Analysis Report\n..."
}
```

### `/urgency/predict` (POST)

Urgency classification only.

**Request:**

```json
{
  "message": "Server is down"
}
```

### `/emotion/predict` (POST)

Emotion classification only.

**Request:**

```json
{
  "message": "I'm so frustrated with this issue"
}
```

### `/health` (GET)

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model": "svantip123/urgency_classificator",
  "slack_enabled": true
}
```

## 📈 Slack Message Format

When a message is analyzed, the bot posts a rich formatted report to your support channel:

```
🎫 Ticket Analysis Report
━━━━━━━━━━━━━━━━━━━━━━━━

🚨 URGENT TICKET - IMMEDIATE ATTENTION REQUIRED 🚨

━━━━━━━━━━━━━━━━━━━━━━━━

🚨 Urgency Level:              😠 Customer Emotion:
HIGH (92.3%)                   ANGRY (87.5%)

📝 Message:
"Production server is completely down! Customers cannot access the site!"

━━━━━━━━━━━━━━━━━━━━━━━━

📊 Urgency Breakdown:
HIGH:   ▓▓▓▓▓▓▓▓▓▓ 92%
MEDIUM: ▓░░░░░░░░░  6%
LOW:    ░░░░░░░░░░  2%

💭 Emotion Breakdown:
ANGRY:      ▓▓▓▓▓▓▓▓▓░ 88%
FRUSTRATED: ▓░░░░░░░░░ 10%
NEUTRAL:    ░░░░░░░░░░  2%

━━━━━━━━━━━━━━━━━━━━━━━━

💡 Recommendation:
High urgency with angry emotion

🕐 Analyzed at 2026-01-22T10:30:00
```

## 🎨 Priority Levels

The system automatically determines priority based on urgency and emotion:

| Urgency | Emotion          | Priority | Escalate |
| ------- | ---------------- | -------- | -------- |
| High    | Any              | High     | ✅ Yes   |
| Medium  | Angry/Frustrated | High     | ✅ Yes   |
| Medium  | Other            | Medium   | ❌ No    |
| Low     | Any              | Normal   | ❌ No    |

## 🔐 Security Best Practices

1. **Always verify Slack signatures in production:**
   - Set `SLACK_SIGNING_SECRET` in your `.env` file
   - The app automatically verifies requests

2. **Use HTTPS in production:**
   - Slack requires HTTPS for webhook URLs
   - Use a reverse proxy (nginx) or hosting platform with SSL

3. **Restrict bot permissions:**
   - Only grant necessary OAuth scopes
   - Review permissions regularly

4. **Keep tokens secret:**
   - Never commit `.env` file to git
   - Use environment variables or secret management

## 🐛 Troubleshooting

### Bot not responding to messages

- Ensure bot is invited to the channel: `/invite @Ticket Analyzer`
- Check Event Subscriptions are enabled
- Verify bot has `message.channels` event subscription
- Check server logs for errors

### "url_verification" fails

- Ensure server is running and publicly accessible
- Check firewall settings
- Verify the URL is correct (should end with `/slack/events`)
- Check server logs for the challenge request

### Messages analyzed but not posted to Slack

- Verify `SLACK_BOT_TOKEN` is set correctly
- Check `SLACK_SUPPORT_CHANNEL` exists and bot has access
- Ensure bot has `chat:write` permission

### Analysis seems incorrect

- Check that models are loaded correctly
- View detailed logs in console
- Test with `/analyze` endpoint for debugging
- Verify model paths if using custom models

### Signature verification fails

- Ensure `SLACK_SIGNING_SECRET` matches your app settings
- Check system time is synchronized
- Verify headers are being passed correctly

## 📝 Example Use Cases

### 1. Support Ticket Triage

Set up a dedicated #support-tickets channel where customers can submit tickets. The bot automatically:

- Analyzes urgency and emotion
- Posts detailed reports to #support-internal
- Flags high-priority tickets for immediate attention

### 2. Customer Feedback Analysis

Monitor #customer-feedback channel to:

- Identify frustrated or angry customers
- Track satisfaction levels
- Escalate urgent issues automatically

### 3. Multi-Channel Support

Subscribe to multiple channels:

- #general-support
- #technical-support
- #billing-support

All get analyzed and routed to appropriate support teams.

## 🔄 Workflow Example

1. **Customer sends message** in #support:

   > "The payment system crashed and I lost my transaction!"

2. **Bot receives event** via webhook

3. **Analysis performed:**
   - Urgency: HIGH (95%)
   - Emotion: FRUSTRATED (82%)
   - Priority: HIGH + ESCALATE

4. **Report posted** to #support-internal with:
   - Color-coded urgency indicator
   - Emotion analysis
   - Message context
   - Escalation recommendation

5. **Support team** sees the report and responds immediately

## 📚 Additional Resources

- [Slack API Documentation](https://api.slack.com/docs)
- [Slack Block Kit Builder](https://app.slack.com/block-kit-builder)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review server logs for detailed error messages
3. Test endpoints individually using curl or Postman
4. Verify all environment variables are set correctly

## 📄 License

This integration is part of the PULSE project.
