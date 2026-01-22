#!/usr/bin/env python3
"""
Test script for Slack integration
Tests the combined urgency and emotion analysis
"""
import requests
import json
from typing import Dict

# API Configuration
API_URL = "http://localhost:8000"

# Test messages with different urgency and emotion levels
TEST_MESSAGES = [
    {
        "text": "URGENT: Production server is completely down! Customers cannot access the site!",
        "expected_urgency": "high",
        "expected_emotion": "angry/frustrated"
    },
    {
        "text": "The application is running a bit slow today, can you take a look when you have time?",
        "expected_urgency": "low",
        "expected_emotion": "neutral"
    },
    {
        "text": "I'm extremely frustrated! This is the third time my order has failed!",
        "expected_urgency": "medium/high",
        "expected_emotion": "angry/frustrated"
    },
    {
        "text": "Thank you so much for the quick fix! Everything is working perfectly now!",
        "expected_urgency": "low",
        "expected_emotion": "happy/satisfied"
    },
    {
        "text": "The payment gateway is not responding. This is blocking our sales team.",
        "expected_urgency": "high",
        "expected_emotion": "neutral/frustrated"
    },
    {
        "text": "How do I reset my password? I can't remember it.",
        "expected_urgency": "low",
        "expected_emotion": "neutral"
    }
]


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed")
            print(f"   Status: {result.get('status')}")
            print(f"   Model: {result.get('model')}")
            print(f"   Slack enabled: {result.get('slack_enabled')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_urgency_only(text: str):
    """Test urgency classification only"""
    try:
        response = requests.post(
            f"{API_URL}/urgency/predict",
            json={"message": text},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_emotion_only(text: str):
    """Test emotion classification only"""
    try:
        response = requests.post(
            f"{API_URL}/emotion/predict",
            json={"message": text},
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def test_combined_analysis(text: str, include_explanation: bool = False):
    """Test combined analysis endpoint"""
    try:
        response = requests.post(
            f"{API_URL}/analyze",
            json={
                "text": text,
                "include_explanation": include_explanation
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"   ❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


def print_analysis_results(result: Dict, test_case: Dict):
    """Pretty print analysis results"""
    print(f"\n📝 Message: \"{result['text']}\"")
    print(
        f"   Expected: Urgency={test_case['expected_urgency']}, Emotion={test_case['expected_emotion']}")
    print()

    # Urgency
    urgency = result['urgency']
    urgency_emoji = {"high": "🚨", "medium": "⚠️", "low": "ℹ️"}
    print(
        f"   {urgency_emoji.get(urgency['level'].lower(), '❓')} URGENCY: {urgency['level'].upper()} ({urgency['confidence']*100:.1f}%)")
    for label, score in sorted(urgency['all_scores'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 10)
        print(f"      {label.upper()}: {bar} {score*100:.0f}%")
    print()

    # Emotion
    emotion = result['emotion']
    emotion_emoji = {
        "happy": "😊", "sad": "😢", "angry": "😠",
        "frustrated": "😤", "neutral": "😐", "satisfied": "😌"
    }
    print(
        f"   {emotion_emoji.get(emotion['type'].lower(), '🎭')} EMOTION: {emotion['type'].upper()} ({emotion['confidence']*100:.1f}%)")
    for label, score in sorted(emotion['all_scores'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 10)
        print(f"      {label.upper()}: {bar} {score*100:.0f}%")
    print()

    # Priority
    priority = result['priority_flag']
    if priority['escalate']:
        print(f"   🚨 PRIORITY: {priority['level'].upper()} - ESCALATE!")
    else:
        print(f"   ✅ PRIORITY: {priority['level'].upper()}")
    print(f"   💡 Reason: {priority['reason']}")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "="*60)
    print("SLACK INTEGRATION TEST SUITE")
    print("="*60)

    # Test health
    if not test_health_check():
        print("\n❌ Server is not running. Please start it first:")
        print("   python controller.py")
        return

    # Test combined analysis
    print("\n" + "="*60)
    print("Testing Combined Analysis (Urgency + Emotion)")
    print("="*60)

    passed = 0
    failed = 0

    for i, test_case in enumerate(TEST_MESSAGES, 1):
        print(f"\n📋 Test Case {i}/{len(TEST_MESSAGES)}")
        print("-" * 60)

        result = test_combined_analysis(
            test_case['text'], include_explanation=False)

        if result:
            print_analysis_results(result, test_case)
            passed += 1
        else:
            print(f"❌ Test case {i} failed")
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(TEST_MESSAGES)}")
    if failed > 0:
        print(f"❌ Failed: {failed}/{len(TEST_MESSAGES)}")
    else:
        print("🎉 All tests passed!")
    print("="*60)


def test_markdown_report():
    """Test markdown report generation"""
    print("\n" + "="*60)
    print("Testing Markdown Report Generation")
    print("="*60)

    test_text = "CRITICAL: Database backup failed! Data loss possible!"

    result = test_combined_analysis(test_text, include_explanation=True)

    if result and 'report' in result:
        print("\n📄 Generated Markdown Report:")
        print("-" * 60)
        print(result['report'])
        print("-" * 60)
        return True
    else:
        print("❌ Failed to generate markdown report")
        return False


if __name__ == "__main__":
    print("""
    🚀 Slack Integration Test Script
    
    This script tests the urgency and emotion classification endpoints.
    Make sure the server is running before executing this script:
    
        python controller.py
    
    or
    
        uvicorn controller:app --reload
    """)

    input("Press Enter to start tests...")

    # Run all tests
    run_all_tests()

    # Test markdown report
    print("\n")
    input("Press Enter to test markdown report generation...")
    test_markdown_report()

    print("\n✨ Testing complete!")
