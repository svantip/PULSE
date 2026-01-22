"""
Test script to verify the urgency classification API with explainability
"""
import requests
import json

API_URL = "http://localhost:8000/urgency/predict"


def test_classification():
    """Test the classification endpoint with different urgency levels."""

    test_cases = [
        {
            "name": "High Urgency",
            "text": "Production server is down! All users are affected. Need immediate assistance!"
        },
        {
            "name": "Medium Urgency",
            "text": "We're experiencing intermittent errors when uploading files. It's affecting some users."
        },
        {
            "name": "Low Urgency",
            "text": "Can someone help me understand how to reset my password? No rush."
        }
    ]

    print("🧪 Testing Urgency Classification API with Explainability\n")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['name']}")
        print(f"Input: \"{test_case['text'][:60]}...\"")
        print("-" * 80)

        try:
            response = requests.post(
                API_URL,
                json={"message": test_case["text"]},
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                print(f"✅ Status: Success")
                print(f"🎯 Prediction: {result['predicted_class'].upper()}")
                print(f"📊 Confidence: {result['confidence']*100:.2f}%")

                print(f"\n📈 All Probabilities:")
                for label, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(prob * 30)
                    print(f"  {label.upper():8} {prob*100:5.2f}% {bar}")

                if 'explanation' in result:
                    exp = result['explanation']

                    print(f"\n🔍 Explainability:")

                    if 'important_tokens' in exp and exp['important_tokens']:
                        print(f"  Top Keywords (by attention):")
                        for token, score in exp['important_tokens'][:5]:
                            print(f"    • {token}: {score:.4f}")

                    if 'urgency_indicators' in exp and exp['urgency_indicators']:
                        print(f"  \n  Urgency Indicators Found:")
                        for indicator in exp['urgency_indicators'][:5]:
                            print(f"    • {indicator}")

                    if 'reasoning' in exp:
                        print(f"\n  💡 Reasoning:")
                        print(f"    {exp['reasoning']}")

                print()

            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   {response.text}")

        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to API at {API_URL}")
            print(f"   Make sure the server is running: python controller.py")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n" + "=" * 80)
    print("✨ Testing complete!\n")


if __name__ == "__main__":
    test_classification()
