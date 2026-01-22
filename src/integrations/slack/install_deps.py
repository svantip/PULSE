#!/usr/bin/env python3
"""
Installation script for Slack integration dependencies
Ensures all required packages are installed
"""
import subprocess
import sys


def check_python_version():
    """Check if Python version is 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")


def install_requirements():
    """Install all requirements from requirements.txt"""
    print("\n📦 Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def verify_imports():
    """Verify that all critical imports work"""
    print("\n🔍 Verifying imports...")

    packages = [
        ("slack_sdk", "Slack SDK"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("pydantic", "Pydantic"),
    ]

    all_ok = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - Failed to import")
            all_ok = False

    return all_ok


def main():
    """Main installation process"""
    print("="*60)
    print("  🚀 PULSE Slack Integration - Dependency Installer")
    print("="*60)
    print()

    # Check Python version
    check_python_version()

    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Verify imports
    if not verify_imports():
        print("\n⚠️  Some imports failed. Try installing again:")
        print("   pip install -r requirements.txt --force-reinstall")
        sys.exit(1)

    # Success!
    print("\n" + "="*60)
    print("  ✅ Installation Complete!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Configure Slack credentials in .env file")
    print("2. Run: python example_usage.py (test without Slack)")
    print("3. Run: python controller.py (start API server)")
    print("4. Run: python test_slack_integration.py (test API)")
    print("5. See SLACK_INTEGRATION.md for full Slack setup")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
