#!/usr/bin/env python3
"""
Wrapper script to run train_models.py with correct Python path.
Run this from the urgency_classificator/ directory.
"""
from scripts.training import train_models
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run the training script

if __name__ == "__main__":
    train_models.main()
