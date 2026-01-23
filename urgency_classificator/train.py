#!/usr/bin/env python3
from scripts.training import train_models
import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    train_models.main()
