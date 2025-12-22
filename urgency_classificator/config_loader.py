"""
Configuration loader for urgency classifier.
"""
import os
import yaml
from typing import Dict, Any, List


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default to config.yaml in the same directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_labels(config: Dict[str, Any] = None) -> List[str]:
    """
    Get urgency labels from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
        
    Returns:
        List of urgency labels
    """
    if config is None:
        config = load_config()
    
    if 'LABELS' not in config:
        raise ValueError("LABELS not found in configuration. Please define urgency labels in config.yaml")
    
    return config['LABELS']


def get_model_name(config: Dict[str, Any] = None) -> str:
    """
    Get model name from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
        
    Returns:
        Model name
    """
    if config is None:
        config = load_config()
    
    if 'MODEL_NAME' not in config:
        raise ValueError("MODEL_NAME not found in configuration. Please define model name in config.yaml")
    
    return config['MODEL_NAME']


def get_training_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get training configuration from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
        
    Returns:
        Training configuration dictionary
    """
    if config is None:
        config = load_config()
    
    return config.get('TRAINING', {})


def get_versioning_config(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get versioning configuration from config.
    
    Args:
        config: Configuration dictionary. If None, loads from default location.
        
    Returns:
        Versioning configuration dictionary
    """
    if config is None:
        config = load_config()
    
    return config.get('VERSIONING', {})
