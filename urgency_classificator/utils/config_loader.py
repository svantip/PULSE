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
        # Default to config.yaml in the project root (one level up from utils/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "..", "config.yaml")

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
        raise ValueError(
            "LABELS not found in configuration. Please define urgency labels in config.yaml")

    return config['LABELS']


def get_models(config: Dict[str, Any] = None) -> List[Dict[str, str]]:
    """
    Get list of models to compare from config.

    Args:
        config: Configuration dictionary. If None, loads from default location.

    Returns:
        List of model configurations with NAME and ALIAS
    """
    if config is None:
        config = load_config()

    if 'MODELS' not in config:
        raise ValueError(
            "MODELS not found in configuration. Please define models in config.yaml")

    return config['MODELS']


def get_model_name(config: Dict[str, Any] = None, alias: str = None) -> str:
    """
    Get model name from config by alias, or return first model if no alias specified.

    Args:
        config: Configuration dictionary. If None, loads from default location.
        alias: Model alias to retrieve. If None, returns first model.

    Returns:
        Model name
    """
    if config is None:
        config = load_config()

    models = get_models(config)

    if alias:
        for model in models:
            if model.get('ALIAS') == alias:
                return model['NAME']
        raise ValueError(
            f"Model with alias '{alias}' not found in configuration")

    # Return first model if no alias specified
    return models[0]['NAME']


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
