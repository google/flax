"""Simple configuration using dataclasses instead of ml_collections."""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    learning_rate: float = 0.001
    latents: int = 20
    batch_size: int = 128
    num_epochs: int = 30

def get_default_config() -> TrainingConfig:
    """Get the default configuration."""
    return TrainingConfig()