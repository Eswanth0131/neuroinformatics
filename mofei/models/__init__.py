MODELS = {}

def register(name):
    """Decorator to register a model class by name."""
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

def get_model(name, **kwargs):
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name](**kwargs)

# Import all model files so they self-register
from . import unet
from . import unet_v2