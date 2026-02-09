MODELS = {}

def register(name):
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator

def get_model(name, **kwargs):
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODELS[name](**kwargs)

from . import unet
from . import unet_v2
from . import unet_v3
from . import unet_25d
from . import smp_unet
from . import nafnet
from . import swinir
