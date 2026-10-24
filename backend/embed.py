from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "sscd_disc_mixup.torchscript.pt"

_preprocess = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_model = None


def _load():
    global _model
    if _model is None:
        _model = torch.jit.load(str(WEIGHTS), map_location="cpu").eval()
    return _model


@torch.no_grad()
def embed(img: Image.Image) -> np.ndarray:
    """512-d L2-normalized SSCD descriptor."""
    x = _preprocess(img.convert("RGB")).unsqueeze(0)
    v = _load()(x)[0].numpy().astype("float32")
    return v / np.linalg.norm(v)
