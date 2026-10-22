"""Text-prompted object masking — say what to track instead of clicking it.

Replaces the 4-corner SAM box picker in generate_mask.py for the common case
where you can just name the object ("black spoon", "wooden beam").

Backends
--------
`sam3`   facebook/sam3 via transformers. One model, text straight to instance
         masks. ~0.22 s per prompt on a 4090 once warm. The repo is GATED:
         accept the licence at https://huggingface.co/facebook/sam3 and have a
         token in $HF_HOME/token (or $HF_TOKEN).
`gdino`  IDEA-Research/grounding-dino-* for a text-conditioned box, then the
         existing SAM box path for the mask. Ungated, so this is the fallback
         when SAM3 access is missing.

Both are cached per process: the weights never change within a run, and
reloading them was a multi-second stall on every re-register.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

_CACHE = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_sam3(model_id: str):
    key = ("sam3", model_id)
    if key not in _CACHE:
        import transformers
        t0 = time.time()
        proc = transformers.Sam3Processor.from_pretrained(model_id)
        model = transformers.Sam3Model.from_pretrained(model_id).to(_device()).eval()
        _CACHE[key] = (proc, model)
        print(f"[text_mask] loaded {model_id} in {time.time()-t0:.1f}s "
              f"(cached for this process)", flush=True)
    return _CACHE[key]


def _load_gdino(model_id: str):
    key = ("gdino", model_id)
    if key not in _CACHE:
        import transformers
        t0 = time.time()
        proc = transformers.AutoProcessor.from_pretrained(model_id)
        model = transformers.AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(_device()).eval()
        _CACHE[key] = (proc, model)
        print(f"[text_mask] loaded {model_id} in {time.time()-t0:.1f}s "
              f"(cached for this process)", flush=True)
    return _CACHE[key]


def _cpu_default_tensor():
    """FoundationPose installs torch.cuda.FloatTensor as the default tensor
    type; HF processors build CPU tensors and blow up under it. Same swap the
    existing SAM path uses."""
    class _Ctx:
        def __enter__(self):
            self.was_cuda = torch.tensor(0.).is_cuda
            if self.was_cuda:
                torch.set_default_tensor_type(torch.FloatTensor)
            return self
        def __exit__(self, *exc):
            if self.was_cuda:
                torch.set_default_tensor_type(torch.cuda.FloatTensor)
            return False
    return _Ctx()


def mask_from_text_sam3(rgb: np.ndarray, prompt: str, *,
                        model_id: str = "facebook/sam3",
                        threshold: float = 0.3,
                        mask_threshold: float = 0.5,
                        min_score: float = 0.0,
                        ) -> Tuple[Optional[np.ndarray], float]:
    """Return (mask uint8 HxW, score) for the highest-scoring instance, or
    (None, 0.0) if nothing matched."""
    H, W = rgb.shape[:2]
    proc, model = _load_sam3(model_id)
    with _cpu_default_tensor():
        inputs = proc(images=Image.fromarray(rgb), text=prompt,
                      return_tensors="pt").to(_device())
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_instance_segmentation(
        out, threshold=threshold, mask_threshold=mask_threshold,
        target_sizes=[(H, W)])[0]
    scores = res["scores"]
    if len(scores) == 0:
        return None, 0.0
    best = int(torch.as_tensor(scores).argmax())
    score = float(torch.as_tensor(scores)[best])
    if score < min_score:
        return None, score
    mask = res["masks"][best].detach().cpu().numpy()
    return (mask > 0).astype(np.uint8), score


def mask_from_text_gdino(rgb: np.ndarray, prompt: str, *,
                         model_id: str = "IDEA-Research/grounding-dino-base",
                         box_threshold: float = 0.25,
                         text_threshold: float = 0.25,
                         ) -> Tuple[Optional[np.ndarray], float]:
    """Grounding DINO for the box, then the existing SAM box path for the
    mask."""
    from generate_mask import segment_with_box, BoundingBox
    H, W = rgb.shape[:2]
    proc, model = _load_gdino(model_id)
    # Grounding DINO wants lowercase, period-terminated prompts.
    text = prompt.strip().lower()
    if not text.endswith("."):
        text += "."
    pil = Image.fromarray(rgb)
    with _cpu_default_tensor():
        inputs = proc(images=pil, text=text, return_tensors="pt").to(_device())
    with torch.inference_mode():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], threshold=box_threshold,
        text_threshold=text_threshold, target_sizes=[(H, W)])[0]
    if len(res["scores"]) == 0:
        return None, 0.0
    best = int(torch.as_tensor(res["scores"]).argmax())
    score = float(torch.as_tensor(res["scores"])[best])
    x0, y0, x1, y1 = [int(v) for v in res["boxes"][best].tolist()]
    with _cpu_default_tensor():
        mask = segment_with_box(pil, BoundingBox(x0, y0, x1, y1),
                                polygon_refinement=True)
    if mask is None:
        return None, score
    return (mask > 0).astype(np.uint8), score


def mask_from_text(rgb: np.ndarray, prompt: str, backend: str = "sam3",
                   **kw) -> Tuple[Optional[np.ndarray], float]:
    if backend == "sam3":
        return mask_from_text_sam3(rgb, prompt, **kw)
    if backend == "gdino":
        return mask_from_text_gdino(rgb, prompt, **kw)
    raise ValueError(f"unknown text-mask backend: {backend}")
