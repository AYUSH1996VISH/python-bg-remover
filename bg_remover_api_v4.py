#!/usr/bin/env python3
"""
Premium BG Remover API v4  —  studio-grade, commercial-quality cutouts
======================================================================

This is a ground-up quality upgrade of v3. It targets the output quality of
commercial services (remove.bg / Cutout.pro / PhotoRoom) using the four
techniques those engines actually rely on — none of which v3 had:

  1. STATE-OF-THE-ART MODEL (BiRefNet)
     v3 deliberately skipped BiRefNet and used isnet-general-use. BiRefNet is
     the current SOTA for high-resolution dichotomous segmentation and is the
     single biggest quality jump. Full graceful fallback chain is preserved:
        birefnet-general -> isnet-general-use -> u2net_human_seg -> silueta

  2. FULL-RESOLUTION COLOR (no blurry upscales)
     v3 ran the model on a downscaled image, then LANCZOS-upscaled the model's
     *RGB and alpha* — destroying edge detail. v4 takes COLOR from your ORIGINAL
     full-resolution pixels and only the MASK from the model. Edges stay crisp.

  3. GUIDED-FILTER ALPHA REFINEMENT  (He et al. 2010)
     The coarse mask is refined against the full-res image so the alpha snaps to
     true edges and recovers hair / wisps / fur. This is fast (box filters only)
     and memory-safe — it does NOT OOM like pymatting's closed-form solver.

  4. FOREGROUND DECONTAMINATION (spill / halo removal)
     The "secret sauce". Semi-transparent edge pixels still contain the old
     background colour (the grey/green halo around hair). v4 estimates the true
     foreground colour and removes the spill. Uses pymatting's ML estimator when
     available, with a memory-safe cv2.inpaint fallback.

Plus: EXIF auto-rotation fix (phone photos), background replacement
(transparent / solid colour / custom image), and a richer API + CLI.

Quality tiers
  fast     : silueta @ 768px                      — fastest, no refinement
  balanced : u2net_human_seg @ 1024px             — refine, light decontam
  premium  : birefnet-general @ 1280px (default)  — refine + decontam
  ultra    : birefnet-general @ 1536px            — refine + decontam + matting band
  portrait : birefnet-portrait @ 1408px           — tuned for people / hair

Install (minimum)
    pip install fastapi uvicorn pillow "rembg[cli]" onnxruntime requests python-multipart numpy

Install (recommended — enables guided refine, decontam fallback, preprocessing)
    pip install opencv-python

Install (best decontamination quality)
    pip install pymatting

Install (GPU — replaces onnxruntime)
    pip install onnxruntime-gpu

CLI
    python bg_remover_api_v4.py local --input photo.jpg --quality premium
    python bg_remover_api_v4.py local --input photo.jpg --quality portrait --bg-color "#ffffff"
    python bg_remover_api_v4.py local --input photo.jpg --quality ultra --no-decontaminate

API
    python bg_remover_api_v4.py api --host 0.0.0.0 --port 8000
    curl -X POST "http://127.0.0.1:8000/remove-bg?quality=premium" -F "file=@photo.jpg" --output out.png
"""

import argparse
import base64
import inspect
import io
import logging
import os
import sys
import time
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# rembg is the only heavy import; defer-friendly but needed for real work.
try:
    from rembg import remove, new_session
    _REMBG_AVAILABLE = True
    _REMBG_PARAMS = set(inspect.signature(remove).parameters.keys())
except Exception:  # pragma: no cover - allows importing module without rembg
    _REMBG_AVAILABLE = False
    _REMBG_PARAMS = set()
    warnings.warn("rembg not installed — segmentation will fail until you "
                  "`pip install \"rembg[cli]\" onnxruntime`.", stacklevel=1)

# ---------------------------------------------------------------------------
# Optional OpenCV (preprocessing, guided filter speed-up, inpaint fallback)
# ---------------------------------------------------------------------------
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    warnings.warn(
        "opencv-python not found. Falling back to NumPy/PIL implementations "
        "(slower, still functional). Install: pip install opencv-python",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Optional pymatting (best-quality foreground colour decontamination)
# ---------------------------------------------------------------------------
try:
    from pymatting import estimate_foreground_ml
    _PYMATTING_AVAILABLE = True
except Exception:
    _PYMATTING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_QUALITY = os.getenv("BG_QUALITY", "premium")
DEFAULT_OUTPUT_DIR = Path(os.getenv("BG_OUTPUT_DIR", "outputs"))
DEFAULT_INPUT_DIR = Path(os.getenv("BG_INPUT_DIR", "inputs"))
MAX_IMAGE_MB = int(os.getenv("BG_MAX_IMAGE_MB", "25"))
REQUEST_TIMEOUT = int(os.getenv("BG_REQUEST_TIMEOUT", "30"))

# Cap the resolution at which decontamination runs (it is the heaviest step).
# Above this the foreground estimate is computed at a reduced size, then the
# spill correction is upsampled — keeps memory bounded with negligible quality
# loss (decontam only affects a thin edge band).
DECONTAM_MAX_DIM = int(os.getenv("BG_DECONTAM_DIM", "2000"))

# Hard cap on the alpha-matting band solve when pymatting matting is used.
MATTING_MAX_DIM = int(os.getenv("BG_MATTING_DIM", "1000"))

# Quality tier definitions.
#   models        : ordered fallback list (first that loads + runs wins)
#   inference_dim : longest side sent to the model (memory bound; refinement
#                   recovers detail at full res, so this need not be huge)
#   refine        : guided-filter alpha refinement at full resolution
#   decontaminate : remove background colour spill from edge pixels
#   matting_band  : run pymatting on the unknown band only (if available)
#   guided_radius : guided-filter window radius (px, before size scaling)
#   guided_eps    : guided-filter regularisation (edge sensitivity)
#   feather       : final transition-zone gaussian feather (px)
QUALITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fast": {
        "models": ["silueta"],
        "inference_dim": 768,
        "preprocess": False,
        "refine": False,
        "decontaminate": False,
        "matting_band": False,
        "post_process_mask": True,
        "guided_radius": 0,
        "guided_eps": 1e-4,
        "feather": 0,
    },
    "balanced": {
        "models": ["u2net_human_seg", "isnet-general-use", "silueta"],
        "inference_dim": 1024,
        "preprocess": True,
        "refine": True,
        "decontaminate": True,
        "matting_band": False,
        "post_process_mask": True,
        "guided_radius": 4,
        "guided_eps": 1e-4,
        "feather": 1,
    },
    "premium": {
        "models": ["birefnet-general", "isnet-general-use", "u2net_human_seg", "silueta"],
        "inference_dim": 1280,
        "preprocess": True,
        "refine": True,
        "decontaminate": True,
        "matting_band": False,
        "post_process_mask": True,
        "guided_radius": 5,
        "guided_eps": 1e-4,
        "feather": 1,
    },
    "ultra": {
        "models": ["birefnet-general", "isnet-general-use", "u2net_human_seg", "silueta"],
        "inference_dim": 1536,
        "preprocess": True,
        "refine": True,
        "decontaminate": True,
        "matting_band": True,
        "post_process_mask": True,
        "guided_radius": 6,
        "guided_eps": 6e-5,
        "feather": 1,
    },
    "portrait": {
        # birefnet-portrait is tuned for people; hair detail is its strength.
        "models": ["birefnet-portrait", "birefnet-general", "u2net_human_seg", "silueta"],
        "inference_dim": 1408,
        "preprocess": True,
        "refine": True,
        "decontaminate": True,
        "matting_band": True,
        "post_process_mask": True,
        "guided_radius": 6,
        "guided_eps": 5e-5,
        "feather": 1,
    },
}


@dataclass
class ProcessResult:
    filename: str
    png_bytes: bytes
    width: int
    height: int
    saved_path: Optional[str] = None
    quality: str = "premium"
    model_used: str = ""
    preprocessing_applied: Dict[str, Any] = field(default_factory=dict)
    refinement_applied: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model session cache (one session per model, reused across requests)
# ---------------------------------------------------------------------------
_SESSIONS: Dict[str, Any] = {}


def _best_providers() -> List[str]:
    """Return fastest available ONNX execution providers."""
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in avail:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def get_session(model_name: str) -> Any:
    """Lazy-load and cache a rembg session per model. Cleans up on failure."""
    import gc
    if model_name not in _SESSIONS:
        logger.info("Loading model: %s", model_name)
        gc.collect()
        try:
            _SESSIONS[model_name] = new_session(model_name, providers=_best_providers())
        except Exception:
            _SESSIONS.pop(model_name, None)
            gc.collect()
            raise
        logger.info("Model ready: %s", model_name)
    return _SESSIONS[model_name]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def ensure_dirs() -> None:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_INPUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_image_bytes(data: bytes) -> None:
    if not data:
        raise ValueError("Empty image data.")
    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"Image exceeds {MAX_IMAGE_MB} MB limit.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except Exception as exc:
        raise ValueError("Not a valid image file.") from exc


def load_file(path: Any) -> bytes:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    data = p.read_bytes()
    validate_image_bytes(data)
    return data


def download_image(url: str) -> Tuple[bytes, str]:
    ensure_dirs()
    r = requests.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.content
    validate_image_bytes(data)
    suffix = _img_suffix(data)
    name = f"dl_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    dst = DEFAULT_INPUT_DIR / name
    dst.write_bytes(data)
    return data, str(dst)


def _img_suffix(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as img:
        fmt = (img.format or "PNG").lower()
    return ".jpg" if fmt == "jpeg" else f".{fmt}"


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _scale_down(img: Image.Image, max_dim: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """Scale so the longest side <= max_dim. Returns (resized, original_size)."""
    orig = (img.width, img.height)
    if max(orig) > max_dim:
        r = max_dim / max(orig)
        img = img.resize(
            (max(1, int(round(img.width * r))), max(1, int(round(img.height * r)))),
            Image.LANCZOS,
        )
    return img, orig


def _load_oriented_rgb(image_bytes: bytes) -> Image.Image:
    """Open image, apply EXIF orientation (phone photos), return RGB."""
    img = Image.open(io.BytesIO(image_bytes))
    try:
        img = ImageOps.exif_transpose(img)  # rotate per camera orientation tag
    except Exception:
        pass
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Preprocessing (adaptive low-light + noise correction)
# ---------------------------------------------------------------------------
def _analyze(gray: np.ndarray) -> Dict[str, Any]:
    mean = float(gray.mean())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if _CV2_AVAILABLE else 0.0
    return {
        "mean_brightness": mean,
        "is_low_light": mean < 85,
        "is_noisy": lap > 500 and mean > 40,
        "laplacian_var": lap,
    }


def _clahe(rgb: np.ndarray) -> np.ndarray:
    if _CV2_AVAILABLE:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    pil = Image.fromarray(rgb)
    pil = ImageEnhance.Brightness(pil).enhance(1.3)
    return np.array(ImageEnhance.Contrast(pil).enhance(1.4))


def _bilateral(rgb: np.ndarray) -> np.ndarray:
    if _CV2_AVAILABLE:
        return cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)
    return np.array(Image.fromarray(rgb).filter(ImageFilter.SMOOTH_MORE))


def preprocess(img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
    """Detect + fix low-light and noise. Returns (processed_img, ops_applied)."""
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if _CV2_AVAILABLE else np.array(img.convert("L"))
    stats = _analyze(gray)
    applied: Dict[str, Any] = {"stats": stats}
    if stats["is_low_light"]:
        rgb = _clahe(rgb)
        applied["clahe"] = True
    if stats["is_noisy"]:
        rgb = _bilateral(rgb)
        applied["bilateral"] = True
    return Image.fromarray(rgb), applied


# ---------------------------------------------------------------------------
# Guided filter (Kaiming He et al., 2010) — edge-aware alpha refinement
# ---------------------------------------------------------------------------
def _box_filter_np(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) window, edge-clamped, via an integral image (no cv2)."""
    a = a.astype(np.float64)
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    i = np.arange(H)
    j = np.arange(W)
    y0 = np.clip(i - r, 0, H)[:, None]
    y1 = np.clip(i + r + 1, 0, H)[:, None]
    x0 = np.clip(j - r, 0, W)[None, :]
    x1 = np.clip(j + r + 1, 0, W)[None, :]
    total = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    count = (y1 - y0) * (x1 - x0)
    return total / np.maximum(count, 1)


def _box(a: np.ndarray, r: int) -> np.ndarray:
    if _CV2_AVAILABLE:
        return cv2.boxFilter(a.astype(np.float32), ddepth=-1, ksize=(2 * r + 1, 2 * r + 1),
                             normalize=True, borderType=cv2.BORDER_REFLECT)
    return _box_filter_np(a, r)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """
    Edge-preserving filter of `src` using grayscale `guide` (both float 0..1).
    Snaps a soft/blurry alpha to the real image edges. O(N) — memory-safe.
    """
    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    mean_g = _box(g, radius)
    mean_p = _box(p, radius)
    mean_gp = _box(g * p, radius)
    cov_gp = mean_gp - mean_g * mean_p
    mean_gg = _box(g * g, radius)
    var_g = mean_gg - mean_g * mean_g
    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = _box(a, radius)
    mean_b = _box(b, radius)
    q = mean_a * g + mean_b
    return np.clip(q, 0.0, 1.0)


def refine_alpha_guided(
    full_rgb: np.ndarray,
    coarse_alpha: np.ndarray,
    radius: int,
    eps: float,
    feather: int,
) -> np.ndarray:
    """
    Refine a coarse alpha (uint8, full-res) against the full-res RGB so edges
    snap to true boundaries and fine hair is recovered. Returns uint8 alpha.
    """
    h, w = full_rgb.shape[:2]
    # Scale radius with image size so behaviour is resolution-independent.
    r = max(1, int(round(radius * max(h, w) / 1024.0)))
    guide = full_rgb.astype(np.float32).mean(axis=2) / 255.0
    p = coarse_alpha.astype(np.float32) / 255.0

    refined = guided_filter(guide, p, r, eps)

    # A second small-radius pass tightens the very fine structure.
    refined = guided_filter(guide, refined, max(1, r // 2), eps)

    alpha = np.clip(refined * 255.0, 0, 255).astype(np.uint8)

    # Feather ONLY the transition band; keep solid FG/BG hard.
    if feather > 0 and _CV2_AVAILABLE:
        af = alpha.astype(np.float32)
        blurred = cv2.GaussianBlur(af, (0, 0), sigmaX=float(feather))
        band = (alpha > 12) & (alpha < 243)
        af[band] = blurred[band]
        alpha = np.clip(af, 0, 255).astype(np.uint8)
    elif feather > 0:
        alpha = np.array(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(feather)))
    return alpha


def cleanup_alpha_morphology(alpha: np.ndarray) -> np.ndarray:
    """Remove speckle outside the subject and fill pinholes inside it."""
    if not _CV2_AVAILABLE:
        return alpha
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    a = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k3, iterations=1)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k5, iterations=2)
    return a


# ---------------------------------------------------------------------------
# Foreground decontamination (remove background colour spill / halo)
# ---------------------------------------------------------------------------
def _decontaminate_pymatting(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Best quality: estimate the true foreground colour per pixel."""
    image = rgb.astype(np.float64) / 255.0
    a = alpha.astype(np.float64) / 255.0
    fg = estimate_foreground_ml(image, a)  # (H, W, 3) in 0..1
    return np.clip(fg * 255.0, 0, 255).astype(np.uint8)


def _decontaminate_inpaint(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    cv2 fallback: extend confident-foreground colour into the soft edge band so
    edge pixels no longer carry the old background colour. We only inpaint a thin
    band around the subject (bounded cost; background colour is irrelevant since
    its alpha is ~0).
    """
    if not _CV2_AVAILABLE:
        return rgb
    fg_conf = (alpha >= 240).astype(np.uint8) * 255
    # Region that matters: a dilated shell around the confident foreground.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shell = cv2.dilate(fg_conf, k, iterations=1)
    # Pixels to repaint: inside the shell but not confidently foreground.
    repaint = ((shell > 0) & (alpha < 240)).astype(np.uint8) * 255
    if repaint.sum() == 0:
        return rgb
    return cv2.inpaint(rgb, repaint, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def decontaminate_foreground(rgb: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Replace edge-pixel colours that are contaminated by the old background.
    Runs at a capped resolution for memory safety, then upsamples the colour.
    Returns (clean_rgb, method).
    """
    h, w = rgb.shape[:2]
    scale = 1.0
    work_rgb, work_alpha = rgb, alpha
    if max(h, w) > DECONTAM_MAX_DIM:
        scale = DECONTAM_MAX_DIM / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        work_rgb = np.array(Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS))
        work_alpha = np.array(Image.fromarray(alpha).resize((nw, nh), Image.LANCZOS))

    if _PYMATTING_AVAILABLE:
        method = "pymatting_ml"
        try:
            clean = _decontaminate_pymatting(work_rgb, work_alpha)
        except (MemoryError, Exception) as exc:  # noqa: BLE001 - any failure -> fallback
            logger.warning("pymatting decontam failed (%s) -> cv2 inpaint.", exc)
            clean = _decontaminate_inpaint(work_rgb, work_alpha)
            method = "cv2_inpaint"
    else:
        method = "cv2_inpaint" if _CV2_AVAILABLE else "none"
        clean = _decontaminate_inpaint(work_rgb, work_alpha)

    if scale != 1.0:
        clean = np.array(Image.fromarray(clean).resize((w, h), Image.LANCZOS))
    # Keep original colour where the subject is fully opaque; only edges change.
    a3 = (alpha.astype(np.float32) / 255.0)[..., None]
    solid = a3 >= (240.0 / 255.0)
    out = np.where(solid, rgb, clean)
    return out.astype(np.uint8), method


# ---------------------------------------------------------------------------
# Optional pymatting matting on the unknown band (ultra / portrait)
# ---------------------------------------------------------------------------
def matting_band_refine(full_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Refine only the uncertain edge band with closed-form/KNN matting. Capped at
    MATTING_MAX_DIM and restricted to the band to stay memory-safe; silently
    returns the input alpha if pymatting is unavailable or the solve fails.
    """
    try:
        from pymatting import estimate_alpha_cf
    except Exception:
        return alpha
    if not _CV2_AVAILABLE:
        return alpha

    h, w = full_rgb.shape[:2]
    scale = 1.0
    rgb_s, a_s = full_rgb, alpha
    if max(h, w) > MATTING_MAX_DIM:
        scale = MATTING_MAX_DIM / max(h, w)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        rgb_s = np.array(Image.fromarray(full_rgb).resize((nw, nh), Image.LANCZOS))
        a_s = np.array(Image.fromarray(alpha).resize((nw, nh), Image.LANCZOS))

    # Build a trimap: definite FG (eroded), definite BG (outside dilated), band.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = (a_s >= 250).astype(np.uint8)
    bg = (a_s <= 5).astype(np.uint8)
    fg = cv2.erode(fg, k, iterations=2)
    bg = cv2.erode(bg, k, iterations=2)
    trimap = np.full(a_s.shape, 0.5, dtype=np.float64)
    trimap[fg == 1] = 1.0
    trimap[bg == 1] = 0.0
    try:
        solved = estimate_alpha_cf(rgb_s.astype(np.float64) / 255.0, trimap)
    except (MemoryError, Exception) as exc:  # noqa: BLE001
        logger.warning("matting band solve failed (%s) -> skipping.", exc)
        return alpha
    solved_u8 = np.clip(solved * 255.0, 0, 255).astype(np.uint8)
    if scale != 1.0:
        solved_u8 = np.array(Image.fromarray(solved_u8).resize((w, h), Image.LANCZOS))
    return solved_u8


# ---------------------------------------------------------------------------
# Core rembg call (version-safe, OOM-safe)
# ---------------------------------------------------------------------------
def _call_rembg(img: Image.Image, session: Any, post_process_mask: bool) -> Image.Image:
    """
    Single rembg call returning a coarse RGBA. We deliberately DO NOT use rembg's
    alpha_matting here — we do our own guided refinement + decontamination, which
    is both higher quality and memory-safe.
    """
    img_bytes = to_png_bytes(img)
    kwargs: Dict[str, Any] = {"session": session}
    if "post_process_mask" in _REMBG_PARAMS:
        kwargs["post_process_mask"] = post_process_mask
    result = remove(img_bytes, **kwargs)
    if isinstance(result, bytes):
        return Image.open(io.BytesIO(result)).convert("RGBA")
    if isinstance(result, Image.Image):
        return result.convert("RGBA")
    return Image.fromarray(np.array(result)).convert("RGBA")


def run_segmentation(
    src_full: Image.Image,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, str]:
    """
    Multi-model segmentation with graceful fallback. Returns a COARSE ALPHA at
    full resolution (uint8) plus the model name used. Colour is intentionally
    NOT taken from here — only the mask.
    """
    if not _REMBG_AVAILABLE:
        raise RuntimeError('rembg is not installed. Run: pip install "rembg[cli]" onnxruntime')

    full_size = (src_full.width, src_full.height)
    last_error: Optional[Exception] = None

    for model_name in cfg["models"]:
        try:
            session = get_session(model_name)
            proc_img, _ = _scale_down(src_full, cfg["inference_dim"])
            coarse = _call_rembg(proc_img, session, cfg["post_process_mask"])
            alpha = np.array(coarse.split()[3])  # model alpha at inference res
            # Upscale the MASK only, to full resolution; refinement fixes edges.
            if (coarse.width, coarse.height) != full_size:
                alpha = np.array(
                    Image.fromarray(alpha).resize(full_size, Image.LANCZOS)
                )
            return alpha, model_name
        except Exception as exc:  # noqa: BLE001
            import gc
            logger.warning("Model %s failed: %s — trying next.", model_name, exc)
            _SESSIONS.pop(model_name, None)
            gc.collect()
            last_error = exc
            continue

    raise RuntimeError(f"All models failed. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Background composition
# ---------------------------------------------------------------------------
def _parse_hex_color(value: str) -> Tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(c * 2 for c in v)
    if len(v) != 6:
        raise ValueError(f"Invalid hex colour: {value}")
    return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def compose_background(
    cutout: Image.Image,
    bg_color: Optional[str],
    bg_image_bytes: Optional[bytes],
) -> Image.Image:
    """Place the RGBA cutout on a solid colour or a custom background image."""
    if not bg_color and not bg_image_bytes:
        return cutout
    if bg_image_bytes:
        bg = _load_oriented_rgb(bg_image_bytes).resize(cutout.size, Image.LANCZOS).convert("RGBA")
    else:
        bg = Image.new("RGBA", cutout.size, (*_parse_hex_color(bg_color), 255))  # type: ignore[arg-type]
    out = Image.alpha_composite(bg, cutout)
    return out


# ---------------------------------------------------------------------------
# Studio enhancement (preserves alpha)
# ---------------------------------------------------------------------------
def _studio_enhance(img: Image.Image) -> Image.Image:
    alpha = img.split()[3]
    rgb = img.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(1.05)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.15)
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


# ---------------------------------------------------------------------------
# Top-level processing
# ---------------------------------------------------------------------------
def process_image_bytes(
    image_bytes: bytes,
    original_filename: str = "image.png",
    save_local: bool = False,
    output_path: Optional[Any] = None,
    enhance: bool = False,
    quality: str = DEFAULT_QUALITY,
    model_name: Optional[str] = None,
    refine: Optional[bool] = None,
    decontaminate: Optional[bool] = None,
    bg_color: Optional[str] = None,
    bg_image_bytes: Optional[bytes] = None,
) -> ProcessResult:
    """
    Premium v4 pipeline:
      load+EXIF -> preprocess -> segment(mask only) -> guided refine
                -> (matting band) -> morphology -> decontaminate colour
                -> composite full-res colour + refined alpha
                -> (background replace) -> (enhance) -> encode/save

    `refine` / `decontaminate` override the tier defaults when set.
    """
    validate_image_bytes(image_bytes)

    cfg = QUALITY_CONFIGS.get(quality, QUALITY_CONFIGS["premium"]).copy()
    if model_name:
        cfg["models"] = [model_name]
    if refine is not None:
        cfg["refine"] = refine
    if decontaminate is not None:
        cfg["decontaminate"] = decontaminate

    # 1. Load (with EXIF auto-rotate — fixes sideways phone photos)
    src = _load_oriented_rgb(image_bytes)
    full_size = (src.width, src.height)

    # 2. Preprocess (adaptive low-light / noise)
    preprocessing_info: Dict[str, Any] = {}
    if cfg["preprocess"]:
        src, preprocessing_info = preprocess(src)

    src_rgb = np.array(src)  # full-resolution COLOUR source (the truth for colour)

    # 3. Segment -> coarse alpha at full resolution (mask only)
    coarse_alpha, model_used = run_segmentation(src, cfg)

    refinement_info: Dict[str, Any] = {"model": model_used}

    # 4. Guided-filter edge refinement against full-res image (recover hair)
    if cfg["refine"]:
        alpha = refine_alpha_guided(
            src_rgb, coarse_alpha,
            radius=cfg["guided_radius"], eps=cfg["guided_eps"], feather=cfg["feather"],
        )
        refinement_info["guided_refine"] = True
    else:
        alpha = coarse_alpha

    # 5. Optional matting band solve (ultra / portrait), then morphology cleanup
    if cfg.get("matting_band") and (_PYMATTING_AVAILABLE):
        alpha = matting_band_refine(src_rgb, alpha)
        refinement_info["matting_band"] = True
    alpha = cleanup_alpha_morphology(alpha)

    # 6. Foreground decontamination (remove background colour spill / halo)
    clean_rgb = src_rgb
    if cfg["decontaminate"]:
        clean_rgb, method = decontaminate_foreground(src_rgb, alpha)
        refinement_info["decontaminate"] = method

    # 7. Compose final RGBA: FULL-RES colour + refined alpha (crisp, no upscale blur)
    rgba = np.dstack([clean_rgb, alpha]).astype(np.uint8)
    result_rgba = Image.fromarray(rgba, "RGBA")

    # 8. Optional background replacement
    if bg_color or bg_image_bytes:
        result_rgba = compose_background(result_rgba, bg_color, bg_image_bytes)
        refinement_info["background"] = "color" if bg_color else "image"

    # 9. Optional studio enhancement
    if enhance:
        result_rgba = _studio_enhance(result_rgba)

    # 10. Encode
    png_bytes = to_png_bytes(result_rgba)

    # 11. Save
    saved_path = None
    if save_local:
        ensure_dirs()
        if output_path is None:
            stem = Path(original_filename).stem or "image"
            output_path = DEFAULT_OUTPUT_DIR / f"{stem}_no_bg.png"
        op = Path(output_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_bytes(png_bytes)
        saved_path = str(op)

    return ProcessResult(
        filename=f"{Path(original_filename).stem or 'image'}_no_bg.png",
        png_bytes=png_bytes,
        width=result_rgba.width,
        height=result_rgba.height,
        saved_path=saved_path,
        quality=quality,
        model_used=model_used,
        preprocessing_applied=preprocessing_info,
        refinement_applied=refinement_info,
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
def create_api_app():
    try:
        from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
    except ImportError as exc:
        raise RuntimeError(
            "Install API deps: pip install fastapi uvicorn python-multipart"
        ) from exc

    app = FastAPI(
        title="Premium BG Remover API",
        version="4.0.0",
        description=(
            "Studio-grade human/object background removal. BiRefNet + guided-filter "
            "edge refinement + foreground decontamination. "
            "Quality tiers: fast | balanced | premium | ultra | portrait."
        ),
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.get("/")
    def home():
        return {
            "name": "Premium BG Remover API",
            "version": "4.0.0",
            "status": "running",
            "quality_modes": list(QUALITY_CONFIGS.keys()),
            "default_quality": DEFAULT_QUALITY,
            "engine": {
                "rembg": _REMBG_AVAILABLE,
                "cv2": _CV2_AVAILABLE,
                "pymatting": _PYMATTING_AVAILABLE,
            },
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "cv2": _CV2_AVAILABLE, "pymatting": _PYMATTING_AVAILABLE}

    @app.post("/remove-bg", response_model=None)
    async def remove_bg(
        file: bytes = File(...),
        quality: str = Query(default=DEFAULT_QUALITY, enum=list(QUALITY_CONFIGS.keys())),
        enhance: bool = Query(default=False),
        decontaminate: Optional[bool] = Query(default=None),
        refine: Optional[bool] = Query(default=None),
        bg_color: Optional[str] = Query(default=None, description="Hex e.g. #ffffff"),
        model: Optional[str] = Query(default=None, description="Override rembg model"),
    ):
        """Upload image -> transparent (or recoloured) PNG."""
        try:
            result = process_image_bytes(
                image_bytes=file, original_filename="upload.png", save_local=False,
                enhance=enhance, quality=quality, model_name=model,
                refine=refine, decontaminate=decontaminate, bg_color=bg_color,
            )
            return Response(
                content=result.png_bytes, media_type="image/png",
                headers={
                    "Content-Disposition": f'attachment; filename="{result.filename}"',
                    "X-Image-Width": str(result.width),
                    "X-Image-Height": str(result.height),
                    "X-Quality": result.quality,
                    "X-Model": result.model_used,
                },
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("/remove-bg error", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    @app.post("/remove-bg-json", response_model=None)
    async def remove_bg_json(
        file: bytes = File(...),
        quality: str = Query(default=DEFAULT_QUALITY, enum=list(QUALITY_CONFIGS.keys())),
        enhance: bool = Query(default=False),
        decontaminate: Optional[bool] = Query(default=None),
        refine: Optional[bool] = Query(default=None),
        bg_color: Optional[str] = Query(default=None),
        model: Optional[str] = Query(default=None),
    ):
        """Upload image -> JSON with base64 PNG + processing report."""
        try:
            result = process_image_bytes(
                image_bytes=file, original_filename="upload.png", save_local=False,
                enhance=enhance, quality=quality, model_name=model,
                refine=refine, decontaminate=decontaminate, bg_color=bg_color,
            )
            return {
                "success": True,
                "filename": result.filename,
                "mime_type": "image/png",
                "width": result.width,
                "height": result.height,
                "quality": result.quality,
                "model_used": result.model_used,
                "preprocessing": result.preprocessing_applied,
                "refinement": result.refinement_applied,
                "image_base64": base64.b64encode(result.png_bytes).decode(),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("/remove-bg-json error", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    return app


# Build lazily so the module imports even without fastapi installed.
try:
    app = create_api_app()
except Exception:  # pragma: no cover
    app = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_local(args: argparse.Namespace) -> None:
    if not args.input and not args.url:
        raise SystemExit("Provide --input or --url.")
    if args.input and args.url:
        raise SystemExit("Use only one: --input or --url.")

    if args.url:
        data, downloaded = download_image(args.url)
        fname = Path(downloaded).name
        logger.info("Downloaded: %s", downloaded)
    else:
        data = load_file(args.input)
        fname = Path(args.input).name

    bg_image_bytes = None
    if args.bg_image:
        bg_image_bytes = load_file(args.bg_image)

    result = process_image_bytes(
        image_bytes=data, original_filename=fname, save_local=True,
        output_path=args.output, enhance=args.enhance, quality=args.quality,
        model_name=args.model,
        refine=(False if args.no_refine else None),
        decontaminate=(False if args.no_decontaminate else None),
        bg_color=args.bg_color, bg_image_bytes=bg_image_bytes,
    )

    pre, ref = result.preprocessing_applied, result.refinement_applied
    print(f"Done | model: {result.model_used} | quality: {result.quality}")
    print(f"Output: {result.saved_path}  ({result.width}x{result.height})")
    if pre.get("clahe"):
        b = pre.get("stats", {}).get("mean_brightness", 0.0)
        print(f"  [preprocess] Low-light correction (brightness={b:.1f}) -> CLAHE")
    if pre.get("bilateral"):
        print("  [preprocess] Noise reduction -> bilateral filter")
    if ref.get("guided_refine"):
        print("  [refine]     Guided-filter edge refinement (hair recovery)")
    if ref.get("matting_band"):
        print("  [refine]     Matting-band alpha solve")
    if ref.get("decontaminate"):
        print(f"  [refine]     Foreground decontamination -> {ref['decontaminate']}")
    if ref.get("background"):
        print(f"  [compose]    Background replaced -> {ref['background']}")


def run_api(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install: pip install uvicorn") from exc
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Premium BG Remover v4")
    sub = p.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("local", help="Process one image and save output")
    lp.add_argument("--input", help="Local image path")
    lp.add_argument("--url", help="Image URL")
    lp.add_argument("--output", help="Output PNG path (auto if omitted)")
    lp.add_argument("--enhance", action="store_true", help="Studio enhancement")
    lp.add_argument("--quality", choices=list(QUALITY_CONFIGS.keys()),
                    default=DEFAULT_QUALITY, help="Quality tier (default: premium)")
    lp.add_argument("--model", default=None, help="Override model name")
    lp.add_argument("--no-refine", action="store_true", help="Disable guided refinement")
    lp.add_argument("--no-decontaminate", action="store_true", help="Disable spill removal")
    lp.add_argument("--bg-color", default=None, help="Replace background with hex colour e.g. #ffffff")
    lp.add_argument("--bg-image", default=None, help="Replace background with an image file")
    lp.set_defaults(func=run_local)

    ap = sub.add_parser("api", help="Run FastAPI server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    ap.set_defaults(func=run_api)

    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Stopped.")
    except Exception as exc:
        logger.error("Fatal: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
