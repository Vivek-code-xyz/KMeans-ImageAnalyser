"""K-Means colour quantisation. Pure NumPy / Pillow / scikit-learn, no Streamlit here."""
import io
import math

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans


def load_image(file_bytes: bytes, max_side: int = 800) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes))     # bytes -> PIL image (any format, any mode)
    img = ImageOps.exif_transpose(img)           # respect phone-camera rotation flags
    img = img.convert("RGB")                     # force 3 channels (drops alpha, expands grayscale)
    img.thumbnail((max_side, max_side))          # shrink in place, keep aspect ratio, never enlarge
    return np.array(img)                         # shape (H, W, 3), dtype uint8, values 0..255


def quantize(image: np.ndarray, k: int, sample_size: int = 20_000, seed: int = 42):
    h, w, _ = image.shape                        # e.g. (533, 800, 3)
    pixels = image.reshape(-1, 3)                # (H*W, 3) = (426400, 3): one row per pixel, cols = (R, G, B)

    rng = np.random.default_rng(seed)
    n = pixels.shape[0]                          # N = H*W
    idx = rng.choice(n, size=min(sample_size, n), replace=False)   # (m,) distinct row numbers, m = 20000
    sample = pixels[idx].astype(np.float32)      # (m, 3): the subset we actually train on

    k_eff = min(k, len(np.unique(sample, axis=0)))   # np.unique(...) -> (u, 3) distinct colours; cannot have more clusters than colours
    km = KMeans(n_clusters=k_eff, n_init=3, random_state=seed).fit(sample)   # internal centroids: (k_eff, 3)

    labels = km.predict(pixels.astype(np.float32))   # (H*W,) palette index 0..k_eff-1 for EVERY pixel
    palette = np.clip(np.rint(km.cluster_centers_), 0, 255).astype(np.uint8)  # (k_eff, 3): centroids rounded to valid colours
    return labels.reshape(h, w), palette         # index map (H, W), palette (k_eff, 3)


def render(labels: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return palette[labels]                       # fancy indexing: (H, W) ints into (k,3) rows -> (H, W, 3) image


def mse_psnr(original: np.ndarray, compressed: np.ndarray):
    diff = original.astype(np.float64) - compressed.astype(np.float64)   # (H, W, 3); float avoids uint8 wrap-around
    mse = float(np.mean(diff ** 2))              # average over all H*W*3 numbers -> scalar
    psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    return mse, psnr


def to_png_bytes_rgb(image: np.ndarray) -> bytes:
    buf = io.BytesIO()                           # an in-memory "file"
    Image.fromarray(image).save(buf, format="PNG", optimize=True)   # (H, W, 3) uint8 -> full-colour PNG
    return buf.getvalue()


def to_png_bytes_indexed(labels: np.ndarray, palette: np.ndarray) -> bytes:
    img = Image.fromarray(labels.astype(np.uint8))   # (H, W) uint8 -> greyscale image whose "brightness" is the palette index
    img.putpalette(palette.flatten().tolist())       # (k,3) -> flat list of 3k ints; image becomes mode "P" (indexed)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def ideal_bytes(n_pixels: int, k: int) -> float:
    bits_per_pixel = max(1, math.ceil(math.log2(k)))
    return (n_pixels * bits_per_pixel + 24 * k) / 8


def palette_strip(palette: np.ndarray, swatch: int = 48) -> np.ndarray:
    row = np.repeat(palette[np.newaxis, :, :], swatch, axis=0)   # (k,3)->(1,k,3)->(swatch, k, 3): stack copies downwards
    return np.repeat(row, swatch, axis=1)                        # (swatch, k, 3)->(swatch, k*swatch, 3): widen each colour


def palette_hex(palette: np.ndarray) -> list:
    return ["#{:02x}{:02x}{:02x}".format(*map(int, c)) for c in palette]   # (k,3) -> list of k strings like "#a3c1ff"