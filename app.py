import streamlit as st

from kmeans_utils import (ideal_bytes, load_image, mse_psnr, palette_hex,
                          palette_strip, quantize, render,
                          to_png_bytes_indexed, to_png_bytes_rgb)

st.set_page_config(page_title="K-Means Colour Compressor", layout="wide")  # must be the first Streamlit call


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes, max_side: int):
    return load_image(file_bytes, max_side)                 # -> (H, W, 3) uint8


@st.cache_data(show_spinner=False)
def original_png(image):
    return to_png_bytes_rgb(image)                          # bytes of the full-colour baseline PNG


@st.cache_data(show_spinner="Clustering pixels...")
def compress(image, k):                                     # cache key = (image contents, k)
    labels, palette = quantize(image, k)                    # (H, W) indices, (k, 3) palette
    recon = render(labels, palette)                         # (H, W, 3) compressed image
    mse, psnr = mse_psnr(image, recon)                      # two scalars
    return {"recon": recon, "palette": palette, "psnr": psnr,
            "png": to_png_bytes_indexed(labels, palette)}   # bytes ready for download


st.title("K-Means Colour Compressor")
st.write("Upload a photo, pick palette sizes, and compare how few colours an image really needs.")

st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
ks = st.sidebar.multiselect("Palette sizes (colours)",
                            options=[2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32, 64],
                            default=[10, 7, 4])             # list of ints chosen by the user
max_side = st.sidebar.slider("Max image side (pixels)", 200, 1600, 800, step=100)

if uploaded is None:
    st.info("Upload an image in the sidebar to begin.")
    st.stop()                                               # end this rerun here; nothing below executes
if not ks:
    st.warning("Pick at least one palette size.")
    st.stop()

image = cached_load(uploaded.getvalue(), max_side)          # .getvalue() -> raw bytes of the upload
h, w, _ = image.shape
n_pixels = h * w
base_png = original_png(image)

st.subheader("Original")
st.image(image, width=520,
         caption=f"{w} x {h} px = {n_pixels:,} pixels | full-colour PNG = {len(base_png) / 1024:,.0f} KB")

st.subheader("Compressed versions")
ordered = sorted(ks, reverse=True)                          # most colours first, so quality declines left to right
PER_ROW = 3
for start in range(0, len(ordered), PER_ROW):
    row_ks = ordered[start:start + PER_ROW]                 # up to 3 K-values for this row
    cols = st.columns(PER_ROW)                              # 3 equal-width columns
    for col, k in zip(cols, row_ks):
        res = compress(image, k)
        n_colours = len(res["palette"])                     # may be < k for images with few distinct colours
        size_kb = len(res["png"]) / 1024
        with col:
            st.markdown(f"**{n_colours} colours**")
            st.image(res["recon"], use_container_width=True)
            st.image(palette_strip(res["palette"]))         # (48, n_colours*48, 3) swatch strip
            st.caption(" ".join(palette_hex(res["palette"])))
            st.metric("PSNR (higher = closer to original)", f"{res['psnr']:.1f} dB")
            st.metric("PNG size", f"{size_kb:,.0f} KB",
                      delta=f"{100 * (len(res['png']) / len(base_png) - 1):.0f}% vs original",
                      delta_color="inverse")
            st.caption(f"Ideal fixed-width size: {ideal_bytes(n_pixels, n_colours) / 1024:,.0f} KB")
            st.download_button(f"Download {n_colours}-colour PNG", data=res["png"],
                               file_name=f"compressed_{n_colours}_colours.png",
                               mime="image/png", key=f"dl_{k}")