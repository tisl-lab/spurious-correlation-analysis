"""
Image Gallery — Streamlit app
=============================

What this does
---------------
1. You give it a base folder. Images can live *anywhere* under it, nested
   arbitrarily deep, spread across many sub-folders — the app doesn't
   assume a fixed layout.
2. It recursively scans that folder once (and caches the scan), then shows
   what it found, grouped by the sub-folder each image came from.
3. Optionally you narrow things down: pick one sub-folder from the
   dropdown, and/or type a filter string.

Point it at a concepts/patches folder and it just shows everything — no
word list, nothing to prepare. (An earlier version required uploading a
word list and could only show images matching one word from it; the filter
box below covers that case without the upload.)

How to run
----------
    pip install streamlit
    streamlit run image_gallery_app.py

Then open the URL it prints (usually http://localhost:8501).

Notes
-----
- The base folder is given as a path on the machine running the app, since
  it's expected to be large and stay put.
- The filter is a plain substring check, case-insensitive by default —
  "apple" matches "apple", "red_apple_v2", ".../apple/close_up.png".
  Turn on "Case sensitive" in the sidebar if you need exact-case matching.
- The full recursive file scan is cached per base-folder path. If images
  are added or removed while the app is running (a MACO run writing new
  renders, say), use "Rescan folder" in the sidebar rather than restarting.
- "Max images to display" is a safety cap: a big folder can hold thousands
  of images and rendering them all will hang the browser, not the app.
"""

import os
from pathlib import Path

import streamlit as st

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

st.set_page_config(page_title="Image gallery", layout="wide")


# ----------------------------------------------------------------------
# Recursive image scan (cached per base folder)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner="Scanning folder for images…")
def scan_images(base_folder: str) -> list[str]:
    """Recursively collect every image file path under base_folder."""
    base = Path(base_folder)
    found: list[str] = []
    for root, _dirs, files in os.walk(base):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                found.append(str(Path(root) / fname))
    return found


def matches_filter(path: str, needle: str, case_sensitive: bool, target: str) -> bool:
    """Substring test against the folder path, the file name, or both."""
    p = Path(path)
    haystacks = []
    if target in ("Folder names", "Both"):
        haystacks.append(str(p.parent))
    if target in ("File names", "Both"):
        haystacks.append(p.name)

    needle = needle if case_sensitive else needle.lower()
    for h in haystacks:
        h = h if case_sensitive else h.lower()
        if needle in h:
            return True
    return False


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
st.sidebar.header("Configuration")

base_folder = st.sidebar.text_input(
    "Base folder (searched recursively)",
    value="",
    placeholder="/absolute/path/to/your/results",
    help="Every sub-folder under this path is searched, no matter how deep.",
)

text_filter = st.sidebar.text_input(
    "Filter (optional)",
    value="",
    placeholder="e.g. squinting, patch, concept_1435",
    help="Substring match against paths and/or file names. Leave empty to show everything.",
)
match_target = st.sidebar.radio(
    "Match the filter against", options=["Both", "Folder names", "File names"], index=0
)
case_sensitive = st.sidebar.checkbox("Case sensitive match", value=False)
n_cols = st.sidebar.slider("Gallery columns", min_value=2, max_value=8, value=4)
max_images = st.sidebar.slider(
    "Max images to display", min_value=10, max_value=500, value=80, step=10,
    help="Safety cap so a large folder doesn't try to render thousands of images.",
)

if st.sidebar.button("🔄 Rescan folder"):
    scan_images.clear()
    st.sidebar.success("Cache cleared — will rescan on next run.")


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("🖼️ Image gallery")

if not base_folder:
    st.info("⬅️ Enter the base folder to search in the sidebar.")
    st.stop()

if not Path(base_folder).is_dir():
    st.error(f"'{base_folder}' isn't a folder I can find on this machine. Check the path.")
    st.stop()

all_images = scan_images(base_folder)
if not all_images:
    st.warning(f"No image files found anywhere under `{base_folder}`.")
    st.stop()

st.caption(f"Indexed **{len(all_images):,}** image files under `{base_folder}`.")

# Sub-folder picker, built from what the scan actually found — this is what
# replaces the old word list: the folders ARE the concepts.
folders = sorted({str(Path(p).parent) for p in all_images})
folder_labels = ["All folders"] + [os.path.relpath(f, base_folder) for f in folders]
chosen = st.selectbox(f"Sub-folder ({len(folders)} with images)", options=folder_labels)

matches = all_images
if chosen != "All folders":
    wanted = os.path.normpath(os.path.join(base_folder, chosen))
    matches = [p for p in matches if os.path.normpath(str(Path(p).parent)) == wanted]
if text_filter.strip():
    matches = [p for p in matches
               if matches_filter(p, text_filter.strip(), case_sensitive, match_target)]

if not matches:
    st.warning("Nothing matched that sub-folder / filter combination.")
    st.stop()

label = chosen if chosen != "All folders" else "all folders"
suffix = f" matching '{text_filter.strip()}'" if text_filter.strip() else ""
st.subheader(f"{label}{suffix} — {len(matches)} image(s)")

if len(matches) > max_images:
    st.caption(
        f"Showing the first {max_images} of {len(matches)} matches "
        f"(raise the cap in the sidebar to see more)."
    )
    matches = matches[:max_images]

# group by the folder each image lives in, so results from different
# paths under the base folder are clearly separated
by_folder: dict[str, list[str]] = {}
for path in matches:
    folder = str(Path(path).parent)
    by_folder.setdefault(folder, []).append(path)

for folder, paths in sorted(by_folder.items()):
    rel = os.path.relpath(folder, base_folder)
    with st.expander(f"📁 {rel}  ·  {len(paths)} image(s)", expanded=True):
        cols = st.columns(n_cols)
        for i, path in enumerate(sorted(paths)):
            with cols[i % n_cols]:
                st.image(path, caption=Path(path).name, use_container_width=True)
