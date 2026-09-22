"""
Word-Filtered Image Gallery — Streamlit app
=============================================

What this does
---------------
1. You give it a word list (a small file: one word per line, or a CSV/Excel
   file with a column of words).
2. You give it a base folder. Images for these words can live *anywhere*
   under that folder, nested arbitrarily deep, spread across many
   sub-folders — the app doesn't assume a fixed layout.
3. You pick one word from a searchable dropdown.
4. The app recursively scans the base folder once (and caches the scan),
   then shows every image whose folder path or filename contains that
   word as a substring, grouped by the folder it came from.

How to run
----------
    pip install streamlit
    streamlit run image_gallery_app.py

Then open the URL it prints (usually http://localhost:8501).

Notes
-----
- The word list file is uploaded through the browser each session (small
  file, no path issues). The image base folder is given as a path on the
  machine running the app, since it's expected to be large and stay put.
- Matching is a plain substring check, case-insensitive by default —
  "apple" matches "apple", "red_apple_v2", ".../apple/close_up.png", etc.
  Turn on "Case sensitive" in the sidebar if you need exact-case matching.
- The full recursive file scan is cached per base-folder path. If you add
  or remove images while the app is running, use the "Rescan folder"
  button in the sidebar rather than restarting the app.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

st.set_page_config(page_title="Image gallery by word", layout="wide")


# ----------------------------------------------------------------------
# Word list loading
# ----------------------------------------------------------------------
def load_words(uploaded_file) -> list[str]:
    """Accepts a .txt (one word per line) or a .csv/.xlsx with a
    'word'/'words' column (falls back to the first column)."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="ignore")
        words = [line.strip() for line in text.splitlines() if line.strip()]
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        words = _column_of_words(df)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
        words = _column_of_words(df)
    else:
        st.error(f"Unsupported word-list file type: {uploaded_file.name}")
        return []

    # de-duplicate, keep first-seen order
    seen = set()
    deduped = []
    for w in words:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped


def _column_of_words(df: pd.DataFrame) -> list[str]:
    for col in df.columns:
        if str(col).strip().lower() in ("word", "words"):
            return [str(v).strip() for v in df[col].dropna().tolist()]
    # no obviously-named column — just use the first one
    first_col = df.columns[0]
    return [str(v).strip() for v in df[first_col].dropna().tolist()]


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


def matches_word(path: str, word: str, case_sensitive: bool, target: str) -> bool:
    p = Path(path)
    haystacks = []
    if target in ("Folder names", "Both"):
        haystacks.append(str(p.parent))
    if target in ("File names", "Both"):
        haystacks.append(p.name)

    needle = word if case_sensitive else word.lower()
    for h in haystacks:
        h = h if case_sensitive else h.lower()
        if needle in h:
            return True
    return False


# ----------------------------------------------------------------------
# Sidebar — configuration
# ----------------------------------------------------------------------
st.sidebar.header("Configuration")

word_file = st.sidebar.file_uploader(
    "Word list (.txt, .csv, or .xlsx)", type=["txt", "csv", "xlsx", "xls"]
)

base_folder = st.sidebar.text_input(
    "Base folder (searched recursively)",
    value="",
    placeholder="/absolute/path/to/your/results",
    help="Every sub-folder under this path is searched, no matter how deep.",
)

match_target = st.sidebar.radio(
    "Match the word against", options=["Both", "Folder names", "File names"], index=0
)
case_sensitive = st.sidebar.checkbox("Case sensitive match", value=False)
n_cols = st.sidebar.slider("Gallery columns", min_value=2, max_value=8, value=4)
max_images = st.sidebar.slider(
    "Max images to display", min_value=10, max_value=500, value=80, step=10,
    help="Safety cap so a very common word doesn't try to render thousands of images.",
)

if st.sidebar.button("🔄 Rescan folder"):
    scan_images.clear()
    st.sidebar.success("Cache cleared — will rescan on next run.")


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
st.title("🖼️ Image gallery by word")

if not word_file:
    st.info("⬅️ Upload a word-list file in the sidebar to get started.")
    st.stop()

words = load_words(word_file)
if not words:
    st.error("Couldn't find any words in that file.")
    st.stop()

if not base_folder:
    st.info("⬅️ Enter the base folder to search in the sidebar.")
    st.stop()

if not Path(base_folder).is_dir():
    st.error(f"'{base_folder}' isn't a folder I can find on this machine. Check the path.")
    st.stop()

all_images = scan_images(base_folder)
st.caption(f"Indexed **{len(all_images):,}** image files under `{base_folder}`.")

selected_word = st.selectbox(
    f"Pick a word ({len(words)} available)", options=sorted(words)
)

matches = [p for p in all_images if matches_word(p, selected_word, case_sensitive, match_target)]

if not matches:
    st.warning(f"No images matched **'{selected_word}'**.")
    st.stop()

st.subheader(f"'{selected_word}' — {len(matches)} image(s) found")

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
