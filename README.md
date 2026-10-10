# Histology Workbench

A local, browser-based tool for **reviewing Iba1/DAPI immunofluorescence images** and counting candidate microglial somata. It uses classical image processing (NumPy, SciPy, scikit-image). It is **not** a trained machine-learning model.

Automatic detections are *staining-based candidates*, not validated microglial identities. Every count is meant to be confirmed by manual review.

## Features

- **Soma-candidate detection** on the Iba1 (green) channel:
  - Gaussian background correction
  - robust MAD-based thresholding
  - morphological opening to suppress thin processes
  - distance-transform / h-maxima watershed to split touching somata, with saddle-ratio merging of over-split cells
- **DAPI association**: each candidate is linked to nearby nuclei (centre-to-footprint distance).
- **Automatic review flags**: `small_soma`, `large_or_merged_soma`, `elongated_region`, `no_nearby_DAPI`, `multiple_nearby_nuclei`, `weak_green_support`, `shared_DAPI_review_split`.
- **Manual review in the browser**: accept, reject, add, move, undo (100 steps), border-cell rule, optional ROI, "finish review of this image".
- **Folder / batch workflow**: analyse many images with shared settings; export a batch summary (CSV/JSON/ZIP).
- **Exports**: per-image cell CSV, annotated PNG, review session JSON (SHA-256 checked on restore).
- **Validation**: compare candidates with a *complete* reference annotation (one-to-one Hungarian matching) to report precision, recall, missed cells and duplicate-like extras.
- **Reproducibility**: settings, intensity scales, thresholds and method version (`1.0-soma-review`) are saved with every result.

## Install and run

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8501> (set a different port with the `PORT` environment variable). Keep the terminal open; Ctrl+C stops the server. The server listens on `127.0.0.1` only, and images are held in memory (max 3 active) and not written to disk. Restarting the server erases them.

## Command-line batch counting

```bash
python batch_count.py            # opens a folder picker
python batch_count.py path/to/folder
```

Writes a timestamped `batch_results_*` folder containing, per image, `*.cells.csv` and `*.settings.json`, plus `summary.csv`. The "final reviewed count" column stays **blank until manual review**. Default trial setting: `core_radius = 4`; edit `OPTIONS` in the script.

## Input requirements

- One 2D field per file: multichannel TIFF (`YXS`, `YXC`, `CYX`, `SYX`) or RGB/RGBA PNG/JPG/WebP.
- Choose which channel is Iba1 and which is DAPI (RGB default: green = 1, blue = 2).
- Limits: 100 MB per file, 16 megapixels. Z/T stacks, multi-series TIFFs and grayscale-only files are rejected.
- Pixel size is **not inferred**; enter a verified value if you need physical units.
- Compressed exports (JPEG/WebP) are for demonstration only. Use original acquisition files for analysis.

## Main settings

| Setting | Default | Meaning |
|---|---|---|
| `green_sensitivity` | 65 | Iba1 threshold sensitivity |
| `background_sigma` / `background_strength` | 15 / 1 | Background subtraction |
| `min_soma_area` / `max_soma_area` | 18 / 1800 px | Size flags |
| `core_radius` | 3 | Opening radius (removes processes) |
| `split_prominence` / `merge_saddle` | 1.5 / 0.75 | Watershed split / merge control |
| `dapi_sensitivity`, `min_nucleus_area` | 55, 5 | Nucleus detection |
| `association_distance` | 6 px | Max soma–nucleus distance |

Defaults are starting points, not calibrated assay thresholds.

## Limitations

- Broad Gaussian background subtraction can remove large, dim somata.
- Dense clusters may be merged or over-split; dim staining may be missed.
- Iba1-positive objects are not proven to be microglia without tissue/assay context.
- Precision/recall are meaningful only on fully annotated fields or ROIs. No accuracy claims are made for this prototype.
- For rigorous use, freeze parameters first, then evaluate on held-out samples split by animal, not by adjacent fields.
- No authentication or persistent storage. Intended for single-user, local use.

## Files

| File | Purpose |
|---|---|
| `app.py` | Local HTTP server, image decoding, export |
| `analysis.py` | Detection and validation logic |
| `batch_count.py` | Folder batch script |
| `index.html` | Browser interface |
| `requirements.txt` | Dependencies |

## Author

Reema Chowdhury, Ph.D. (Neuroscience)
