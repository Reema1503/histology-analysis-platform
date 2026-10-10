# Histology Workbench

A runnable website prototype for Iba1/DAPI fluorescence candidate-cell screening and manual H&E lung-region measurement. This is classical image processing, not a trained AI model. It does not implement automatic lung-metastasis identification.

## Run on Windows

Extract this folder, open it in VS Code, and open a PowerShell terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:8501 in your browser. Keep the terminal running. Ctrl+C stops it.

## Microglia workflow

1. Choose Iba1 / DAPI microglia and upload a single 8-bit RGB TIFF, PNG or JPG. Green must represent Iba1, blue DAPI. Raw channels, 16-bit images, stacks and whole slides need a separate importer.
2. Set an analysis rectangle excluding labels and the scale bar. Enter pixel size if independently known; the application does not infer it from an embedded scale bar.
3. Adjust brightness, contrast, gamma and channel visibility for viewing. Analysis always uses original pixels.
4. Detect candidates. The baseline thresholds the smoothed DAPI channel, finds connected nuclear components, filters by area, and checks Iba1-positive pixels in a surrounding ring.
5. Review the overlay. Add missed candidates or remove false positives. Save CSV, settings and the display overlay.

Defaults are starting parameters, not calibrated assay thresholds. Touching nuclei may merge. Nearby Iba1 processes may be incorrectly associated with other nuclei. Weak staining may be missed. Iba1-positive cells are not automatically proven microglia; interpretation depends on tissue and assay context. Density uses rectangular ROI area, not segmented tissue area. Adjusting detection parameters does not update existing results until detection is run again.

## Lung H&E workflow

Upload a genuine H&E image. Outline lung tissue using the polygon tool and finish each polygon. Outline tumour regions similarly. The app computes the union of tumour pixels inside the union of outlined tissue pixels within the ROI, avoiding overlap double-counting. Tumour region count is the number of annotations, not an automatically validated lesion count. Save settings to retain polygon coordinates.

Next development step: obtain representative H&E images and expert region annotations, develop a segmentation model, and evaluate it on held-out biological samples. No automatic lesion count is reported by this version.

## Validate before using measurements

Compare marked detections with blinded manual annotations; assess missed cells and false detections, not just total-count agreement. Include different staining intensities, background levels and cell densities. Freeze parameters before evaluating held-out images, splitting by animal/patient rather than adjacent image fields. No accuracy claims are established by this prototype.

## Host as a separate Render website

Create a new GitHub repository and upload these files at the root. In Render create a new Web Service from that repository and use the included Dockerfile. The server binds to 0.0.0.0 inside Docker and reads Render's PORT environment variable. Do not replace the existing OncoVista repository.

This prototype has no authentication or project database. Files are processed in memory and not written to disk. Uploaded data and results disappear on refresh unless exported. Keep access restricted; do not expose confidential images publicly. Add authentication, access controls and persistent project storage before shared use.

## Limits

25 MB per image, 16 megapixels, one frame, 8-bit image. Browser uploads are sent to the hosting server for decoding and counting. TIFF label/metadata is not used for channel assignment or calibration. Repeated polygon area measurement may be slow for large fields. Exported overlays reflect current zoom and display adjustments; numeric analysis uses original resolution.

## Version 0.2: stricter DAPI screening

Nuclear candidates must now contain blue-dominant pixels with signal above a local median background. New adjustable controls: minimum DAPI contrast (default 15 intensity units) and blue/other-colour ratio (default 1.2). Markers are anchored within supported nuclear pixels. CSV exports include nuclear blue intensity and local contrast. These settings may reject true cells with dim DAPI or strong overlapping green fluorescence. Review DAPI-only overlays and tune on annotated examples; defaults are not validated biological thresholds.

To update the hosted website, replace analysis.py and index.html in the existing GitHub repository, commit, and deploy the latest commit on Render. No dependency or Dockerfile changes are needed.
