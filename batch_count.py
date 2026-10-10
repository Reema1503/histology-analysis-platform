"""Run beside app.py and analysis.py: python batch_count.py"""
import base64
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog
from app import decode_image
from analysis import count_candidates

# Trial settings: radius 4 suppresses more branches but may miss small somata.
OPTIONS = {"core_radius": 4}
GREEN_CHANNEL, BLUE_CHANNEL = 1, 2  # RGB channel indices.


def main():
    if len(sys.argv) > 1:
        selected = sys.argv[1]
    else:
        root = Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Select your microscopy image folder")
        root.destroy()
    if not selected:
        print("No folder selected.")
        return
    folder = Path(selected)
    files = sorted(p for p in folder.iterdir() if p.is_file() and
                   p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        print("No supported images found in this folder.")
        return
    output = folder / datetime.now().strftime("batch_results_%Y%m%d_%H%M%S_%f")
    output.mkdir()
    summary = []
    for path in files:
        row = {"image": path.name, "automatic_candidates": "", "uncertain": "",
               "border": "", "final_reviewed_count": "", "error": ""}
        try:
            encoded = base64.b64encode(path.read_bytes()).decode()
            image, metadata = decode_image(encoded, GREEN_CHANNEL, BLUE_CHANNEL)
            result = count_candidates(image, OPTIONS)
            cells = result["cells"]
            row.update(automatic_candidates=len(cells),
                       uncertain=sum(c["uncertain"] for c in cells),
                       border=sum(c["border"] for c in cells))
            with (output / (path.name + ".cells.csv")).open("w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "x_px", "y_px", "review_status", "uncertain", "border", "flags"])
                for c in cells:
                    writer.writerow([c["id"], c["x"], c["y"], c["review_status"],
                                     c["uncertain"], c["border"], ";".join(c["reasons"])])
            (output / (path.name + ".settings.json")).write_text(json.dumps(
                {"image": metadata, "settings": result["settings"],
                 "method": result["method_version"], "noise_regions": result["noise_regions"]},
                indent=2), encoding="utf-8")
            print(f"{path.name}: {len(cells)} automatic candidates (not reviewed)")
        except Exception as error:
            row["error"] = str(error)
            print(f"{path.name}: ERROR — {error}")
        summary.append(row)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Results saved to: {output}\nReviewed counts are blank until manual review.")


if __name__ == "__main__":
    main()
