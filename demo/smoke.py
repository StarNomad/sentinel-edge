"""SENTINEL-EDGE smoke test: measures detector latency/FPS on-device.

Run from the repo root (so the `detector` package imports):
    cd C:\\tools\\sentineledge-src
    python -m demo.smoke --model C:\\tools\\sentineledge\\models\\best.onnx ^
        --image C:\\tools\\sentineledge\\models\\test.jpg

--image accepts one image or a directory of images. Timing covers the full
infer() path (preprocess + session.run + decode + NMS). A repeated single
image measures steady-state throughput; a directory measures per-image
variation. The report says which one it was — no silent substitution.

Numbers go into docs/measured-results.md only from this script's output.
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script (python demo/smoke.py) as well as -m.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from detector.detector import Detector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ONNX detector path")
    ap.add_argument("--image", required=True,
                    help="test image or directory of images")
    ap.add_argument("--classes", nargs="+", default=["Hardhat", "NO-Hardhat"])
    ap.add_argument("--iters", type=int, default=50,
                    help="timed iterations per image")
    args = ap.parse_args()

    det = Detector(args.model, class_names=tuple(args.classes))
    print(f"providers: {det.active_providers} | {det.provider_note}")

    p = Path(args.image)
    images = sorted(i for i in p.iterdir() if i.suffix.lower() in IMG_EXTS) \
        if p.is_dir() else [p]
    if not images:
        raise SystemExit(f"no images found at {args.image}")
    mode = ("directory sweep" if len(images) > 1
            else "single image repeated (steady-state throughput)")

    all_ts: list[float] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: cannot read {img_path}, skipping")
            continue
        for _ in range(5):  # warmup
            det.infer(img)
        dets = det.infer(img)
        print(f"image={img_path.name} shape={img.shape[1]}x{img.shape[0]} "
              f"detections={len(dets)}")
        for d in dets:
            print(f"  {d.class_name} conf={d.confidence:.2f} "
                  f"box={[round(float(x), 1) for x in d.xyxy]}")
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            det.infer(img)
            ts.append((time.perf_counter() - t0) * 1000.0)
        all_ts.extend(ts)
        ts.sort()
        mean = sum(ts) / len(ts)
        print(f"  iters={len(ts)} mean_ms={mean:.1f} "
              f"p50_ms={ts[len(ts) // 2]:.1f} "
              f"p95_ms={ts[int(len(ts) * 0.95)]:.1f} "
              f"fps={1000.0 / mean:.1f}")

    all_ts.sort()
    n = len(all_ts)
    mean = sum(all_ts) / n
    print(f"MODE={mode}")
    print(f"OVERALL iters={n} mean_ms={mean:.1f} p50_ms={all_ts[n // 2]:.1f} "
          f"p95_ms={all_ts[int(n * 0.95)]:.1f} fps={1000.0 / mean:.1f}")


if __name__ == "__main__":
    main()
