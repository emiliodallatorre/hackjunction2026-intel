"""`brick-ad` command line: camera -> dataset -> PatchCore -> stats -> inference."""

import argparse
from pathlib import Path

from .config import GOOD, LABELS, load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="brick-ad", description=__doc__)
    parser.add_argument("--config", type=Path, help="config file (default: anomaly_detection/config.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cameras", help="list camera devices that deliver frames")

    p = sub.add_parser("stream", help="live camera preview with ROI (optionally with live inference)")
    p.add_argument("--infer", action="store_true", help="run the trained model on every frame")
    p.add_argument("--heatmap", action="store_true",
                   help="overlay the anomaly heatmap on the brick in the main view (implies --infer)")
    p.add_argument("--model", type=Path, help="model folder (default: paths.model)")

    p = sub.add_parser("lock", help="lock focus, exposure and white balance to measured values")
    p.add_argument("--off", action="store_true", help="go back to automatic focus/exposure/white balance")

    p = sub.add_parser("roi", help="draw the crop region and save it to the config")
    p.add_argument("--image", type=Path, help="draw on a saved frame instead of the live camera")

    p = sub.add_parser("collect", help="capture labelled raw frames")
    p.add_argument("--label", choices=LABELS, default=GOOD, help="starting label (switch live with g/a)")

    p = sub.add_parser("prepare", help="crop raw frames and build the train/test split")
    p.add_argument("--shuffle", action="store_true",
                   help="hold out a random sample instead of the most recent captures")
    p.add_argument("--seed", type=int, default=0, help="random seed for --shuffle (default 0)")

    p = sub.add_parser("train", help="train PatchCore, export to OpenVINO and print test stats")
    p.add_argument("--no-eval", action="store_true", help="skip the stats on the test split")

    sub.add_parser("fit-empty", help="recalibrate empty-ROI detection only (after prepare, no retraining)")

    p = sub.add_parser("evaluate", help="stats on a labelled folder with good/, anomaly/ and empty/ subfolders")
    p.add_argument("--data", type=Path, help="folder to evaluate (default: the prepared test split)")
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=False,
                   help="apply the ROI first (use for raw camera frames)")
    p.add_argument("--save", type=Path, help="write heatmap overlays to this folder")

    p = sub.add_parser("predict", help="score images or folders of images")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True,
                   help="apply the ROI first (default; use --no-crop for already-cropped images)")

    for name in ("evaluate", "predict"):
        sp = sub.choices[name]
        sp.add_argument("--model", type=Path, help="model folder (default: paths.model)")
        sp.add_argument("--device", help="OpenVINO device (default: inference.device)")
        sp.add_argument("--threshold", type=float, help="decision threshold (default: inference.threshold in the config, else 0.5)")

    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = getattr(args, "model", None) or cfg.paths.model
    device = getattr(args, "device", None) or cfg.inference.device
    threshold = getattr(args, "threshold", None)
    threshold = cfg.inference.threshold if threshold is None else threshold

    if args.command == "cameras":
        from .cameras import probe_cameras

        cameras = probe_cameras()
        if not cameras:
            print("No camera delivers frames. Is it plugged in, or in use by another program?")
        for source, (width, height), stable in cameras:
            current = "  <- config" if str(cfg.camera.source) in (source, stable) else ""
            print(f"  {source:12} {width}x{height:<5} {stable or ''}{current}")
        print("Put the stable /dev/v4l/by-id/... path in camera.source in config.yaml.")

    elif args.command == "stream":
        from .capture import stream

        stream(cfg, model_dir if (args.infer or args.heatmap or args.model) else None, heatmap=args.heatmap)

    elif args.command == "lock":
        from .lock import lock, unlock

        if args.off:
            unlock(cfg)
        else:
            lock(cfg)

    elif args.command == "roi":
        from .capture import select_roi

        select_roi(cfg, args.image)

    elif args.command == "collect":
        from .capture import collect

        collect(cfg, args.label)

    elif args.command == "prepare":
        from .dataset import prepare

        prepare(cfg, shuffle=args.shuffle, seed=args.seed)

    elif args.command == "train":
        from .train import train

        model_dir = train(cfg)
        if not args.no_eval:
            from .evaluate import evaluate

            evaluate(model_dir, cfg.paths.dataset / "test", device=device, threshold=threshold,
                     uncertain_margin=cfg.inference.uncertain_margin)
            print("Note: the threshold was calibrated on this same test split; confirm on fresh captures with\n"
                  "      `brick-ad evaluate --data <folder with good/ anomaly/ empty/> --crop`.")

    elif args.command == "fit-empty":
        from .train import fit_empty

        fit_empty(cfg, model_dir)

    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(model_dir, args.data or cfg.paths.dataset / "test", device=device, threshold=threshold,
                 uncertain_margin=cfg.inference.uncertain_margin, crop=args.crop, save_dir=args.save)

    elif args.command == "predict":
        from .imaging import expand_paths
        from hackjunction2026_intel.anomaly_detection.inspector import BrickInspector

        inspector = BrickInspector(model_dir, device=device, threshold=threshold,
                                   uncertain_margin=cfg.inference.uncertain_margin)
        for path in expand_paths(args.paths):
            pred = inspector.predict(path, crop=args.crop)
            score = "  -  " if pred.score is None else f"{pred.score:.3f}"
            print(f"{pred.status:8} {score} {'uncertain' if pred.uncertain else '         '}  {path}")


if __name__ == "__main__":
    main()
