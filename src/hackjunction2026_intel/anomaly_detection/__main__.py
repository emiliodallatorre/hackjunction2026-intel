"""Inspection demo: the robot loop with keyboard prompts in place of the robot.

    capture frame -> no brick?        -> error message
                  -> score near 0.5?  -> ask to rotate the brick and look again
                  -> otherwise        -> GOOD or DEFECTIVE

After `brick-ad train`:

    uv run python -m hackjunction2026_intel.anomaly_detection           # Enter = inspect, q + Enter = quit
    uv run python -m hackjunction2026_intel.anomaly_detection --show    # also show the crop with the heatmap

In the robot code, replace the `input()` prompts with robot events (brick presented,
brick rotated) and the printed messages with actions (sort, re-grasp, rotate).
"""

import argparse

import cv2

from . import Decision, InspectionStation, Verdict
from .imaging import overlay_heatmap, put_text


def show(decision: Decision) -> None:
    pred = decision.looks[-1]
    view = pred.patch.copy() if pred.anomaly_map is None else overlay_heatmap(pred.patch, pred.anomaly_map)
    view = cv2.resize(view, (512, 512))
    put_text(view, decision.verdict.name)
    cv2.imshow("inspection", view)
    cv2.waitKey(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="config file (default: $BRICK_AD_CONFIG or anomaly_detection/config.yaml)")
    parser.add_argument("--max-looks", type=int, default=3, help="views per brick before deciding anyway")
    parser.add_argument("--show", action="store_true", help="show the crop and heatmap in a window")
    args = parser.parse_args()

    def rotate(decision: Decision) -> None:
        """Robot: rotate the wrist to show another side, return once the brick is still."""
        if args.show:
            show(decision)
        print(decision.message)
        input("  rotate the brick, then press Enter ")

    # Load once at startup: opens the camera with the config's resolution and locked
    # focus / exposure / gain / white balance, and loads the model.
    with InspectionStation(args.config, max_looks=args.max_looks) as station:
        if station.inspector.presence is None:
            print("Note: model has no empty-ROI reference, so empty detection is off "
                  "(collect empty frames, then `brick-ad prepare` + `brick-ad fit-empty`).")
        while input("\nPresent a brick and press Enter (q to quit) ").strip().lower() != "q":
            decision = station.inspect(rotate=rotate)
            if args.show:
                show(decision)
            print(decision.message)
            # Robot pipeline: act on the verdict here.
            if decision.verdict is Verdict.EMPTY:
                pass  # re-grasp
            elif decision.verdict is Verdict.DEFECTIVE:
                pass  # defective bowl
            else:
                pass  # good bowl
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
