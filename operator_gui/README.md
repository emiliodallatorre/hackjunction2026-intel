# Operator GUI

A [NiceGUI](https://nicegui.io) dashboard for the brick sorting robot. It has:
- **Auto cycle**, which sorts bricks until you switch it off;
- a **Run action...** prompt for typing manual instructions to the model;
- the live inspection camera with the ROI drawn on it;
- the latest verdict with its heatmap;
- running counts;
- a table of every sorted brick, with a thumbnail.

It is an add-on. It drives [run_model.py](../src/hackjunction2026_intel/run_model.py) as it is and changes nothing in the app. To remove it, delete this folder.

## Run

On the robot host, set `MODEL_EXPORT` to a Physical AI Studio runtime export ZIP or extracted
directory, then run from the repository root:

```bash
uv run --extra robot --with-editable ./operator_gui operator-gui --export "$MODEL_EXPORT"
```

Then open `http://<robot-host>:8080`. It also works over the tailnet.

To try the GUI without hardware, use a dry run. It simulates the robot, the camera and the model, gives random verdicts, and takes about 1.5 s per stage:

```bash
uv run --with-editable ./operator_gui operator-gui --dry-run
```

| Option | Default | Meaning |
|---|---|---|
| `--export` / `--dry-run` | | The Studio OpenVINO export (a .zip or a folder), or a simulation. |
| `--action-duration` | `15` | Control seconds a typed action runs for. |
| `--home` | `./home_pose.json` | Home pose file that Reset moves the arm to. |
| `--host` | `0.0.0.0` | Interface to serve on. |
| `--port` | `8080` | Port to serve on. Studio already uses 7860. |
| `--history` | `operator_gui/history/` | Folder for `bricks.csv` and the thumbnails. It is ignored by git. |

## Controls

- **Auto cycle** starts one process that builds a single `PolicyRunner` and runs `run_model.run_cycle` again and again: move the brick to the centre, inspect it, place it.
  - The icon spins while the loop runs.
  - Clicking it again lets the current cycle **finish** and then ends the loop. Until the loop ends, the prompt stays grayed out and shows "Shutting down...".
  - A cycle that fails ends the loop. Examples: no brick, an inconclusive inspection, a timeout.
- **Run action...**: type an instruction and press Enter or the send button. It runs through `run_model.run_task` for `--action-duration` seconds. It is not counted as a brick. While it runs, the send button becomes a stop button.
- **Reset** (gray) moves the arm back to its home pose with `run_model.reset_robot`. It is grayed out while anything else runs.
  - Teach the home pose first with `run_model --export ... reset --save-home`, run from the folder the GUI starts in.
  - Or pass the file with `--home`.
- **Esc / Space** stop anything **immediately** by sending SIGINT, which run_model already treats as an interruption: the policy stops and the arm is told to hold its measured position.
  - If the runner has not exited after 5 s, it is terminated, then killed.
  - While stopping, a **Force kill** button appears. It skips the hold.

> These are software stops. Keep the robot's physical emergency stop within reach.

## How it plugs in

```
browser ◀── NiceGUI ── ui/dashboard.py, ui/widgets.py   (only these import NiceGUI)
                          │ polls Status every 100 ms
                     controller.py ── history.py ──▶ history/bricks.csv + thumbnails/
                          │ spawn · SIGUSR1 = finish cycle · SIGINT = stop now
                          ▼            ▲ JSON lines on a dedicated pipe (events.py)
      python -m operator_gui.runner (--export P | --dry-run) (cycles | action TEXT)
                          │ run_model.PolicyRunner + run_cycle / run_task, unchanged
                          └─ observer.py (real) or dryrun.py (simulated)
```

**The workflow is only observed.** `observer.py` replaces `anomaly_detection.InspectionStation`, which `run_cycle` imports at call time, with a subclass in the runner process. The subclass returns exactly what the original returns. It also publishes:
- camera frames, limited to 10 fps;
- every scored view;
- the final decision.

A logging handler matches the cycle's task instructions to report the current stage. If publishing fails, the error is swallowed, so the robot code is never disturbed.

**A brick is counted** when the arm is sent to a dish during an auto cycle.

| File | Role |
|---|---|
| `events.py` | Wire protocol shared by the runner and the controller. |
| `runner.py` | Child process: runs the auto-cycle loop or one action, and reports events. |
| `observer.py` | Read-only taps on the inspection station; loaded in the runner only. |
| `dryrun.py` | Simulated `PolicyRunner` and `InspectionStation` for `--dry-run`. |
| `controller.py` | Starts, finishes and stops runs, follows events, holds `Status`. No UI code. |
| `history.py` | CSV and thumbnail store of sorted bricks. |
| `ui/theme.py` | Colours, fonts and a few CSS classes. Layout is done with Tailwind. |
| `ui/widgets.py` | One class per panel, each with `refresh(status)`. |
| `ui/dashboard.py` | Composes the page and handles notifications and keyboard STOP. |
