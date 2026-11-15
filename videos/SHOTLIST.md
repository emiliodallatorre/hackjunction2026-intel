# Shot list for presentation.html

Videos are **fullscreen backgrounds** — the text rides on top of the footage,
kinetic-caption style, with a dark gradient scrim so captions stay readable.
Shoot wide and leave the lower third relatively calm; that's where captions sit.

## Workflow

1. Record the clip, name it as in the table below, drop it in `videos/`.
2. Run `./prepare.sh` — it transcodes to browser-playable H.264 in `videos/web/`.
3. Reload `presentation.html`.

Step 2 is **required**: phone footage is 4K HEVC (H.265), which browsers won't
play in a `<video>` tag. `prepare.sh` downscales to 1080p H.264, strips audio,
and skips files already converted. A hatched "record & run prepare.sh"
placeholder shows for any clip not yet present.

### Speeding clips up

`prepare.sh` also speeds each clip up — the multipliers live in the `speed_for`
table at the top of the script. This is free: it rewrites frame timestamps
inside the encode pass that was already happening, so there is no second
re-encode. Current settings turn the 35 s one-take run into ~12 s at 3×, and
screen captures run at 4× since they're just skimmed.

```sh
./prepare.sh 04-full-run.mp4 2     # try this one clip at 2x instead
FORCE=1 ./prepare.sh               # redo everything after editing the table
```

After changing a speed, update that slide's `data-dur` in `presentation.html`
to the new clip length, so autoplay neither cuts the clip off nor sits on a
looped frame. The script prints each output's duration.

## Clips

| file | slide | content | status |
|------|-------|---------|--------|
| `01-intro.mp4` | 1 | hero shot of the robot + camera | ⬜ to record |
| `02-pick.mp4` | 4 | picking the brick up | ✅ recorded |
| `02-place.mp4` | 5 | placing it in the yellow tray | ✅ recorded |
| `03-orange-tray.mp4` | 6 | the orange tray appears / brick into orange | ✅ recorded |
| `04-full-run.mp4` | 8 | full run end-to-finish, unicorn distractor, bricks everywhere — ONE TAKE (deck adds the red REC badge) | ✅ recorded |
| `05-anomaly-sidebyside.mp4` | 9 | RGB feed beside the anomaly heatmap; person draws an angry face on a brick and puts it back | ⬜ to record |
| `06-training.mp4` | 10 | Physical AI Studio screen capture (cameras / episodes / training) | ⬜ to record |
| `07-dashboard.mp4` | 15 | speed-scroll of the long Claude prompt, then the dashboard firing the alert | ⬜ to record |

`videos/random videos/` holds your unused takes — the deck ignores that folder.

## Using the deck

Open `presentation.html` in Chrome or Firefox (double-click works, no server).

- `→` / `Space` / click — next slide, `←` back, `Home` restart
- `A` — narration via the browser's built-in text-to-speech (robotic voice for
  π-0.5; a deeper voice on the training slide, per the script). The dashboard
  slide also fires a loud WebAudio siren.
- `P` — autoplay: advances when narration ends (audio on) or on the script's
  timecodes (audio off).
- `F` fullscreen, `H` hides the HUD — use both, plus a screen recorder, to
  export the final video.

## Still to fill in

- Slide 14: the `N×` compile-speedup number is a red EDIT-ME placeholder; no
  measured number exists in `benchmark/` yet.
- Benchmark numbers already inserted (real, from `benchmark/results/` and
  `pi05_benchmark.out`): π-0.5 iGPU 105 ms p50 vs CPU 8 104 ms (77×), 476
  actions/s, 6.3 % of the 30 Hz budget; PatchCore CPU 110 / iGPU 54 / NPU
  182 ms p50, energy 4 383 / 1 711 / 2 535 mJ.
