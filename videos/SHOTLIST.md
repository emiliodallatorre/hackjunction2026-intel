# Shot list for presentation.html

Drop each clip in this folder with the exact filename. The deck picks them up
automatically on reload — a hatched placeholder shows until the file exists.
Any 16:9 mp4 (h264) works; clips loop muted.

| file | slide | content | suggested length |
|------|-------|---------|------------------|
| `01-intro.mp4` | 1 | hero shot of the robot + camera | 5 s |
| `02-pick-place.mp4` | 4 | pick the brick → yellow tray | 6 s |
| `03-orange-tray.mp4` | 5 | the orange tray appears / brick into orange | 5 s |
| `04-full-run.mp4` | 7 | full run end-to-finish, unicorn distractor, bricks dropped everywhere — ONE TAKE (slide adds the red REC badge) | 10 s |
| `05-anomaly-sidebyside.mp4` | 8 | RGB feed side-by-side with the anomaly heatmap; person draws angry face on a brick and puts it back | 15 s |
| `06-training.mp4` | 9 | Physical AI Studio screen capture (cameras / episodes / training) | 10 s |
| `07-dashboard.mp4` | 14 | speed-scroll of the long Claude prompt, then the dashboard firing the alert | 12 s |

## Using the deck

Open `presentation.html` in Chrome/Firefox (double-click works, no server needed).

- `→` / `Space` / click — next slide, `←` back, `Home` restart
- `A` — toggle narration (browser text-to-speech; robotic voice fits π-0.5,
  slide 9 switches to a deeper voice per the script). The dashboard slide also
  fires a comically loud WebAudio siren when audio is on.
- `P` — autoplay: advances when narration ends (audio on) or on the script's
  timecodes (audio off). Combine with `F` fullscreen + screen recorder to
  produce the final video.

## Still to fill in

- Slide 13: the `N×` compile-speedup number is a red EDIT-ME placeholder —
  no measured number exists yet in `benchmark/`.
- Benchmark numbers already inserted (real, from `benchmark/results/` +
  `pi05_benchmark.out`): π-0.5 iGPU 105 ms p50 vs CPU 8 104 ms (77×), 476
  actions/s, 6.3 % of the 30 Hz budget; PatchCore CPU 110 / iGPU 54 / NPU
  182 ms p50, energy 4 383 / 1 711 / 2 535 mJ.
