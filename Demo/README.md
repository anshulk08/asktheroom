# Demo assets

Placeholder. These are the assets to add before the Devpost submission (Sun Sep 27, 8 AM EDT). Keep files small (GIFs under 10 MB). Don't commit raw recordings or anything with visitors' faces or voices.

| File | What | Source | Status |
|---|---|---|---|
| `architecture.png` | Architecture diagram: camera → detector → world model → answers → voice + laser, with the local/online boundary marked (only TTS text and SMS leave the device) | Redraw the ASCII diagram in the top-level `README.md` | to add |
| `shell-game.gif` | The shell game from overhead: keys under the notebook, notebook into the box, box slid across, question asked, laser lands on the box. 10–15 s, with captions | Screen-record the dashboard (`/video` overlay) during a rehearsal (PLANS E2) | to add |
| `dashboard.png` | Dashboard with the world-state list showing a nested chain (keys → notebook → box) | Screenshot of `/` during the shell game | to add |
| `eval-hidden.png` | Bar chart: hidden-object accuracy (covered, inside, inside_box_moved), full system vs the `current_frame`, `last_seen` and `nearest_object` baselines, **real trials only** | `python -m eval.report --trials trials --out report.md` after PLANS F5; spec 0004 | to add (needs real trials) |
| `eval-interpreter.png` | Bar chart: interpreter accuracy per set, rules only vs rules + Qwen3-1.7B (48/64 → 58/64 on the laptop; Jetson numbers if measured) | `python scripts/eval_understand.py` | to add |
| `latency.png` | Question-to-laser latency distribution from live runs | `speech_end_to_laser_s` / `click_to_laser_s` in the n8n log, or `questions.latency_ms` in `data/events.db` | to add (needs rig runs) |

Rules:

- Charts show real numbers only. Never chart the synthetic eval as accuracy.
- Label laptop numbers as laptop numbers.
