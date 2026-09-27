# 0008: Thing identity: one object, one thing:N

Status: implemented Sat 26 Sep, before the 6 PM freeze. Rebirth is off in `config.yaml` after review (below); one-per-kind merging is on with the settle check (spec 0007, itself off by default); Grok label belief is off by default. Unit-tested; tried live on the Jetson for about 6 minutes (results and handoff at the end).

## Problem

One real object kept turning into several `thing:N`. Measured live on the Jetson: 13 minutes, `proposals.kind: change`, hands active.

| Measure | Result |
|---|---|
| Unnamed `thing:N` births | 39 |
| Births within 5 cm of an earlier thing | 31 |
| Births at the bottom edge, where hands enter | 24 |
| With `proposals.kind: yoloe` | 3 → 17 things in 2 minutes |

Grok's settle check (spec 0007) called most of the extras phantom. But single verdicts flip-flop: the same notebook got 9 phantom and 33 agree. Verdicts that stayed the same over several checks were right: the tracker's "phone" was a roll of tape.

**Cause.** Rule (b) in `core/things.py` only revived a thing lost in place with no hand involved. After a pick-up, a false pick-up (a passing hand hid it) or an edge exit, the object seen again became a new `thing:N`. Appearance re-id (`reid`) is off on the rig, so nothing else linked the two.

## What is built

### 1. Rebirth: a thing lost recently and seen again at its spot is itself

`thing_identity:` section, at the end of `config.yaml` (`core/things.py` `IdentityConfig`).

| Key | Value | Meaning |
|---|---|---|
| `rebirth_s` | 60 (0 = off) | how recently the thing was lost |
| `rebirth_cm` | 5 | how close to its old spot |

A thing lost (UNKNOWN, GONE or HELD) within `rebirth_s` and seen again, of its size, within `rebirth_cm` of where it was last seen or picked up from, is that thing again. With several candidates, the nearest wins; on a tie, the most recently seen. Tests: `tests/test_thing_identity.py`.

### 2. One object per kind

`grok_check.merge_same: true`, and `grok_check.bind_conf` raised from 0.7 to 0.9 (at 0.7–0.8 on the rig, Grok named one thing "phone" and "phone charger" on two checks).

When Grok names a new unnamed thing at `bind_conf` or more, with a label a lost thing already answers to, the new thing is folded into the older one (`world.merge_same_kind` → `confirm_same`, a `CORRECTED` event). All of these must hold:

- no other mark in that frame has a similar label;
- the lost thing is not among the frame's marks;
- the older thing was not seen after the newer one was confirmed (seen together means two objects);
- the newer thing is VISIBLE;
- nothing is inside or under either thing.

### 3. Grok label belief (off by default)

`grok_check:` keys `belief_enabled` (false), `min_obs` (3), `name_p` (0.6), `name_margin` (0.2), `retire_p` (0.7), `half_life_s` (120), `veto_s` (120). The dashboard state gives each unnamed thing its top-3 `belief` ([[label, share], ...]); the check summary lists `retired`. The `grok_checks` table gains `guesses`, `not_object` and `held` (added to older databases on start).

**What Grok returns.** For each unnamed mark, up to 3 label guesses with probabilities, plus `not_object`: the probability that the mark is clutter (a cable, part of the desk or monitor, a shadow, a hand, part of another object). Grok also returns `held` per mark.

**Tally.** A per-thing tally sums these over checks, with a 120 s half-life. A phantom verdict counts as `not_object` at its confidence.

**Name.** After at least 3 checks, once the top label has p ≥ 0.6 and leads the next by ≥ 0.2, the thing is named: the alias is bound and marked `named_by: grok`.

**Retire as clutter.** After at least 3 checks, once `not_object` ≥ 0.7. Only for a thing that is VISIBLE, has no taught name, has nothing inside or under it, and has had no hand contact since the frame. It leaves state (`merged_into` itself), and no new thing is born on its box, grown by 5 cm, for 120 s (`veto_s`). A box, not a radius around the centre: a cable is long. A thing whose mark moved more than 5 cm between checks starts its tally over (it was carried, or the tracker moved its name to another object); a named thing shows no `belief`.

**Why a tally, not one verdict.** Verbalized VLM confidence is overconfident. Top-k verbalized probabilities calibrate far better than a single number (Tian et al. 2023, https://arxiv.org/abs/2305.14975; Xiong et al. 2024, https://arxiv.org/abs/2306.13063). grok-4.3 gives no logprobs (xAI docs: `logprobs` is ignored for grok-4.20 and newer).

**`held` is recorded only, never acted on.** Frontier VLMs are near chance at telling contact from near (arXiv 2511.20162). The settle check only runs once hands have been gone 1.5 s. The tracker keeps deciding HELD.

**Store.** New `grok_checks` columns: `guesses` (JSON), `not_object`, `held`. Older databases get them by `ALTER TABLE` on start.

### 4. Phone (BLE)

The bridge (`mobile/bridge/`) sends per thing:

| Field | Meaning |
|---|---|
| `g` | the guess, or the top belief label |
| `gc` | its confidence |
| `as: "grok"` | the name was bound by Grok, not taught |

It drops unnamed things lost more than 10 minutes ago. The iOS app shows guesses hedged ("phone charger?") and never says "your X" for a Grok-bound name.

## Review before enabling (Sat 26 Sep, afternoon, on branch `integration`)

An outside review reproduced four faults; until each is fixed and measured, the demo config keeps these off
(`rebirth_s: 0`, `merge_same: false`, `belief_enabled: false`, `grok_check.enabled: false`):

| Fault | Reproduced | Needed before enabling |
|---|---|---|
| Rebirth hands a HELD thing's identity to a different object of its size set down at its spot | yes | never reborn from HELD while its hand is elsewhere; treat a match as "possibly the same", not the same |
| One-per-kind merge folds a visitor's second mug into yours (seen apart is not one object) | by construction | a suggested match only, or only for props marked unique |
| Belief normalises over the supplied guesses only: three checks at 1% "mug" name it; 1% "not an object" retires it | yes | keep the unguessed mass; require absolute evidence to name or retire |
| A Grok reply that lands after a name was taught becomes the primary label (`named_by: grok`) | yes | re-check eligibility atomically when applying the reply |

Replays (`eval/score_clip.py`) cannot catch these: they run with `visual=None` and no settle checker, and
rebirth only runs with it in the replay config. Identical replay scores show merge compatibility only.

## Not built (proposals)

- A combined proposer: YOLOE and change detection together.
- Grok decides HELD: one capped check about 0.5 s after `PICKED_UP`, with the hand boxes drawn.
- A start-up ignore list of static clutter.
- Stricter `openworld.still_s`. At 1.0, 12 tests break. Tune it on the rig from a recorded clip: record with `python -m eval.raw_record --out data/clips/clutter_1 --seconds 120`, then compare false births per minute with `eval/score_clip.py`.

## Rig config advice

- Set the table outline: `python -m core.table --outline`.
- Use `proposals.kind: change`.

## Privacy

No change to what leaves the device. Belief uses the same settle-check frame (spec 0007); it only asks Grok for more fields per mark.

## Acceptance tests

| # | Test | Pass | Status |
|---|---|---|---|
| I1 | `.venv/bin/python -m pytest -q tests/test_thing_identity.py tests/test_grok_check.py` | all pass: off by default in code; rebirth after a pick-up and a false pick-up; too long ago, elsewhere or another size is a new thing; the nearest lost thing wins; one-per-kind merge, but two of a kind in one frame or seen together stay two, and a lost thing still among the marks is not merged; belief names and retires only past its thresholds | pass (laptop): 39 across both files, including belief naming, split guesses never naming, retiring clutter with the spot vetoed, no retiring after a touch, off by default, and the old-table migration |
| I2 | Rig: record a clip with hand activity, replay with `eval.score_clip` before (`rebirth_s: 0`, `merge_same: false`) and after | false births per minute clearly lower after, identity changes not higher | not yet run |
| I3 | Rig: live dashboard, 5 minutes of hand activity over a few objects | `thing:N` count stays near the number of real objects | partly: see Live results. Clutter is retired and names stick, but births at the bottom edge (an arm) are not fewer |

## Live results (Jetson, Sat 26 Sep, 14:12–14:22 EDT)

Branch `thing-identity` at 885de26 in a scratch copy (`~/askroom_grokcheck` on the Jetson, container `askroom_grokcheck_live`, dashboard `http://192.168.55.1:8001`). It used the rig's `config.yaml` plus `grok_check.enabled: true`, `belief_enabled: true`, `thing_identity.rebirth_s: 60` and `proposals.kind: change`, with the fine-tuned `models/askroom-yolo26s-pro9000.engine`. Someone was at the desk handling things.

| Measure | Run 1 (before the fixes below, ~2 min) | Run 2 (885de26, ~3.5 min) |
|---|---|---|
| fps | 15.0 | 15.0 |
| `thing:N` births | 11 by 14:14:33 | 17 |
| Retired as clutter | 4 | 7 |
| Named by belief or check | 2 ("blue bottle", "blue tape") | 2 ("smartphone", "white cable") |
| Things left in state at the end | 7 | 10 (2 VISIBLE) |
| Settle-check latency | 1.8–4.2 s | 1.75 s min, 3.2 s mean, 5.6 s max (17 checks) |

Verdicts in run 2: agree 27, named 13, phantom 22, relabel 1, unmarked 46, unsure 28.

**Found and fixed in run 1** (885de26):
- The veto was a 5 cm radius around the retired thing's centre, so another stretch of the same cable was born as a new thing right away. It now covers the whole box, grown by `veto_cm`.
- One `thing:N` was named "blue packet" at one spot, then its tally, built up at another spot, retired it. The tracker had moved the name between objects. A tally now starts over when its mark moves more than 5 cm.

**Still open** (for the next agent):
- **Births are not fewer.** Most births in both runs are at the bottom edge (y about 25–34 cm, x about 17–24 cm), where the person's arm enters. They have hand contact, so retiring rightly refuses them. The fix belongs in the proposer: `table_area` outline (`python -m core.table --outline`, not yet set on the rig) or a hand/arm mask for `change` proposals.
- **A cable can be named.** Grok named `thing:14` "white cable" at 0.9. The prompt tells Grok to ignore cables only for `unmarked`. Consider asking for `not_object` high for cables in marks too, or refusing to bind "cable".
- **Phone view.** The BLE bridge and the iOS app were not tried against this run; only unit and simulator tests.
- **Rebirth and merge were not measured separately.** Run I2 (replay a recorded clip with each switch off and on) to tell which part helps.

## Handoff: where everything is

**Branches (all pushed to `origin`, github.com/anshulk08/asktheroom)**

| Branch | What | Base |
|---|---|---|
| `thing-identity` | This spec: rebirth, one-per-kind merge, Grok belief and retiring, BLE bridge fields, docs | `overnight` plus the `grok-settle-check` commits (70da9cb, 0a83c1f, bbbdc1a, 7480526). Not based on `teammate-tasks`: merging needs a look at `core/things.py` and `config.yaml` |
| `mobile-app` | iOS app: shows `g`/`gc` hedged ("tape roll?"), never "your" for `as: "grok"`, hides `gc` < 0.5; `run.sh` bundle id fix | its own line; the bridge lives on `thing-identity` |
| `grok-settle-check` | spec 0007; 3 local commits there were not pushed from this session | `teammate-tasks` |

**Commits on `thing-identity`, oldest first**

| Commit | Change |
|---|---|
| 1a3229d | `core/things.py` `IdentityConfig`, `_reborn`: rebirth (part 1) |
| 14a9da6 | `World.merge_same_kind`, `GrokCheck._one_of_kind`/`_merge`, `bind_conf` 0.9 (part 2) |
| 996e281 | `tests/test_narration.py`: WALL0 was a fixed date that the 24 h prune outlived on Sep 26 |
| b48a01f | belief: schema `guesses`/`not_object`/`held`, `GrokCheck._believe`/`belief`/`_retire`, `World.retire_thing`, vetoes, `bind_alias(by="grok")`, `named_by` (part 3) |
| 905cdb9 | `mobile/bridge/bleproto.py`: `g`, `gc`, `as`, stale-thing drop, status `gk`; `mobile/PROTOCOL.md` |
| 8294c93 | this spec, spec 0007 effects, FEATURE_STATUS, CONTEXT, README privacy line |
| 885de26 | box veto, tally restart on a move, no `belief` on named things (from the live run) |

**Code map**
- `core/things.py`: `IdentityConfig`; `_reborn` (called from `_identify` after rule (b)); `merge_same_kind`; `retire_thing`; `_vetoes` checked in `_may_create`; `_alias_by` (who bound an alias) kept through `_bind`/`_absorb`; `_thing_json` adds `named_by`.
- `core/grok_check.py`: `GrokCheckConfig` belief keys; `SYSTEM`/`SCHEMA` ask for `guesses`, `not_object`, `held`; `guesses()`, `_p()`, `NOT_OBJECT`; `CheckStore` adds missing columns (`ADDED`); `_bind` → `_one_of_kind`/`_merge`/`_name`; `_believe` (tally, fade, restart past `MOVED_CM`, name or retire); `attach` puts `belief` on unnamed things in `state_json`.
- `mobile/bridge/bleproto.py`: `_best_guess` (guess vs top real belief label), `_stale_thing`, `status_msg` `gk`.
- Tests: `tests/test_thing_identity.py` (rebirth, merge, belief, veto, migration), `tests/test_grok_check.py`, `tests/test_mobile_protocol.py`. Full suite on the Mac: 1574 passed, 27 skipped.

**How to run it on the Jetson again** (never touch the teammate's `~/askroom` there)
1. From a checkout of `thing-identity`: `git ls-files > /tmp/f && rsync -a --files-from=/tmp/f ./ guru@192.168.55.1:askroom_grokcheck/`.
2. On the Jetson, in `~/askroom_grokcheck`: copy `~/askroom/config.yaml` and `~/askroom/table_cal.json` in, then append the `grok_check:` and `thing_identity:` sections from this branch with `enabled` and `belief_enabled` set to true.
3. Start `askroom:latest` with `--runtime=nvidia --network=host`, the video devices, `--env-file ~/askroom/.env`, `-v ~/askroom/models:/askroom/models:ro`, `-v $PWD:/askroom`, running `python3 main.py --camera /dev/v4l/by-id/usb-046d_0809_A1C0DC94-video-index0 --no-voice --port 8001`.
4. Watch `/state` (`grok_check.last`: `text`, `bound`, `retired`; each thing's `belief`, `named_by`) and `docker logs` lines with "retired" or "is now".
5. Clean up: `docker stop askroom_grokcheck_live`, then remove `data/` through a container (its files are root-owned), then `rm -rf ~/askroom_grokcheck`.

**Standing rules that shaped this** (AGENTS.md): no edits to the owned `core/world.py`, `core/detect.py`, `core/hands.py`, `core/table.py`, `core/relations.py`, `core/events.py`, `core/capture.py` (everything here is in `core/things.py`, a mixin, and `core/grok_check.py`); new config keys only in new or own sections at the end; the freeze after Sat 6 PM EDT allows only bug fixes, replay tuning and docs; xAI spend needs a team OK (each live settle check is about $0.003).
