# 0008: Thing identity: one object, one thing:N

Status: implemented Sat 26 Sep, before the 6 PM freeze. Rebirth is on in `config.yaml`; one-per-kind merging is on with the settle check (spec 0007, itself off by default); Grok label belief is off by default. Unit-tested; not yet measured on the rig.

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

**Retire as clutter.** After at least 3 checks, once `not_object` ≥ 0.7. Only for a thing that is VISIBLE, has no taught name, has nothing inside or under it, and has had no hand contact since the frame. It leaves state (`merged_into` itself), and no new thing is born within 5 cm of it for 120 s (`veto_s`).

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
| I3 | Rig: live dashboard, 5 minutes of hand activity over a few objects | `thing:N` count stays near the number of real objects | not yet run |
