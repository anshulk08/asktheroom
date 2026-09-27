"""Generates n8n/ask-the-room.json, the project workflow (the live question log and the health check).

The JSON is generated: edit this script, not the JSON. Standard library only. From the repo root:

    python3 scripts/gen_n8n_workflow.py      # writes n8n/ask-the-room.json

Then re-import n8n/ask-the-room.json in n8n (Import from File, then Publish; see n8n/README.md) and set
the Settings node again. Node ids are uuid5 of the node names, so regenerating gives a clean diff.
"""
import json
import uuid
from pathlib import Path

NS = uuid.UUID("5b0e3c1e-7a51-4c55-9d7e-a5c7e0b0a001")
OUT = Path("n8n") / "ask-the-room.json"     # run from the repo root


def uid(s: str) -> str:
    return str(uuid.uuid5(NS, s))


nodes, conns = [], {}


def node(name, type_, version, pos, params, **extra):
    n = {"parameters": params, "type": type_, "typeVersion": version, "position": list(pos),
         "id": uid(name), "name": name}
    n.update(extra)
    nodes.append(n)
    return name


def link(src, dst, out=0):
    m = conns.setdefault(src, {"main": []})["main"]
    while len(m) <= out:
        m.append([])
    m[out].append({"node": dst, "type": "main", "index": 0})


def setnode(name, pos, fields, keep=False):
    return node(name, "n8n-nodes-base.set", 3.4, pos, {
        "assignments": {"assignments": [
            {"id": uid(name + k), "name": k, "value": v, "type": "string"} for k, v in fields.items()]},
        "includeOtherFields": keep, "options": {}})


def ifnode(name, pos, expr, operation="true"):
    """IF on one boolean expression: output 0 when it's `operation` (true/false), output 1 otherwise."""
    return node(name, "n8n-nodes-base.if", 2.2, pos, {
        "conditions": {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2},
                       "conditions": [{"id": uid(name + "-cond"), "leftValue": expr, "rightValue": "",
                                       "operator": {"type": "boolean", "operation": operation,
                                                    "singleValue": True}}],
                       "combinator": "and"},
        "options": {}})


SET = "$('Settings').first().json"
RIG = SET + ".rig_url"
FALLBACK = "I can tell you where things are, what happened to them, or what changed."   # voice/local_llm.py
NOT_HEARD = "Sorry, I didn't catch that."                                               # main.py

# ---------------------------------------------------------------- triggers + shared settings
heard = node("Rig heard a question", "n8n-nodes-base.webhook", 2, (0, 0), {
    "httpMethod": "POST", "path": "ask-the-room", "responseMode": "onReceived", "options": {}},
    webhookId=uid("heard-webhook"))
sched = node("Every 5 minutes", "n8n-nodes-base.scheduleTrigger", 1.2, (0, 400), {
    "rule": {"interval": [{"field": "minutes", "minutesInterval": 5}]}})
settings = setnode("Settings", (240, 200), {
    "rig_url": "http://192.168.55.1:8000",
    "webhook_token": "",           # = n8n.token in the rig's config.yaml; "" = don't check
    "discord_webhook_url": "",     # "" = alerts fail the execution instead (red under Executions)
    "gone_grace_min": "10",        # GONE (carried off the table) is only a problem after this long
    "unknown_grace_min": "2",      # UNKNOWN (lost track) likewise
}, keep=True)
link(heard, settings)
link(sched, settings)
route = node("Spoken question?", "n8n-nodes-base.if", 2.2, (460, 200), {
    "conditions": {"options": {"caseSensitive": True, "leftValue": "", "typeValidation": "loose", "version": 2},
                   "conditions": [{"id": uid("route-cond"), "leftValue": "={{ $json.body }}",
                                   "rightValue": "", "operator": {"type": "object", "operation": "exists",
                                                                  "singleValue": True}}],
                   "combinator": "and"},
    "options": {}})
link(settings, route)

# ---------------------------------------------------------------- spoken question: what the rig heard, understood, did
token = ifnode("Token OK?", (680, 0),
               "={{ !$json.webhook_token || ($json.headers || {})['x-askroom-token'] === $json.webhook_token }}")
link(route, token, 0)
node("Reject: bad token", "n8n-nodes-base.stopAndError", 1, (900, 160), {
    "errorMessage": "=Rejected a post to the webhook: bad token (header x-askroom-token doesn't match "
                    "Settings → webhook_token). Nothing was processed."})
link(token, "Reject: bad token", 1)

read = node("Read the question", "n8n-nodes-base.code", 2, (900, -60), {"jsCode": f"""\
// main.py posts one of these for every question it answers (n8n.webhook_url in config.yaml):
// {{heard, intent, object, understood_by: grok|qwen|rules|gate, qwen_ms (the model's time, Grok or Qwen),
//   answer, point_at, action, laser_err_cm, online, mode: asked|overheard, ignored_since_last, thinking_cue,
//   stt_ms, ask_s, record_transcribe_s, t, click_to_laser_s (asked, clicker) or speech_end_to_laser_s (overheard)}}
// ignored_since_last counts overheard speech the rig dropped since its last answer; that text is never sent.
// thinking_cue: the answer was slow, so the rig said "Let me look." first (demo.thinking_cue_s).
const q = $json.body;
const problems = [];
if (!q.heard) problems.push('heard nothing: mic level, or the visitor spoke before the click?');
if (q.answer === {json.dumps(FALLBACK)})
  problems.push(q.online ? 'open question got the fallback (Grok failed or timed out?)'
                         : 'open question got the fallback (rig offline: switch to the phone hotspot)');
const asked = q.click_to_laser_s != null;
const lat = asked ? q.click_to_laser_s : q.speech_end_to_laser_s;
const latName = asked ? 'click to laser' : 'end of speech to laser';
if ((lat || 0) > 3) problems.push(`${{latName}} took ${{lat}} s (target 3 s)`);
const modelLimit = q.understood_by === 'grok' ? 2000 : 1000;     // Grok median ~850 ms, local Qwen ~115 ms
if ((q.qwen_ms || 0) > modelLimit) problems.push(`${{q.understood_by}} took ${{q.qwen_ms}} ms to read it`);
if (q.laser_err_cm != null && q.laser_err_cm > 3) problems.push(`laser landed ${{q.laser_err_cm}} cm off`);
// Rolling latency over the last 20 answers (kept in the workflow's static data: production runs only,
// not "Execute workflow" test runs).
const st = $getWorkflowStaticData('global');
if (lat != null) st.lat = (st.lat || []).concat([lat]).slice(-20);
const xs = [...(st.lat || [])].sort((a, b) => a - b);
const pct = p => xs.length ? xs[Math.min(xs.length - 1, Math.floor(p * xs.length))] : null;
const last20 = {{ n: xs.length, median_s: pct(0.5), p90_s: pct(0.9) }};
const what = q.object ? `${{q.intent}} ${{q.object}}` : (q.intent || '-');
return [{{ json: {{
  summary: `[${{q.mode || '-'}}] "${{q.heard}}" -> ${{what}} (${{q.understood_by || '-'}}) -> "${{q.answer}}"`
           + (lat != null ? ` in ${{lat}} s` : '') + (q.thinking_cue ? ' (said "let me look" first)' : '')
           + (q.ignored_since_last ? ` (${{q.ignored_since_last}} overheard ignored before it)` : '')
           + (xs.length >= 5 ? ` [last ${{xs.length}}: median ${{last20.median_s}} s, p90 ${{last20.p90_s}} s]` : ''),
  ok: problems.length === 0, problems, last20, ...q }} }}];
"""})
link(token, read, 0)
qbad = ifnode("Question went wrong?", (1120, -60), "={{ $json.ok }}", "false")
link(read, qbad)
qmsg = setnode("Alert text: question", (1340, -140), {
    "alert": "=Ask the Room, question needs a look: {{ $json.summary }}: {{ $json.problems.join('; ') }}"})
link(qbad, qmsg, 0)

# ---------------------------------------------------------------- schedule: health check
state = node("Get room state", "n8n-nodes-base.httpRequest", 4.2, (680, 400), {
    "url": f"={{{{ {RIG} }}}}/state", "options": {"timeout": 5000}},
    onError="continueErrorOutput")
link(route, state, 1)
check = node("Check the room", "n8n-nodes-base.code", 2, (1120, 320), {"jsCode": """\
// /state is {state: world.state_json(), last_answer, server_t}. Hidden objects (INSIDE, UNDER, HELD)
// are normal during a demo. A stalled detector, no internet (no Grok), a lost track or an object carried
// off the table are worth a look. state.online is the rig reaching api.x.ai (net.check_host). Lost and
// carried-off objects only count once they have lasted (a pocketed pill bottle is part
// of the demo). entity.last_seen and server_t are both the rig's wall clock (time.time()).
const cfg = $('Settings').first().json;
const r = $('Get room state').first().json;
const s = r.state || {};
const now = r.server_t || Date.now() / 1000;
const graceMin = (v, d) => { const x = parseFloat(v); return Number.isFinite(x) ? x : d; };
const goneMin = graceMin(cfg.gone_grace_min, 10), lostMin = graceMin(cfg.unknown_grace_min, 2);
const minutes = e => e.last_seen == null ? Infinity : (now - e.last_seen) / 60;
const ago = e => e.last_seen == null ? 'never seen' : `${Math.round(minutes(e))} min`;
const problems = [];
if ((s.fps || 0) < 10) problems.push(`perception at ${s.fps || 0} fps (want 10+): is the detector running?`);
if (!s.online) problems.push("rig can't reach Grok (api.x.ai): only the rule parser and templates answer, "
                             + "voice is Piper, camera questions wait, texts (SMS) can't go out. "
                             + 'Switch the rig to the phone hotspot');
const ents = s.entities || [];
const lost = ents.filter(e => e.status === 'UNKNOWN' && minutes(e) > lostMin);
if (lost.length) problems.push(`lost track of: ${lost.map(e => `${e.name} (${ago(e)})`).join(', ')}`);
const gone = ents.filter(e => e.status === 'GONE' && minutes(e) > goneMin);
if (gone.length) problems.push(`off the table for over ${goneMin} min: `
                               + gone.map(e => `${e.name} (${e.edge || 'edge'}, ${ago(e)})`).join(', '));
return [{ json: { ok: problems.length === 0, problems, fps: s.fps, online: s.online,
                  gone_recent: ents.filter(e => e.status === 'GONE' && minutes(e) <= goneMin).map(e => e.name),
                  checked: new Date().toISOString() } }];
"""})
link(state, check, 0)
bad = ifnode("Problems?", (1340, 320), "={{ $json.ok }}", "false")
link(check, bad)
rmsg = setnode("Alert text: room", (1560, 300), {
    "alert": "=Ask the Room, room needs a look: {{ $json.problems.join('; ') }}"})
link(bad, rmsg, 0)
umsg = setnode("Alert text: rig unreachable", (900, 520), {
    # say it in words: n8n's Stop and Error swaps a message containing ECONNREFUSED & co. for its own text
    "alert": "=Ask the Room: rig unreachable at {{ " + RIG + " }} ({{ (m => /ECONNREFUSED/.test(m) ? "
             "'connection refused: is main.py running?' : /ETIMEDOUT|ESOCKETTIMEDOUT|timeout/i.test(m) ? "
             "'timed out' : /ENOTFOUND|EAI_AGAIN/.test(m) ? 'host not found' : /EHOSTUNREACH|ENETUNREACH/.test(m) "
             "? 'host unreachable: is the USB-C link up?' : m.replace(/\\bE[A-Z_]{4,}\\b/g, 'network error'))"
             "($json.error?.message || 'no response') }})"})
link(state, umsg, 1)

# ---------------------------------------------------------------- alerts: Discord if set, else fail the execution
discord = ifnode("Discord set?", (1780, 120), "={{ !!" + SET + ".discord_webhook_url }}")
for m in (qmsg, rmsg, umsg):
    link(m, discord)
post = node("Post to Discord", "n8n-nodes-base.httpRequest", 4.2, (2000, 40), {
    "method": "POST", "url": "={{ " + SET + ".discord_webhook_url }}",
    "sendBody": True, "specifyBody": "json",
    "jsonBody": "={{ JSON.stringify({ content: $json.alert.slice(0, 1900) }) }}",
    "options": {"timeout": 5000}})
link(discord, post, 0)
node("Alert: fail the execution", "n8n-nodes-base.stopAndError", 1, (2000, 220), {
    "errorMessage": "={{ $json.alert }}"})
link(discord, "Alert: fail the execution", 1)


# ---------------------------------------------------------------- notes
def note(name, pos, w, h, color, text):
    node(name, "n8n-nodes-base.stickyNote", 1, pos, {"content": text, "height": h, "width": w, "color": color})


note("About", (-40, -700), 540, 640, 4, """\
## Ask the Room: project workflow
Visitors **speak**; nobody types. The rig listens all the time and answers what's meant for it. Everything real-time runs on the Jetson (`main.py`), offline. n8n sits **around** it, on the laptop, and never slows an answer down.

**Rig heard a question** (top): for every question it answers, the rig posts what it heard, whether it was *asked* (clicker) or *overheard*, how it understood it (rules, Grok or the gate), what it said and how long it took, with the median and p90 of the last 20. Overheard speech it dropped is only counted, never sent. Each question is an execution here: a live log of the demo. Nothing heard, the fallback sentence, a slow answer (over 3 s to the laser), Grok slower than 2 s to read it or a laser miss raises an alert. With `webhook_token` set, posts without the matching `x-askroom-token` header are rejected.

**Every 5 min** (bottom): `GET /state` → flags a stalled detector, no internet (no Grok: switch to the hotspot), and objects lost (UNKNOWN over 2 min) or off the table (GONE over 10 min).

**Alerts** go to Discord when `discord_webhook_url` is set; otherwise they fail the execution (red under *Executions*). Texts (Twilio SMS) stay on the rig's own `/sms`.""")
note("Inside the rig", (560, -700), 780, 480, 7, """\
## What main.py runs (for reference, not executed here)
```
camera 30 fps ─▶ detector (YOLO, config prompts) ─▶ hand tracker
               ─▶ world model (rules) ─▶ event log (SQLite)
mic, always on ─▶ Silero VAD ─▶ whisper.cpp       "uh, show me my specs"
  (muted while the rig speaks; audio stays in RAM)
   ─▶ overheard gate: not for the rig → dropped, only counted
   ─▶ rule parser │ Grok for what it can't read   → WHERE glasses
   ─▶ camera questions → Grok with the frame (set-of-marks)
   ─▶ answer templates │ Grok answers open questions (offline: templates only)
   ─▶ slow? "Let me look." first
   ─▶ voice (ElevenLabs online, Piper offline) + laser, together
   ─▶ POST this workflow's webhook
clicker = "listen now" override
dashboard + /state  ◀─ FastAPI on :8000
```
Detector model: `detect.model` in `config.yaml`; nothing here depends on it.""")
note("Setup", (1380, -700), 480, 520, 6, """\
## Setup (self-hosted, laptop)
1. `npx n8n` (or Docker), open http://localhost:5678, **Import from File** → `n8n/ask-the-room.json`, **Publish**.
2. **Settings** node: `rig_url` is the Jetson over USB-C (`http://192.168.55.1:8000`). For `main.py --fake` on this laptop use `http://127.0.0.1:8000`, not `localhost` (n8n tries IPv6 `::1`).
3. Optional: `discord_webhook_url` (Discord channel → Integrations → Webhooks), `webhook_token` (same value as `n8n.token` in the rig's `config.yaml`), `gone_grace_min` / `unknown_grace_min`.
4. On the Jetson: `XAI_API_KEY` in `.env`, and in `config.yaml` set `n8n.webhook_url: http://192.168.55.100:5678/webhook/ask-the-room` (the laptop's USB-C address).
No n8n credentials needed.""")
note("Grok", (1380, -160), 480, 400, 3, """\
## Grok on the rig (xAI)
- **Reads** spoken questions the rules can't ("has anybody messed with my meds") → {kind, object}.
- **Answers** open questions from the world state, after the templates. Pill filter on every answer.
- **Looks**: questions about what the camera sees go with the current frame and numbered marks; Grok picks a mark, the laser points at it. **Recall** asks over saved frames.
- Offline: rules and templates only, camera questions wait. Keep a phone hotspot ready.

*Audio stays on the rig. Transcripts, and a frame for camera questions, go to Grok when online.*""")

wf = {"name": "Ask the Room (project workflow)", "nodes": nodes, "connections": conns,
      "settings": {"executionOrder": "v1"}, "pinData": {}, "meta": {"templateCredsSetupCompleted": True}}
with open(OUT, "w") as f:
    json.dump(wf, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(len(nodes), "nodes ->", OUT)
