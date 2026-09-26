# n8n workflows

Two workflows, both self-contained JSON you import into n8n:

- `ask-the-room.json`: the **project workflow**. A live log of every spoken question (what the rig heard,
  how it understood it, what it said, how long it took), plus a health check every 5 minutes. See below.
- `ask-the-repo.json`: a chat bot that answers questions about the **codebase**, for the team. See *Ask the Repo*.

## Ask the Room (project workflow)

Visitors ask out loud; nobody types. Everything real-time runs on the Jetson, offline: clicker, mic,
Silero VAD, whisper.cpp, the rule parser with Qwen2.5 1.5B (llama.cpp) for what it can't read, answers,
voice and laser (`main.py`). n8n runs on the laptop, around the rig, and never slows an answer down.

```
rig: clicker ─▶ speech ─▶ Whisper ─▶ rules │ Qwen ─▶ answer ─▶ speak + laser ─▶ POST webhook (background)
                                                                                  │
n8n: Rig heard a question ─▶ Settings ─▶ Read the question ─▶ went wrong? ─▶ fail the execution
     every 5 min ──────────▶ Settings ─▶ GET /state ─▶ GET Qwen /health ─▶ Check the room ─▶ problems? ─▶ fail
                                             └─ rig unreachable ─▶ fail the execution
```

- **Rig heard a question.** After every clicker press `main.py` posts
  `{heard, intent, object, understood_by (qwen | rules), qwen_ms, answer, point_at, laser_err_cm, online,
  click_to_laser_s, ...}` to the webhook in a background thread. Each question is one execution, so
  *Executions* reads like a transcript of the demo. It fails (red) when nothing was heard, an open question
  got the canned reply, click to laser took over 3 s, Qwen took over 1 s, or the laser landed over 3 cm off.
- **Health check** flags Qwen not answering (questions fall back to the rule parser), perception under
  10 fps, no internet (canned reply for open questions, Piper voice), objects with status UNKNOWN, and
  objects that left the table (GONE). Hidden objects (INSIDE, UNDER, HELD) are normal and not flagged.
- Swap the *Alert* nodes for Slack or Discord when there's a channel.
- **SMS** stays on the rig's own `/sms` route (Twilio signature check and whitelist). n8n isn't involved.
- **Detector:** the default YOLO-World v2 (`detect.model` in `config.yaml`) while the team compares models.
  Nothing in the workflow depends on which model it is.

### Setup (self-hosted on the laptop, about 2 minutes)

1. Start n8n: `npx n8n`, or `docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n`.
   Open http://localhost:5678.
2. **Create workflow → ⋯ → Import from File** → `n8n/ask-the-room.json`. No credentials are needed.
   **Save**, then **Publish** (the webhook only listens once published).
3. Open the **Settings** node and set `rig_url` and `qwen_url`:
   - The Jetson over USB-C: `http://192.168.55.1:8000` and `http://192.168.55.1:8081` (the defaults)
   - `python main.py --fake` on the same laptop: `http://127.0.0.1:8000` and `:8081`. Use `127.0.0.1`,
     not `localhost`: n8n resolves `localhost` to IPv6 `::1`, and the rig only listens on IPv4.
   - n8n in Docker, rig on the same laptop: `http://host.docker.internal:8000`
4. On the Jetson, start Qwen so the laptop can reach it: `QWEN_HOST=0.0.0.0 scripts/qwen_server.sh`.
5. In the rig's `config.yaml`, point the rig at the webhook (the laptop is `192.168.55.100` over USB-C):
   `n8n: webhook_url: http://192.168.55.100:5678/webhook/ask-the-room`. Empty turns reporting off.

Tested with n8n 2.40.7 against `main.py --fake --no-voice` and llama-server with Qwen2.5 1.5B on the
laptop: spoken questions driven through `Room.voice_loop` (a good answer, the canned reply, nothing heard)
and the scheduled check.

# Ask the Repo (n8n)

`ask-the-repo.json` is an n8n chat bot that answers questions about this codebase, for example "how does
the world model decide an object is UNDER something?", "where's the laser calibration?" or "what's left to
build?". It uses Claude Sonnet 5 and reads the repo live from GitHub, so it always sees the latest `main`.

```
chat message ─▶ Load CONTEXT.md (GitHub, main) ─▶ Project Guide (AI agent, Claude Sonnet 5, 10-message memory)
                                                     ├─ read_repo_file   open any file
                                                     ├─ list_repo_dir    list a folder
                                                     └─ recent_commits   last 15 commits on main
```

Each message starts from `CONTEXT.md` at the repo root, which is the project map. The agent then opens
the files it needs before answering. **Keep `CONTEXT.md` current**: the bot is only as up to date as that
file and the code. You don't need to re-import the workflow after editing it.

## Setup (one person, about 5 minutes)

1. Import the workflow. In n8n, choose **Create workflow → ⋯ → Import from File** and pick `n8n/ask-the-repo.json`.
   Tested against n8n 2.40.7. Any recent 2.x should work.
2. Create two credentials:
   - **Anthropic**: an API key from console.anthropic.com. Select it on the *Claude Sonnet 5* node.
   - **GitHub API**: a [fine-grained token](https://github.com/settings/personal-access-tokens/new)
     scoped to `anshulk08/asktheroom` with **Contents: read** (Metadata: read is added automatically).
     The repo is private, so this is required. Select it on *Load CONTEXT.md*, *read_repo_file*,
     *list_repo_dir* and *recent_commits*.
3. **Save**, then **Publish** (older versions call this *Activate*).
4. Open *When chat message received* and copy the **public chat URL**. Share that link with the team.
   Nobody else needs an n8n account.

The chat URL has to be reachable by everyone, so n8n Cloud is the easiest host. If you self-host
(`docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n`), it's only reachable on your own
machine unless you put it behind a tunnel.

## Notes

- Anyone with the chat URL can use the bot and spend your Anthropic credits, and it can read the private
  repo through your token. Share the link only within the team. If that's a concern, switch the chat
  trigger's authentication to Basic Auth.
- The bot can't open the spec (a Claude Doc). It points people to the link instead.
- To change the model, pick a different one on the *Claude Sonnet 5* node. The answering rules are in the
  *Project Guide* node's system message.
