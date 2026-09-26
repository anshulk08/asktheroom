# n8n workflows

Two workflows, both self-contained JSON you import into n8n:

- `ask-the-room.json`: the **project workflow**. Chat questions go to the rig and come back as text, and a health check runs every 5 minutes. See below.
- `ask-the-repo.json`: a chat bot that answers questions about the **codebase**. See *Ask the Repo*.

## Ask the Room (project workflow)

n8n sits around the rig: everything real-time runs in Python (`main.py`), and n8n talks to its HTTP API.

```
chat message ─▶ Settings (rig_url) ─▶ GET /healthz ─▶ POST /ask {text, source: "n8n"} ─▶ answer text
                                           └─ unreachable ─────────┴─▶ "I can't reach the rig at ..."
every 5 min  ─▶ Settings (rig_url) ─▶ GET /state ─▶ Check the room ─▶ problems? ─▶ fail the execution
                                           └─ unreachable ─▶ fail the execution ("Rig unreachable at ...")
```

- **Text only.** `source: "n8n"` tells the rig to answer without speaking or moving the laser, the
  same as texts. The rig still logs the question.
- **Health check** flags perception under 10 fps, no internet (the rig falls back to templates and
  Piper), objects with status UNKNOWN (lost track), and objects that left the table (GONE). Hidden objects
  (INSIDE, UNDER, HELD) are normal and not flagged. A problem makes the execution fail, so it shows red
  under *Executions*. Swap the two *Alert* nodes for Slack or Discord when there's a channel.
- **SMS** stays on the rig's own `/sms` route (Twilio signature check and whitelist). n8n isn't involved.
- **Detector:** the default YOLO-World v2 (`detect.model` in `config.yaml`) while the team compares models.
  Nothing in the workflow depends on which model it is.

### Setup (self-hosted on the laptop, about 2 minutes)

1. Start n8n: `npx n8n`, or `docker run -it --rm -p 5678:5678 -v n8n_data:/home/node/.n8n n8nio/n8n`.
   Open http://localhost:5678.
2. **Create workflow → ⋯ → Import from File** → `n8n/ask-the-room.json`. No credentials are needed.
3. Open the **Settings** node and set `rig_url`:
   - The Jetson over USB-C: `http://192.168.55.1:8000` (the default)
   - `python main.py --fake` on the same laptop: `http://127.0.0.1:8000`. Use `127.0.0.1`, not `localhost`:
     n8n resolves `localhost` to IPv6 `::1`, and the rig only listens on IPv4.
   - n8n in Docker, rig on the same laptop: `http://host.docker.internal:8000`
4. **Save**, then **Publish**. Open *When chat message received* for the chat URL.

Tested with n8n 2.40.7 against `main.py --fake --no-voice`: the chat questions, the unreachable-rig reply,
and both scheduled outcomes (healthy, rig down).

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
