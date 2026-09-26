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
