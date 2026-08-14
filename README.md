git reset --soft HEAD~1  删除最新提交记录(本地)

git push --force-with-lease 删除最新提交记录(远程)

Agent runtime

All Agent requests use one execution entry point:

```text
Web /agent/ask ──────┐
                     ├── AgentRuntime ── AgentGraphEngine
CLI → Daemon RPC ────┘       planner → select_step → AgentLoop
```

`/agent/ask` accepts `message` and an existing `session_id` as form fields. It
returns `run_id`, `status`, and `message`. Uploaded-document retrieval, RAG,
and per-request system prompts are not available on this route during the
AgentLoop transition.

CLI daemon

Run a one-line daemon ping from cmd:

```cmd
sorrow ping
sorrow session new
sorrow session switch session_1
sorrow session list
sorrow session current
sorrow session del session_1
sorrow UpdateMemory
sorrow CheckMemory
sorrow run "hello agent"
sorrow shutdown
sorrow trace
sorrow trace show run_id
```

Run the same command from PowerShell:

```powershell
.\sorrow.ps1 ping
.\sorrow.ps1 session new
.\sorrow.ps1 session switch session_1
.\sorrow.ps1 session list
.\sorrow.ps1 session current
.\sorrow.ps1 session del session_1
.\sorrow.ps1 UpdateMemory
.\sorrow.ps1 CheckMemory
.\sorrow.ps1 run "hello agent"
.\sorrow.ps1 shutdown
.\sorrow.ps1 trace
.\sorrow.ps1 trace show run_id
```

The first ping starts the core daemon automatically. The CLI process exits after
the command finishes; the daemon remains running until it is stopped manually.

`sorrow run` also starts the daemon automatically when needed. The CLI sends the
goal to the daemon with JSON-RPC over NDJSON, keeps the TCP connection open, and
prints Agent events streamed back by the daemon until the run finishes.

Session commands manage per-session memory and trace storage:

- `sorrow session new` creates a new session id and sets it as current.
- `sorrow session switch session_id` changes the current session.
- `sorrow session list` lists all known session ids and marks the current one.
- `sorrow session current` prints the current session id.
- `sorrow session del session_id` deletes that session's memory and trace data. The current session cannot be deleted; switch to another session first.
- `sorrow UpdateMemory` merges raw records not yet covered by the current session's concise memory summary. It retries records left uncovered by an earlier failed refresh; if every raw record is already covered, it does not call the model and displays `记忆已为最新状态`.
- `sorrow CheckMemory` displays only the current session's persisted memory summary. It does not call the model and does not append uncovered raw records.

`sorrow run` and `sorrow trace` require a session. If no current session exists,
the CLI creates one automatically. `sorrow ping` and `sorrow shutdown` do not use
sessions.
