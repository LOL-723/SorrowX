git reset --soft HEAD~1  删除最新提交记录(本地)
git push --force-with-lease 删除最新提交记录(远程)

Flow

START

  -> router_node
  
      -> agent_node -> END
      
      -> tool_selector_node -> tool_executor_node -> answer_node
      
      -> answer_node
      
  -> verifier_node
  
      -> finish -> END
      
      -> retry_answer -> answer_node
      
      -> retry_tool -> tool_selector_node
      
      -> retry_router -> router_node
      
      -> fail -> END

Uploaded-document retrieval is only available inside the Agent route through
the `retrieve_uploaded_document` tool. The normal tool route exposes only time,
calculator, and weather tools.

CLI daemon

Run a one-line daemon ping from cmd:

```cmd
sorrow ping
sorrow session new
sorrow session switch session_1
sorrow session list
sorrow session current
sorrow session del session_1
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

`sorrow run` and `sorrow trace` require a session. If no current session exists,
the CLI creates one automatically. `sorrow ping` and `sorrow shutdown` do not use
sessions.
