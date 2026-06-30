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
sorrow shutdown
```

Run the same command from PowerShell:

```powershell
.\sorrow.ps1 ping
.\sorrow.ps1 shutdown
```

The first ping starts the core daemon automatically. The CLI process exits after
the command finishes; the daemon remains running until it is stopped manually:
