from core.ipc.client import RpcCallResult


def format_ping_reply(*, host: str, port: int, rpc_result: RpcCallResult) -> str:
    result = rpc_result.result
    uptime_ms = result.get("uptime_ms", 0)
    version = result.get("server_version", "unknown")
    return (
        f"Reply from {host}:{port}: "
        f"time={max(0, round(rpc_result.elapsed_ms))}ms "
        f"uptime={_format_uptime(uptime_ms)} "
        f"version={version}"
    )


def _format_uptime(uptime_ms: object) -> str:
    try:
        return f"{float(uptime_ms) / 1000:.1f}s"
    except (TypeError, ValueError):
        return "unknown"
