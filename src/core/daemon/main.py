import argparse
import asyncio
from daemon.handlers import HandlerResult, dispatch_rpc
from daemon.state import DaemonState
from ipc.client import DEFAULT_HOST, DEFAULT_PORT
from ipc.protocol import PARSE_ERROR, ProtocolError, decode_message, encode_message, make_error_response


async def run_daemon(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    state = DaemonState(host=host, port=port)
    shutdown_event = asyncio.Event()

    async def handle_client(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        client_id = state.event_hub.register_client(writer)
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                request_trace_entry = None
                request_run_id = None
                try:
                    message = decode_message(line)
                    request_trace_entry = state.trace_recorder.prepare_client_to_core(message)
                    request_run_id = _run_id_from_request(message)
                    handler_result = dispatch_rpc(
                        message,
                        state,
                        client_id=client_id,
                    )
                except ProtocolError as exc:
                    handler_result = _protocol_error_result(exc)

                response_run_id = _run_id_from_response(handler_result.response)
                trace_run_id = response_run_id or request_run_id
                if trace_run_id is not None:
                    state.trace_recorder.write_prepared(
                        trace_run_id,
                        request_trace_entry,
                    )
                    state.trace_recorder.record_core_to_client_reply(
                        trace_run_id,
                        handler_result.response,
                    )
                writer.write(encode_message(handler_result.response))
                await writer.drain()
                if handler_result.should_shutdown:
                    shutdown_event.set()
                    break
        finally:
            state.event_hub.remove_client(client_id)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle_client, host=host, port=port)
    async with server:
        await shutdown_event.wait()
        server.close()
        await server.wait_closed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sorrow-core-daemon")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    try:
        asyncio.run(run_daemon(host=args.host, port=args.port))
    except OSError as exc:
        print(f"daemon failed to start: {exc}", flush=True)
        return 1
    return 0


def _protocol_error_result(exc: ProtocolError) -> HandlerResult:
    return HandlerResult(
        response=make_error_response(
            None,
            code=exc.code if exc.code else PARSE_ERROR,
            message=str(exc),
        )
    )


def _run_id_from_request(message: dict[str, object]) -> str | None:
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    run_id = params.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _run_id_from_response(message: dict[str, object]) -> str | None:
    result = message.get("result")
    if not isinstance(result, dict):
        return None
    run_id = result.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


if __name__ == "__main__":
    raise SystemExit(main())
