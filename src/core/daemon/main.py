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
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                try:
                    message = decode_message(line)
                    handler_result = dispatch_rpc(message, state)
                except ProtocolError as exc:
                    handler_result = _protocol_error_result(exc)

                writer.write(encode_message(handler_result.response))
                await writer.drain()
                if handler_result.should_shutdown:
                    shutdown_event.set()
                    break
        finally:
            writer.close()
            await writer.wait_closed()

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


if __name__ == "__main__":
    raise SystemExit(main())
