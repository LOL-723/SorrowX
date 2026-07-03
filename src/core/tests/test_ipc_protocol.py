import unittest

from ipc.protocol import (
    JsonRpcError,
    decode_message,
    encode_message,
    is_event_push,
    make_event_push,
    make_error_response,
    make_request,
    make_result_response,
    read_event_push,
    read_result_response,
    validate_request,
)


class IpcProtocolTests(unittest.TestCase):
    def test_request_round_trip_uses_ndjson(self) -> None:
        request = make_request("core.ping", {"client": "test"}, request_id="abc")

        encoded = encode_message(request)
        decoded = decode_message(encoded)

        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(decoded["jsonrpc"], "2.0")
        self.assertEqual(decoded["id"], "abc")
        self.assertEqual(decoded["method"], "core.ping")
        validate_request(decoded)

    def test_result_response_is_read_by_matching_id(self) -> None:
        response = make_result_response("abc", {"ok": True})

        result = read_result_response(response, expected_id="abc")

        self.assertEqual(result, {"ok": True})

    def test_error_response_raises_json_rpc_error(self) -> None:
        response = make_error_response("abc", code=-32601, message="missing")

        with self.assertRaises(JsonRpcError) as ctx:
            read_result_response(response, expected_id="abc")

        self.assertEqual(ctx.exception.code, -32601)
        self.assertEqual(ctx.exception.message, "missing")

    def test_event_push_round_trip_uses_ndjson(self) -> None:
        event = {"type": "run.started", "run_id": "run-1"}
        encoded = encode_message(make_event_push(event))
        decoded = decode_message(encoded)

        self.assertTrue(is_event_push(decoded))
        self.assertEqual(read_event_push(decoded), event)


if __name__ == "__main__":
    unittest.main()
