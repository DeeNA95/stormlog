"""Total request deadlines include DNS, headers, body and error-body reads."""

import contextlib
import json
import socket
import struct
import threading
import time
import urllib.error
from typing import Any, Iterator

import pytest

from stormlog.infer.completions_client import OpenAICompletionsClient
from stormlog.infer.http_transport import (
    ConnectError,
    CutResponseError,
    EndpointHTTPError,
    RequestDeadlineExceeded,
)
from stormlog.infer.openai_client import OpenAIChatCompletionsClient
from stormlog.infer.profile import classify_failure


def client(url: str, **options: Any) -> OpenAIChatCompletionsClient:
    return OpenAIChatCompletionsClient(
        endpoint=url, model="opaque", timeout_seconds=0.12, **options
    )


def complete(value: OpenAIChatCompletionsClient) -> Any:
    return value.complete(
        prompt="input", output_tokens=1, stream=False, stream_include_usage=False
    )


@pytest.mark.parametrize(
    "client_class", [OpenAIChatCompletionsClient, OpenAICompletionsClient]
)
@pytest.mark.parametrize(
    "url",
    [
        "http://user:password@localhost/v1/chat/completions",
        "http:///v1/chat/completions",
        "http://localhost:bad/v1/chat/completions",
        "http://localhost/v1/chat/completions#fragment",
        "file:///tmp/model",
    ],
)
def test_endpoint_validation_precedes_transport_creation(
    monkeypatch: pytest.MonkeyPatch,
    client_class: type[OpenAIChatCompletionsClient],
    url: str,
) -> None:
    def unexpected_transport() -> None:
        pytest.fail(
            "An invalid endpoint must be rejected before creating its transport"
        )

    monkeypatch.setattr(
        "stormlog.infer.openai_client.inference_opener", unexpected_transport
    )
    with pytest.raises(ValueError):
        client_class(endpoint=url, model="opaque", timeout_seconds=1)


def test_dns_deadline_is_bounded_without_sending_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stalled(*args: Any) -> list[Any]:
        time.sleep(0.3)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", stalled)
    started = time.perf_counter()
    with pytest.raises(urllib.error.URLError) as raised:
        complete(client("http://invalid.local/v1/chat/completions"))
    assert time.perf_counter() - started < 0.25
    assert isinstance(raised.value.reason, ConnectError)
    assert classify_failure(raised.value) == ("unreachable", None)


@contextlib.contextmanager
def trickling_socket(wire: bytes, *, trickle_headers: bool) -> Iterator[str]:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            connection.recv(65536)
            try:
                if not trickle_headers:
                    header, _, body = wire.partition(b"\r\n\r\n")
                    connection.sendall(header + b"\r\n\r\n")
                else:
                    body = wire
                for byte in body:
                    connection.sendall(bytes([byte]))
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        listener.close()
        thread.join(2)


def test_trickling_headers_cannot_extend_total_deadline() -> None:
    with trickling_socket(
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", trickle_headers=True
    ) as url:
        started = time.perf_counter()
        with pytest.raises(RequestDeadlineExceeded):
            complete(client(url, request_deadline_seconds=0.12))
        assert time.perf_counter() - started < 0.3


def test_error_body_deadline_preserves_rejected_http_status() -> None:
    with trickling_socket(
        b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 100\r\n\r\n" + b"x" * 100,
        trickle_headers=False,
    ) as url:
        with pytest.raises(EndpointHTTPError) as raised:
            complete(client(url, request_deadline_seconds=0.12))
    assert raised.value.status == 429
    assert classify_failure(raised.value) == ("rejected", 429)


def test_a_response_that_keeps_arriving_outlives_the_read_timeout() -> None:
    # --timeout bounds each read, not the request: without --request-deadline
    # a response that keeps arriving runs past it and completes.
    body = b'{"choices": [{"message": {"content": "ok"}}]}'
    wire = b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body
    with trickling_socket(wire, trickle_headers=False) as url:
        started = time.perf_counter()
        result = complete(client(url))
    assert time.perf_counter() - started > 3 * 0.12
    assert result.text == "ok"


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_a_request_deadline_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="request deadline"):
        client("http://127.0.0.1:1/v1/chat/completions", request_deadline_seconds=value)


def _sse(text: str) -> bytes:
    choice = {"delta": {"content": text}, "finish_reason": None}
    return b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n"


def _read_request(connection: socket.socket) -> None:
    """Read the whole request, so closing sends FIN rather than RST."""
    data = b""
    while b"\r\n\r\n" not in data:
        data += connection.recv(65536)
    head, _, body = data.partition(b"\r\n\r\n")
    length = int(head.lower().split(b"content-length:")[1].split(b"\r\n")[0])
    while len(body) < length:
        body += connection.recv(65536)


@contextlib.contextmanager
def chunked_stream(pieces: int, *, delay: float, end: str) -> Iterator[str]:
    """A chunked SSE stream like vLLM's that ends ``done``, ``eof`` or ``reset``."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            _read_request(connection)
            try:
                connection.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                for index in range(pieces):
                    piece = _sse(f"t{index} ")
                    connection.sendall(b"%x\r\n%s\r\n" % (len(piece), piece))
                    time.sleep(delay)
                if end == "done":
                    connection.sendall(b"0\r\n\r\n")
                    return
                connection.sendall(b"40\r\npartial")
                if end == "reset":
                    time.sleep(0.05)
                    linger = struct.pack("ii", 1, 0)
                    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
            except (BrokenPipeError, ConnectionResetError):
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        listener.close()
        thread.join(3)


def stream(value: OpenAIChatCompletionsClient) -> Any:
    return value.complete(
        prompt="input", output_tokens=8, stream=True, stream_include_usage=False
    )


def test_a_deadline_during_a_chunked_stream_is_a_timeout() -> None:
    with chunked_stream(20, delay=0.05, end="done") as url:
        with pytest.raises(RequestDeadlineExceeded) as raised:
            stream(client(url, request_deadline_seconds=0.3))
    assert classify_failure(raised.value) == ("timeout", 200)


@pytest.mark.parametrize("end", ["eof", "reset"])
def test_a_chunked_stream_cut_short_keeps_its_status(end: str) -> None:
    with chunked_stream(2, delay=0.0, end=end) as url:
        with pytest.raises(CutResponseError) as raised:
            stream(client(url))
    assert classify_failure(raised.value) == ("error", 200)


@contextlib.contextmanager
def served(wire: bytes) -> Iterator[str]:
    """Answer one request with ``wire`` all at once."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        connection, _ = listener.accept()
        with connection:
            _read_request(connection)
            connection.sendall(wire)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1/chat/completions"
    finally:
        listener.close()
        thread.join(3)


def test_invalid_utf8_in_a_stream_is_replaced() -> None:
    body = b'data: {"choices": [{"delta": {"content": "ok\xff"}}]}\n\ndata: [DONE]\n\n'
    wire = (
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body)
    ) + body
    with served(wire) as url:
        result = stream(client(url))
    assert result.text == "ok\ufffd"


def test_a_long_error_body_keeps_its_start() -> None:
    body = b"model overloaded; " + b"x" * 100_000
    wire = (
        b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: %d\r\n\r\n" % len(body)
    ) + body
    with served(wire) as url:
        with pytest.raises(EndpointHTTPError) as raised:
            complete(client(url))
    assert raised.value.status == 500
    assert "model overloaded" in str(raised.value)
