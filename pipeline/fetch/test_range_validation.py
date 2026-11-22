"""A wrong range with the right length must never be accepted as source data."""
import http.server
import socketserver
import threading

import pytest

from range_client import RangeClient


@pytest.mark.parametrize("retry_after,minimum_wait", [
    ("5", 5),
    ("Thu, 01 Jan 1970 00:01:47 GMT", 7),
    ("invalid", 3),
])
def test_rate_limit_cooldown_retry_after_and_successful_request_pacing(monkeypatch, retry_after, minimum_wait):
    import range_client
    from types import SimpleNamespace

    class Clock:
        now = 0.0

        def sleep(self, delay):
            self.now += delay

    clock = Clock()
    monkeypatch.setattr(range_client, "time", SimpleNamespace(
        monotonic=lambda: clock.now, time=lambda: 100 + clock.now, sleep=clock.sleep))
    arrivals = []
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            arrivals.append(clock.now)
            if len(arrivals) == 1:
                self.send_response(429)
                self.send_header("Retry-After", retry_after)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", "bytes 0-3/100")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"abcd")

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = RangeClient(f"http://127.0.0.1:{server.server_address[1]}/soil", max_retries=2, backoff_base=0)
            client.min_request_interval = 2
            client.rate_limit_cooldown = 3
            assert client.fetch(0, 3) == b"abcd"
            assert client.fetch(0, 3) == b"abcd"
            assert arrivals[1] - arrivals[0] >= minimum_wait
            assert arrivals[2] - arrivals[1] >= 2
        finally:
            server.shutdown()


@pytest.mark.parametrize("status,content_range,body", [
    (200, None, b"abcd"),
    (206, "bytes 1-4/100", b"abcd"),
    (206, "bytes 0-3/3", b"abcd"),
    (206, None, b"abcd"),
    (206, "bytes 0-3/100", b"abcdefgh"),
])
def test_rejects_unverified_or_oversized_range(status, content_range, body):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            if content_range:
                self.send_header("Content-Range", content_range)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RangeClient(f"http://127.0.0.1:{server.server_address[1]}/raster", max_retries=1)
            with pytest.raises(RuntimeError):
                client.fetch(0, 3)
        finally:
            server.shutdown()
