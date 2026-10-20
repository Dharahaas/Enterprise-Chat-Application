"""Serve the static frontend locally without exposing backend or local data files."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

FRONTEND_ROOT = Path(__file__).resolve().parent
BLOCKED_COMPONENTS = {"backend", "data", "secrets"}


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_ROOT), **kwargs)

    def send_head(self):
        url_path = unquote(urlsplit(self.path).path).replace("\\", "/")
        components = PurePosixPath(url_path).parts
        if any(
            component.startswith(".") or component in BLOCKED_COMPONENTS
            for component in components
        ):
            self.send_error(404)
            return None
        return super().send_head()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the restricted local frontend server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5500)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(f"Frontend available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
