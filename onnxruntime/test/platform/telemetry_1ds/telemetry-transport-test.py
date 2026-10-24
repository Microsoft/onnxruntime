# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Exercise the real telemetry provider and native HTTP transport without external uploads."""

import argparse
import http.server
import os
import pathlib
import socketserver
import subprocess
import tempfile
import threading
import time

if os.name == "nt":
    import winreg


class Collector(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def server_bind(self):
        # HTTPServer resolves a hostname here; this numeric-only fixture needs no DNS.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address

    def __init__(self, mode):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode = mode
        self.received = threading.Event()
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = []
        self.markers = set()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        server = self.server
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with server.lock:
            server.requests += 1
            sequence = server.requests
            if self.path.split("?")[0] != "/collect" or not body:
                server.errors.append("Expected a nonempty telemetry POST to /collect")
            for marker in (b"synthetic-before-disable", b"synthetic-while-disabled", b"synthetic-after-enable"):
                if marker in body:
                    server.markers.add(marker)
        server.received.set()
        if server.mode == "pending":
            server.release.wait(35)
            return
        self.send_response(503 if server.mode == "retry" and sequence == 1 else 200)
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()


def cleanup_test_registry(pid):
    if os.name != "nt":
        return

    def remove(path):
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_READ | winreg.KEY_WRITE)
        except FileNotFoundError:
            return
        with key:
            while winreg.QueryInfoKey(key)[0]:
                remove(path + "\\" + winreg.EnumKey(key, 0))
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)

    remove(r"SOFTWARE\Microsoft\onnxruntime-tests\telemetry-" + str(pid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("upload", "threaded", "retry", "pending", "disabled", "unit-disabled", "ci-disabled", "reload"),
    )
    args = parser.parse_args()
    with (
        tempfile.TemporaryDirectory(prefix="ort-telemetry-") as home,
        Collector(args.mode) as server,
    ):
        env = os.environ.copy()
        for name in (
            "CI",
            "TF_BUILD",
            "GITHUB_ACTIONS",
            "GITLAB_CI",
            "CIRCLECI",
            "TRAVIS",
            "JENKINS_URL",
            "CODEBUILD_BUILD_ID",
            "BUILDKITE",
            "TEAMCITY_VERSION",
            "APPVEYOR",
            "BITBUCKET_BUILD_NUMBER",
            "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
            "ORT_RUNNING_UNIT_TESTS",
            "ORT_DISABLE_TELEMETRY",
        ):
            env[name] = "0"
        for name in ("HOME", "USERPROFILE", "XDG_CACHE_HOME", "LOCALAPPDATA", "APPDATA"):
            env[name] = home
        for name in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(name, None)
        env["no_proxy"] = env["NO_PROXY"] = "*"
        env["ORT_TEST_TELEMETRY_COLLECTOR"] = f"http://127.0.0.1:{server.server_port}/collect"
        suppression = {
            "disabled": "ORT_DISABLE_TELEMETRY",
            "unit-disabled": "ORT_RUNNING_UNIT_TESTS",
            "ci-disabled": "CI",
        }
        if args.mode in suppression:
            env[suppression[args.mode]] = "1"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with subprocess.Popen(
                [args.executable, args.mode, args.library],
                cwd=home,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            ) as child:
                output = ""
                try:
                    deadline = time.monotonic() + 30
                    if args.mode == "pending":
                        if not server.received.wait(20):
                            raise AssertionError("Native transport never delivered the pending request")
                        child.stdin.write("shutdown\n")
                        child.stdin.flush()
                    output, _ = child.communicate(timeout=max(0.1, deadline - time.monotonic()))
                finally:
                    if child.poll() is None:
                        child.kill()
                    if not output:
                        output, _ = child.communicate()
                    print(output, end="")
                    cleanup_test_registry(child.pid)
                assert child.returncode == 0, f"Telemetry child failed: {child.returncode}"
            expected_cycles = 2 if args.mode == "reload" else 1
            assert output.count("PASS telemetry lifecycle") == expected_cycles, "Teardown verification did not execute"
            if args.mode in suppression:
                assert server.requests == 0, "Suppressed telemetry made an HTTP request"
                assert not list(pathlib.Path(home).rglob("*")), "Suppressed telemetry persisted files"
            else:
                assert server.requests >= expected_cycles, "Test did not exercise real native HTTP requests"
                assert b"synthetic-before-disable" in server.markers, "Initial ORT event did not reach the collector"
                assert b"synthetic-while-disabled" not in server.markers, "Disabled ORT event was transmitted"
                if args.mode != "pending":
                    assert b"synthetic-after-enable" in server.markers, "Re-enabled ORT event was not transmitted"
                if os.name == "nt":
                    assert output.count("TRANSPORT WinHTTP") == expected_cycles, "Wrong Windows transport"
                elif os.uname().sysname == "Linux":
                    assert output.count("TRANSPORT curl") == expected_cycles, "Wrong Linux transport"
            assert not server.errors, "; ".join(server.errors)
        finally:
            server.release.set()
            server.shutdown()
            thread.join()


if __name__ == "__main__":
    main()
