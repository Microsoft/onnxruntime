# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tests for the isolated loopback collector."""

import importlib.util
import pathlib
import socket
import unittest.mock

spec = importlib.util.spec_from_file_location(
    "telemetry_transport_runner", pathlib.Path(__file__).with_name("telemetry-transport-test.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CollectorTest(unittest.TestCase):
    def test_bind_does_not_resolve_hostname(self):
        with (
            unittest.mock.patch.object(socket, "getfqdn", side_effect=AssertionError("Unexpected DNS lookup")),
            runner.Collector("upload") as collector,
        ):
            self.assertEqual(collector.server_address[0], "127.0.0.1")
            self.assertEqual(collector.server_name, "127.0.0.1")
            self.assertGreater(collector.server_port, 0)
            self.assertEqual(collector.socket.getsockname(), collector.server_address)


if __name__ == "__main__":
    unittest.main()
