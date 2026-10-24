# Isolated telemetry lifecycle tests

Ordinary ORT tests suppress telemetry. These dedicated desktop CTests exercise the
real provider and SDK without sending telemetry to an external collector.
They are registered when 1DS telemetry is enabled in a native desktop test build;
cross-compilation, Android, iOS, GDK, and WindowsStore are excluded because the
fixtures require local subprocesses and desktop dynamic loading.

From a configured build directory:

```text
cmake --build . --config RelWithDebInfo --target onnxruntime_telemetry_mutex_lifetime_test onnxruntime_telemetry_transport_test onnxruntime_telemetry_transport_fixture
ctest -C RelWithDebInfo -R "^onnxruntime_telemetry_(shutdown|transport)_" --output-on-failure
```

Use the configuration selected for the build instead of `RelWithDebInfo` where
appropriate. The normal build-driver CTest invocation also runs these tests.

## Mutex lifetime

The four `onnxruntime_telemetry_shutdown_*` tests initialize the actual
`Env::Default()` singleton on the main thread or an application thread, with
telemetry enabled or disabled. Test-only SDK debug-lock and native-join observers
verify that the worker finishes before the debug mutex's lifetime ends. An
unexpected upload fails immediately through a transport with no network support.
The observations do not change the production SDK objects.

## Real native transport

The `onnxruntime_telemetry_transport_*` tests use the actual SDK HTTP factory:
WinHTTP on Windows and curl on Linux. Apple builds use their selected SDK
transport. A forwarding `IHttpClient` module observes terminal callbacks and
forces every request URL to the parent-owned IPv4 loopback collector; it does not
replace native I/O, cancellation, or worker management.

The fixture alone compiles the provider source with configuration and manager
observation hooks. Production configuration, public APIs, and CI suppression are
unchanged. The fixture's WinHTTP source differs only in selecting direct access
instead of automatic proxy discovery. The runner removes proxy environment
variables and sets both curl proxy-bypass variables. The server binds explicitly
to `127.0.0.1` on an ephemeral port, performs no hostname lookup, and never
redirects requests. A separate runner regression test rejects DNS lookups during
collector construction.

Each child has a fresh temporary home/cache. Windows children redirect HKCU to a
temporary test key before initializing the real device-ID implementation and
restore/delete it after provider destruction. Test event fields are synthetic;
payloads are inspected in memory and are not printed.
The parent also removes its child's exact test registry subtree after a failure
or forced termination.

| Mode | Required observations |
| --- | --- |
| `upload` | Real ORT event arrives; disabling suppresses a distinct event; re-enabling delivers another event. Every native request completes exactly once. |
| `threaded` | Same checks after singleton initialization on an application thread. |
| `retry` | Collector returns 503 once; the real SDK retries and succeeds. |
| `pending` | Parent receives the full native request and withholds its response. Child exits normally, and provider teardown cancels and drains the request. |
| `disabled` | Environment opt-out creates no SDK transport, files, or requests. |
| `unit-disabled` | Unit-test suppression creates no SDK transport, files, or requests. |
| `ci-disabled` | CI suppression creates no SDK transport, files, or requests. |
| `reload` (Windows/Linux) | A DLL/shared-library fixture constructs and explicitly destroys a real provider before unload, then repeats after reload. |

Apple builds run all seven subprocess transport modes, but do not register the
reload test. The SDK's Apple transport registers Objective-C classes, and its
image remains resident after `dlclose`; a second `dlopen` reuses its static state
instead of exercising an unload/reload. Loading another SDK copy also registers
duplicate Objective-C classes. This fixture does not support that lifecycle.

The reload fixture shuts its provider down and exits its calling thread
**before** entering the loader lock, allowing thread-local destructors to run.
It does not claim that `FreeLibrary` alone safely tears down an active SDK: a live
MSVC worker can retain the DLL, and unloading code still used by callbacks is not
a valid shutdown protocol. Singleton lifetime is exercised by the executable
cases; controlled provider lifetime is exercised by the reload case.

The pinned SDK's factory uses non-destructed static storage (a downstream patch
matching the SDK's PAL pattern), not a heap allocation lost on each unload.
After draining its provider, the isolated reload fixture also releases curl's
global state before unloading its private image. Production SDK code must not do
that: unlike this test host, it cannot know whether other curl users remain.
Leak detection stays enabled for the reload test.

The parent bounds each child, and CTest adds an outer timeout. Missing requests,
missing teardown verification, wrong Windows/Linux transport selection, and
suppression accidentally applied to enabled cases fail rather than skip.

These tests use local HTTP, not TLS. They do not replace certificate-policy
tests, native platform coverage beyond the selected transport, or the distinct Windows
`ExitProcess`/DLL-process-detach regression introduced by #33291. A zero SDK
teardown-upload budget does not bound cancellation or worker draining.

## Windows process termination

The three `onnxruntime_telemetry_{process_exit,normal_destruction,explicit_unload}_test`
cases use a DLL-owned provider and a mock pending uploader. Like the transport
fixture, this DLL compiles a generated provider copy: test-only hooks install the
mock after suppressed initialization and verify shared-state cleanup after
shutdown. No test peer or friendship is needed in the production class.
The destructor, Windows shutdown guard, and lock ordering are unchanged.
The parent requires both successful child exit and a teardown-verification signal.

```text
cmake --build . --config RelWithDebInfo --target onnxruntime_telemetry_shutdown_test
ctest -C RelWithDebInfo -R "^onnxruntime_telemetry_(process_exit|normal_destruction|explicit_unload)_test$" --output-on-failure
```
