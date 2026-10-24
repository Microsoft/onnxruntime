# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import collections
import collections.abc
import os
import typing
import warnings
import weakref
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

import numpy as np

from onnxruntime.capi import _pybind_state as C

if typing.TYPE_CHECKING:
    import numpy.typing as npt


def get_ort_device_type(device_type: str) -> int:
    if device_type == "cuda":
        return C.OrtDevice.cuda()
    elif device_type == "cann":
        return C.OrtDevice.cann()
    elif device_type == "cpu":
        return C.OrtDevice.cpu()
    elif device_type == "dml":
        return C.OrtDevice.dml()
    elif device_type == "webgpu":
        return C.OrtDevice.webgpu()
    elif device_type == "gpu":
        return C.OrtDevice.gpu()
    elif device_type == "npu":
        return C.OrtDevice.npu()
    else:
        raise Exception("Unsupported device type: " + device_type)


class OrtDeviceVendorId(IntEnum):
    """Vendor IDs aligned with OrtDevice::VendorIds in ortdevice.h."""

    NONE = 0x0000
    AMD = 0x1002
    NVIDIA = 0x10DE
    ARM = 0x13B5
    MICROSOFT = 0x1414
    HUAWEI = 0x19E5
    QUALCOMM = 0x5143
    INTEL = 0x8086


def get_vendor_id_for_device_type(device_type: str) -> OrtDeviceVendorId | None:
    if device_type == "cuda":
        return OrtDeviceVendorId.NVIDIA
    elif device_type == "dml":
        return OrtDeviceVendorId.MICROSOFT
    elif device_type == "cann":
        return OrtDeviceVendorId.HUAWEI
    elif device_type == "webgpu":
        return OrtDeviceVendorId.NONE
    else:
        return None


_GPU_GRAPH_ID_RUN_CONFIG_KEY = "gpu_graph_id"
# Mirrors InferenceSession::kGraphAnnotationSkip; core skips capture and replay for this ID.
_GRAPH_ANNOTATION_SKIP = -1


def _graph_annotation_id(run_options) -> int:
    """Return the effective ``gpu_graph_id`` for a run, defaulting to 0 when unset."""
    if run_options is None:
        return 0
    try:
        entry = run_options.get_run_config_entry(_GPU_GRAPH_ID_RUN_CONFIG_KEY)
    except RuntimeError:
        return 0
    if not entry:
        return 0
    try:
        return int(entry)
    except ValueError:
        raise ValueError(f"Run option '{_GPU_GRAPH_ID_RUN_CONFIG_KEY}' must be an integer, got {entry!r}.") from None


def _is_ortvalue_session_compatible(ortvalue, target_session) -> bool:
    """Whether ``target_session`` can share ``ortvalue``'s buffer.

    Sessionless values come from a shared allocator. A session-owned WebGPU buffer is
    usable by any session on the same WebGPU context.
    """
    if ortvalue._session is None or target_session is None or ortvalue._session is target_session:
        return True
    return ortvalue._is_webgpu_buffer and ortvalue._session.webgpu_context_id() == target_session.webgpu_context_id()


def _validate_ortvalue_session_compatibility(ortvalue, target_session, action) -> None:
    """Reject an OrtValue that the target session cannot share buffers with.

    ``action`` is the verb used in the message, e.g. ``"used with"`` or ``"bound to"``.
    """
    if not _is_ortvalue_session_compatible(ortvalue, target_session):
        raise ValueError(f"Session-scoped OrtValue must be {action} the session that created it.")


class AdapterFormat:
    """
    This class is used to create adapter files from python structures
    """

    def __init__(self, adapter=None) -> None:
        if adapter is None:
            self._adapter = C.AdapterFormat()
        else:
            self._adapter = adapter

    @staticmethod
    def read_adapter(file_path: os.PathLike) -> AdapterFormat:
        return AdapterFormat(C.AdapterFormat.read_adapter(file_path))

    def export_adapter(self, file_path: os.PathLike):
        """
        This function writes a file at the specified location
        in onnxrunitme adapter format containing Lora parameters.

        :param file_path: absolute path for the adapter
        """
        self._adapter.export_adapter(file_path)

    def get_format_version(self) -> int:
        return self._adapter.format_version

    def set_adapter_version(self, adapter_version: int) -> None:
        self._adapter.adapter_version = adapter_version

    def get_adapter_version(self) -> int:
        return self._adapter.adapter_version

    def set_model_version(self, model_version: int) -> None:
        self._adapter.model_version = model_version

    def get_model_version(self) -> int:
        return self._adapter.model_version

    def set_parameters(self, params: dict[str, OrtValue]) -> None:
        """Set adapter parameters for export."""
        self._adapter.parameters = {k: v._ortvalue for k, v in params.items()}

    def get_parameters(self) -> dict[str, OrtValue]:
        """Get adapter parameters as a dict of name -> OrtValue.

        On read instances, the returned OrtValues are zero-copy views; the
        backing memory stays alive as long as any returned OrtValue is referenced.
        """
        return {k: OrtValue(v) for k, v in self._adapter.parameters.items()}


def check_and_normalize_provider_args(
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None,
    provider_options: Sequence[dict[Any, Any]] | None,
    available_provider_names: Sequence[str],
):
    """
    Validates the 'providers' and 'provider_options' arguments and returns a
        normalized version.

    :param providers: Optional sequence of providers in order of decreasing
        precedence. Values can either be provider names or tuples of
        (provider name, options dict).
    :param provider_options: Optional sequence of options dicts corresponding
        to the providers listed in 'providers'.
    :param available_provider_names: The available provider names.

    :return: Tuple of (normalized 'providers' sequence, normalized
        'provider_options' sequence).

    'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

    The normalized result is a tuple of:
    1. Sequence of provider names in the same order as 'providers'.
    2. Sequence of corresponding provider options dicts with string keys and
        values. Unspecified provider options yield empty dicts.
    """
    if providers is None:
        return [], []

    provider_name_to_options = collections.OrderedDict()

    def set_provider_options(name, options):
        if name not in available_provider_names:
            warnings.warn(
                "Specified provider '{}' is not in available provider names.Available providers: '{}'".format(
                    name, ", ".join(available_provider_names)
                )
            )

        if name in provider_name_to_options:
            warnings.warn(f"Duplicate provider '{name}' encountered, ignoring.")
            return

        normalized_options = {str(key): str(value) for key, value in options.items()}
        provider_name_to_options[name] = normalized_options

    if not isinstance(providers, collections.abc.Sequence):
        raise ValueError("'providers' should be a sequence.")

    if provider_options is not None:
        if not isinstance(provider_options, collections.abc.Sequence):
            raise ValueError("'provider_options' should be a sequence.")

        if len(providers) != len(provider_options):
            raise ValueError("'providers' and 'provider_options' should be the same length if both are given.")

        if not all(isinstance(provider, str) for provider in providers):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all(isinstance(options_for_provider, dict) for options_for_provider in provider_options):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options, strict=False):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, {})
            elif (
                isinstance(provider, tuple)
                and len(provider) == 2
                and isinstance(provider[0], str)
                and isinstance(provider[1], dict)
            ):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


class Session:
    """
    This is the main class used to run a model.
    """

    def __init__(self, enable_fallback: bool = True):
        # self._sess is managed by the derived class and relies on bindings from C.InferenceSession
        self._sess = None
        self._enable_fallback = enable_fallback
        # Captured graphs retain buffer signatures without extending the IOBinding lifetime.
        self._captured_graph_bindings: dict[int, tuple[weakref.ref, tuple]] = {}

    def get_session_options(self) -> C.SessionOptions:
        "Return the session options. See :class:`onnxruntime.SessionOptions`."
        return self._sess_options

    def get_inputs(self) -> Sequence[C.NodeArg]:
        "Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._inputs_meta

    def get_outputs(self) -> Sequence[C.NodeArg]:
        "Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._outputs_meta

    def get_overridable_initializers(self) -> Sequence[C.NodeArg]:
        "Return the inputs (including initializers) metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._overridable_initializers

    def get_modelmeta(self) -> C.ModelMetadata:
        "Return the metadata. See :class:`onnxruntime.ModelMetadata`."
        return self._model_meta

    def get_input_memory_infos(self) -> Sequence[C.OrtMemoryInfo]:
        "Return the memory info for the inputs."
        return self._input_meminfos

    def get_output_memory_infos(self) -> Sequence[C.OrtMemoryInfo]:
        "Return the memory info for the outputs."
        return self._output_meminfos

    def get_input_epdevices(self) -> Sequence[C.OrtEpDevice]:
        "Return the execution providers for the inputs."
        return self._input_epdevices

    def get_providers(self) -> Sequence[str]:
        "Return list of registered execution providers."
        return self._providers

    def get_provider_options(self):
        "Return registered execution providers' configurations."
        return self._provider_options

    def get_provider_graph_assignment_info(self) -> Sequence[C.OrtEpAssignedSubgraph]:
        """
        Get information about the subgraphs assigned to each execution provider and the nodes within.

        Application must enable the recording of graph assignment information by setting the session configuration
        for the key "session.record_ep_graph_assignment_info" to "1".
        """
        return self._sess.get_provider_graph_assignment_info()

    def set_providers(self, providers=None, provider_options=None) -> None:
        """
        Register the input list of execution providers. The underlying session is re-created.

        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example
        `['CUDAExecutionProvider', 'CPUExecutionProvider']`
        means execute a node using CUDAExecutionProvider if capable,
        otherwise execute using CPUExecutionProvider.
        """
        # recreate the underlying C.InferenceSession
        self._reset_session(providers, provider_options)

    def disable_fallback(self) -> None:
        """
        Disable session.run() fallback mechanism.
        """
        self._enable_fallback = False

    def enable_fallback(self) -> None:
        """
        Enable session.Run() fallback mechanism. If session.Run() fails due to an internal Execution Provider failure,
        reset the Execution Providers enabled for this session.
        If GPU is enabled, fall back to CUDAExecutionProvider.
        otherwise fall back to CPUExecutionProvider.
        """
        self._enable_fallback = True

    def _validate_input(self, feed_input_names):
        missing_input_names = []
        for input in self._inputs_meta:
            if input.name not in feed_input_names and not input.type.startswith("optional"):
                missing_input_names.append(input.name)
        if missing_input_names:
            raise ValueError(
                f"Required inputs ({missing_input_names}) are missing from input feed ({feed_input_names})."
            )

    def _validate_ortvalue_ownership(self, values):
        for value in values:
            if not isinstance(value, OrtValue):
                continue
            _validate_ortvalue_session_compatibility(value, self._sess, "used with")

    def _validate_graph_capture_run_api(self, run_options=None):
        if not self._sess.is_webgpu_graph_capture_enabled():
            return
        if _graph_annotation_id(run_options) == _GRAPH_ANNOTATION_SKIP:
            # gpu_graph_id=-1 skips capture, so transient feeds remain valid.
            return
        raise ValueError(
            "WebGPU graph capture requires fixed device OrtValues and run_with_iobinding. "
            f"Set the '{_GPU_GRAPH_ID_RUN_CONFIG_KEY}' run option to "
            f"{_GRAPH_ANNOTATION_SKIP} to opt a single run out of capture."
        )

    def run(self, output_names, input_feed, run_options=None) -> Sequence[np.ndarray | SparseTensor | list | dict]:
        """
        Compute the predictions.

        WebGPU graph capture requires fixed device OrtValues bound with
        :meth:`run_with_iobinding`; do not use this convenience API for captured replay.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return: list of results, every result is either a numpy array,
            a sparse tensor, a list or a dictionary.

        ::

            sess.run([output_name], {input_name: x})
        """
        self._validate_graph_capture_run_api(run_options)
        self._validate_input(list(input_feed.keys()))
        self._validate_ortvalue_ownership(input_feed.values())
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return self._sess.run(output_names, input_feed, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print(f"EP Error: {err!s} using {self._providers}")
                print(f"Falling back to {self._fallback_providers} and retrying.")
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                self._validate_ortvalue_ownership(input_feed.values())
                return self._sess.run(output_names, input_feed, run_options)
            raise

    def run_async(self, output_names, input_feed, callback, user_data, run_options=None):
        """
        Compute the predictions asynchronously in a separate cxx thread from ort intra-op threadpool.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param callback: python function that accept array of results, and a status string on error.
            The callback will be invoked by a cxx thread from ort intra-op threadpool.
        :param run_options: See :class:`onnxruntime.RunOptions`.

        ::
            class MyData:
                def __init__(self):
                    # ...
                def save_results(self, results):
                    # ...

            def callback(results: np.ndarray, user_data: MyData, err: str) -> None:
              if err:
                 print (err)
              else:
                # save results to user_data

            sess.run_async([output_name], {input_name: x}, callback)
        """
        self._validate_graph_capture_run_api(run_options)
        self._validate_input(list(input_feed.keys()))
        self._validate_ortvalue_ownership(input_feed.values())
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        return self._sess.run_async(output_names, input_feed, callback, user_data, run_options)

    def run_with_ort_values(self, output_names, input_dict_ort_values, run_options=None) -> Sequence[OrtValue]:
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_dict_ort_values: dictionary ``{ input_name: input_ort_value }``
            See ``OrtValue`` class how to create `OrtValue`
            from numpy array or `SparseTensor`
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return: an array of `OrtValue`

        ::

            sess.run([output_name], {input_name: x})
        """

        def invoke(sess, output_names, input_dict_ort_values, run_options):
            input_dict = {}
            for n, v in input_dict_ort_values.items():
                input_dict[n] = v._get_c_value()
            result = sess.run_with_ort_values(input_dict, output_names, run_options)
            if not isinstance(result, C.OrtValueVector):
                raise TypeError("run_with_ort_values() must return a instance of type 'OrtValueVector'.")
            ort_values = [OrtValue(v) for v in result]
            return ort_values

        self._validate_graph_capture_run_api(run_options)
        self._validate_input(list(input_dict_ort_values.keys()))
        self._validate_ortvalue_ownership(input_dict_ort_values.values())
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return invoke(self._sess, output_names, input_dict_ort_values, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print(f"EP Error: {err!s} using {self._providers}")
                print(f"Falling back to {self._fallback_providers} and retrying.")
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                self._validate_ortvalue_ownership(input_dict_ort_values.values())
                return invoke(self._sess, output_names, input_dict_ort_values, run_options)
            raise

    def end_profiling(self):
        """
        End profiling and return results in a file.

        The results are stored in a filename if the option
        :meth:`onnxruntime.SessionOptions.enable_profiling`.
        """
        return self._sess.end_profiling()

    def get_profiling_start_time_ns(self):
        """
        Return the nanoseconds of profiling's start time
        Comparable to time.monotonic_ns() after Python 3.3
        On some platforms, this timer may not be as precise as nanoseconds
        For instance, on Windows and MacOS, the precision will be ~100ns
        """
        return self._sess.get_profiling_start_time_ns

    def io_binding(self) -> IOBinding:
        "Return an onnxruntime.IOBinding object`."
        return IOBinding(self)

    def create_ortvalue_from_shape_and_type(
        self,
        shape: Sequence[int],
        element_type,
        device_type: str,
        device_id: int = 0,
        vendor_id: int | OrtDeviceVendorId = -1,
    ) -> OrtValue:
        """Create an OrtValue using this session's allocator.

        The value retains this session and may only be updated or bound through it.
        A matching allocator and a numeric tensor type are required.
        """
        device = OrtDevice.make(device_type, device_id, vendor_id)._get_c_device()
        if isinstance(element_type, int):
            ortvalue = self._sess.create_ortvalue_from_shape_and_onnx_type(shape, element_type, device)
        else:
            ortvalue = self._sess.create_ortvalue_from_shape_and_type(shape, element_type, device)
        return OrtValue(ortvalue, session=self._sess)

    def release_captured_graph(self, graph_annotation_id: int = 0) -> None:
        """Release a captured graph and unpin its IOBinding.

        ``graph_annotation_id`` matches the capture's ``gpu_graph_id`` and defaults to zero.
        EPs without captured-graph release support treat this as a no-op.
        """
        self._sess.release_captured_graph(graph_annotation_id)
        pinned = self._captured_graph_bindings.pop(graph_annotation_id, None)
        if pinned is not None:
            iobinding = pinned[0]()
            if iobinding is not None:
                iobinding._pinned_graph_ids.discard(graph_annotation_id)

    def run_with_iobinding(self, iobinding, run_options=None):
        """
        Compute the predictions.

        :param iobinding: the iobinding object that has graph inputs/outputs bind.
        :param run_options: See :class:`onnxruntime.RunOptions`.

        WebGPU capture requires static shapes, disabled memory patterns, fixed WebGPU OrtValues
        (session-owned or shared), and no CPU compute nodes other than shape-only nodes. Each
        ``gpu_graph_id`` pins one IOBinding and its buffers until :meth:`release_captured_graph`.
        Update inputs in place and use :meth:`IOBinding.copy_outputs_to_cpu` for readback;
        ``gpu_graph_id=-1`` disables capture.
        """
        if iobinding._session is not self._sess:
            raise ValueError("IOBinding must be used with the session that created it.")

        graph_annotation_id = _graph_annotation_id(run_options)
        capturing = self._sess.is_webgpu_graph_capture_enabled() and graph_annotation_id != _GRAPH_ANNOTATION_SKIP
        if capturing:
            iobinding._validate_capture_bindings()
            signature = iobinding._capture_signature()
            pinned = self._captured_graph_bindings.get(graph_annotation_id)
            if pinned is not None:
                pinned_iobinding, pinned_signature = pinned[0](), pinned[1]
                if pinned_iobinding is not iobinding:
                    raise ValueError(
                        f"WebGPU graph {graph_annotation_id} was captured with a different "
                        "IOBinding. Replay re-issues the buffers recorded at capture and would "
                        "silently ignore this binding. Reuse the original IOBinding, or call "
                        f"release_captured_graph({graph_annotation_id}) first."
                    )
                if pinned_signature != signature:
                    raise ValueError(
                        f"WebGPU graph {graph_annotation_id} was captured with different I/O "
                        "buffers. Replay writes to the buffers recorded at capture, so the "
                        "rebound values would never be read or written. Update the original "
                        "buffers in place, or call "
                        f"release_captured_graph({graph_annotation_id}) first."
                    )

        self._sess.run_with_iobinding(iobinding._iobinding, run_options)

        if capturing and graph_annotation_id not in self._captured_graph_bindings:
            self._captured_graph_bindings[graph_annotation_id] = (weakref.ref(iobinding), signature)
            iobinding._pinned_graph_ids.add(graph_annotation_id)

    def set_ep_dynamic_options(self, options: dict[str, str]):
        """
        Set dynamic options for execution providers.

        :param options: Dictionary of key-value pairs where both keys and values are strings.
                        These options will be passed to the execution providers to modify
                        their runtime behavior.
        """
        self._sess.set_ep_dynamic_options(options)

    def get_tuning_results(self):
        return self._sess.get_tuning_results()

    def set_tuning_results(self, results, *, error_on_invalid=False):
        return self._sess.set_tuning_results(results, error_on_invalid)

    def run_with_ortvaluevector(self, run_options, feed_names, feeds, fetch_names, fetches, fetch_devices):
        """
        Compute the predictions similar to other run_*() methods but with minimal C++/Python conversion overhead.

        :param run_options: See :class:`onnxruntime.RunOptions`.
        :param feed_names: list of input names.
        :param feeds: list of input OrtValue.
        :param fetch_names: list of output names.
        :param fetches: list of output OrtValue.
        :param fetch_devices: list of output devices.
        """
        # Same capture check as run(), run_with_ort_values() and run_async(): replay re-issues the
        # buffers recorded at capture, so transient vectors would be silently ignored. Nothing about
        # a raw vector is unsafe outside capture, including on a WebGPU session.
        self._validate_graph_capture_run_api(run_options)
        self._sess.run_with_ortvaluevector(run_options, feed_names, feeds, fetch_names, fetches, fetch_devices)


class InferenceSession(Session):
    """
    This is the main class used to run a model.
    """

    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
        sess_options: C.SessionOptions | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **kwargs,
    ) -> None:
        """
        :param path_or_bytes: Filename or serialized ONNX or ORT format model in a byte string.
        :param sess_options: Session options.
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:

        ::

            so = onnxruntime.SessionOptions()
            # so.add_session_config_entry('session.load_model_format', 'ONNX') or
            so.add_session_config_entry('session.load_model_format', 'ORT')

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example
        `['CUDAExecutionProvider', 'CPUExecutionProvider']`
        means execute a node using `CUDAExecutionProvider`
        if capable, otherwise execute using `CPUExecutionProvider`.
        """
        super().__init__(enable_fallback=int(kwargs.get("enable_fallback", 1)) == 1)

        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_path = os.fspath(path_or_bytes)
            self._model_bytes = None
        elif isinstance(path_or_bytes, bytes):
            self._model_path = None
            self._model_bytes = path_or_bytes  # TODO: This is bad as we're holding the memory indefinitely
        else:
            raise TypeError(f"Unable to load from type '{type(path_or_bytes)}'")

        self._sess_options = sess_options
        self._sess_options_initial = sess_options
        if "read_config_from_model" in kwargs:
            self._read_config_from_model = int(kwargs["read_config_from_model"]) == 1
        else:
            self._read_config_from_model = os.environ.get("ORT_LOAD_CONFIG_FROM_MODEL") == "1"

        # internal parameters that we don't expect to be used in general so aren't documented
        disabled_optimizers = kwargs.get("disabled_optimizers")

        try:
            self._create_inference_session(providers, provider_options, disabled_optimizers)
        except (ValueError, RuntimeError) as e:
            if self._enable_fallback:
                try:
                    print("*************** EP Error ***************")
                    print(f"EP Error {e} when using {providers}")
                    print(f"Falling back to {self._fallback_providers} and retrying.")
                    print("****************************************")
                    self._create_inference_session(self._fallback_providers, None)
                    # Fallback only once.
                    self.disable_fallback()
                    return
                except Exception as fallback_error:
                    raise fallback_error from e
            # Fallback is disabled. Raise the original error.
            raise e

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Validate that TensorrtExecutionProvider and NvTensorRTRTXExecutionProvider are not both specified
        if providers:
            has_tensorrt = any(
                provider == "TensorrtExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                for provider in providers
            )
            has_tensorrt_rtx = any(
                provider == "NvTensorRTRTXExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "NvTensorRTRTXExecutionProvider")
                for provider in providers
            )
            if has_tensorrt and has_tensorrt_rtx:
                raise ValueError(
                    "Cannot enable both 'TensorrtExecutionProvider' and 'NvTensorRTRTXExecutionProvider' "
                    "in the same session."
                )
        # Tensorrt and TensorRT RTX can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "NvTensorRTRTXExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "NvTensorRTRTXExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "NvTensorRTRTXExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        elif "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        # Print a warning if user passed providers to InferenceSession() but the SessionOptions instance
        # already has provider information (e.g., via add_provider_for_devices()). The providers specified
        # here will take precedence.
        if self._sess_options is not None and (providers or provider_options) and self._sess_options.has_providers():
            warnings.warn(
                "Specified 'providers'/'provider_options' when creating InferenceSession but SessionOptions has "
                "already been configured with providers. InferenceSession will only use the providers "
                "passed to InferenceSession()."
            )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
        sess.initialize_session(providers, provider_options, disabled_optimizers)

        self._sess = sess
        self._sess_options = self._sess.session_options
        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta
        self._overridable_initializers = self._sess.overridable_initializers
        self._input_meminfos = self._sess.input_meminfos
        self._output_meminfos = self._sess.output_meminfos
        self._input_epdevices = self._sess.input_epdevices
        self._model_meta = self._sess.model_meta
        self._providers = self._sess.get_providers()
        self._provider_options = self._sess.get_provider_options()
        self._profiling_start_time_ns = self._sess.get_profiling_start_time_ns

    def _release_captured_graphs(self) -> None:
        """Release every graph captured by the current session handle and unpin its IOBinding.

        A captured graph belongs to the ``C.InferenceSession`` that captured it, so it must be
        released before that handle is replaced. Otherwise the stale bookkeeping would reject an
        IOBinding created by the replacement session and a later ``release_captured_graph`` would
        act on the replacement session while unpinning the old binding.
        """
        for graph_annotation_id in sorted(self._captured_graph_bindings):
            self.release_captured_graph(graph_annotation_id)

    def _reset_session(self, providers, provider_options) -> None:
        "release underlying session object."
        # Captured graphs outlive neither the session handle nor its bookkeeping. Release them
        # first so a failure here leaves the session intact instead of half torn down.
        self._release_captured_graphs()

        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._sess_options = None
        self._inputs_meta = None
        self._outputs_meta = None
        self._overridable_initializers = None
        self._input_meminfos = None
        self._output_meminfos = None
        self._input_epdevices = None
        self._model_meta = None
        self._providers = None
        self._provider_options = None
        self._profiling_start_time_ns = None

        # create a new C.InferenceSession
        self._sess = None
        self._sess_options = self._sess_options_initial
        self._create_inference_session(providers, provider_options)

    def _register_ep_custom_ops(self, session_options, providers, provider_options, available_providers):
        for i in range(len(providers)):
            if providers[i] in available_providers and providers[i] == "TensorrtExecutionProvider":
                C.register_tensorrt_plugins_as_custom_ops(session_options, provider_options[i])
            elif (
                isinstance(providers[i], tuple)
                and providers[i][0] in available_providers
                and providers[i][0] == "TensorrtExecutionProvider"
            ):
                C.register_tensorrt_plugins_as_custom_ops(session_options, providers[i][1])

            if providers[i] in available_providers and providers[i] == "NvTensorRTRTXExecutionProvider":
                C.register_nv_tensorrt_rtx_plugins_as_custom_ops(session_options, provider_options[i])
            elif (
                isinstance(providers[i], tuple)
                and providers[i][0] in available_providers
                and providers[i][0] == "NvTensorrtRTXExecutionProvider"
            ):
                C.register_nv_tensorrt_rtx_plugins_as_custom_ops(session_options, providers[i][1])


def make_get_initializer_location_func_wrapper(
    get_initializer_location_func: GetInitializerLocationFunc,
) -> GetInitializerLocationWrapperFunc:
    """
    Wraps a user's "get initializer location" function. The returned wrapper function adheres to the
    signature expected by ORT.

    Need this wrapper to:
      - Convert the `initializer_value` parameter from `C.OrtValue` to `onnxruntime.OrtValue`, which is more
        convenient for the user's function to use.
      - Allow the user's function to return the original `external_info` parameter (this wrapper makes a copy)
    """

    def get_initializer_location_func_wrapper(
        initializer_name: str,
        initializer_value: C.OrtValue,
        external_info: C.OrtExternalInitializerInfo | None,
    ) -> C.OrtExternalInitializerInfo | None:
        ret_val: C.OrtExternalInitializerInfo | None = get_initializer_location_func(
            initializer_name, OrtValue(initializer_value), external_info
        )
        if ret_val is not None and ret_val == external_info:
            # User returned `external_info` (const and owned by ORT). ORT expects the returned value to be
            # a new instance (that it deletes), so make a copy.
            ret_val = C.OrtExternalInitializerInfo(ret_val.filepath, ret_val.file_offset, ret_val.byte_size)
        return ret_val

    return get_initializer_location_func_wrapper


class ModelCompiler:
    """
    This class is used to compile an ONNX model. A compiled ONNX model has EPContext nodes that each
    encapsulates a subgraph compiled/optimized for a specific execution provider.

    Refer to the EPContext design document for more information about EPContext models:
    https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html

        ::

            sess_options = onnxruntime.SessionOptions()
            sess_options.add_provider("SomeExecutionProvider", {"option1": "value1"})
            # Alternatively, allow ONNX Runtime to select the provider automatically given a policy:
            # sess_options.set_provider_selection_policy(onnxrt.OrtExecutionProviderDevicePolicy.PREFER_NPU)

            model_compiler = onnxruntime.ModelCompiler(sess_options, "input_model.onnx")
            model_compiler.compile_to_file("output_model.onnx")
    """

    def __init__(
        self,
        sess_options: C.SessionOptions,
        input_model_path_or_bytes: str | os.PathLike | bytes,
        embed_compiled_data_into_model: bool = False,
        external_initializers_file_path: str | os.PathLike | None = None,
        external_initializers_size_threshold: int = 1024,
        flags: int = C.OrtCompileApiFlags.NONE,
        graph_optimization_level: C.GraphOptimizationLevel = C.GraphOptimizationLevel.ORT_DISABLE_ALL,
        get_initializer_location_func: GetInitializerLocationFunc | None = None,
    ):
        """
        Creates a ModelCompiler instance.

        :param sess_options: Session options containing the providers for which the model will be compiled.
            Refer to SessionOptions.add_provider() and SessionOptions.set_provider_selection_policy().
        :param input_model_path_or_bytes: The path to the input model file or bytes representing a serialized
            ONNX model.
        :param embed_compiled_data_into_model: Defaults to False. Set to True to embed compiled binary data into
            EPContext nodes in the compiled model.
        :param external_initializers_file_path: Defaults to None. Set to a path for a file that will store the
            initializers for non-compiled nodes.
        :param external_initializers_size_threshold: Defaults to 1024. Ignored if `external_initializers_file_path`
            is None or empty. Initializers larger than this threshold are stored in the external initializers file.
        :param flags: Additional boolean options to enable. Set this parameter to a bitwise OR of
            flags in onnxruntime.OrtCompileApiFlags.
        :param graph_optimization_level: The graph optimization level.
            Defaults to onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL.
        :param get_initializer_location_func: Optional function called for every initializer to allow user to specify
            whether an initializer should be stored within the model or externally. Example:
            ```
                def get_initializer_location(
                    initializer_name: str,
                    initializer_value: onnxrt.OrtValue,
                    external_info: onnxrt.OrtExternalInitializerInfo | None,
                ) -> onnxrt.OrtExternalInitializerInfo | None:
                    byte_size = initializer_value.tensor_size_in_bytes()

                    if byte_size < 64:
                        return None  # Store small initializer within compiled model.

                    # Else, write initializer to new external file.
                    value_np = initializer_value.numpy()
                    file_offset = ext_init_file.tell()
                    ext_init_file.write(value_np.tobytes())
                    return onnxrt.OrtExternalInitializerInfo(initializer_file_path, file_offset, byte_size)
            ```
        """
        input_model_path: str | os.PathLike | None = None
        input_model_bytes: bytes | None = None
        if isinstance(input_model_path_or_bytes, (str, os.PathLike)):
            if not input_model_path_or_bytes:
                raise ValueError("Input model path is empty")
            input_model_path = os.fspath(input_model_path_or_bytes)
        elif isinstance(input_model_path_or_bytes, bytes):
            if len(input_model_path_or_bytes) == 0:
                raise ValueError("Input model bytes array is empty")
            input_model_bytes = input_model_path_or_bytes
        else:
            raise TypeError(f"Unable to load from type '{type(input_model_path_or_bytes)}'")

        if external_initializers_file_path:
            if not isinstance(external_initializers_file_path, (str, os.PathLike)):
                arg_type = type(external_initializers_file_path)
                raise TypeError(f"Output external initializer filepath is of unexpected type '{arg_type}'")
            external_initializers_file_path = os.fspath(external_initializers_file_path)
        else:
            external_initializers_file_path = ""

        if get_initializer_location_func is not None:
            if external_initializers_file_path:
                raise ValueError(
                    "Cannot initialize ModelCompiler with both `external_initializers_file_path` "
                    "and `get_initializer_location_func`"
                )
            self.get_initializer_location_func_wrapper = make_get_initializer_location_func_wrapper(
                get_initializer_location_func
            )
        else:
            self.get_initializer_location_func_wrapper = None

        if input_model_path:
            self._model_compiler = C.ModelCompiler(
                sess_options,
                input_model_path,
                True,  # is path
                embed_compiled_data_into_model,
                external_initializers_file_path,
                external_initializers_size_threshold,
                flags,
                graph_optimization_level,
                self.get_initializer_location_func_wrapper,
            )
        else:
            self._model_compiler = C.ModelCompiler(
                sess_options,
                input_model_bytes,
                False,  # is bytes
                embed_compiled_data_into_model,
                external_initializers_file_path,
                external_initializers_size_threshold,
                flags,
                graph_optimization_level,
                self.get_initializer_location_func_wrapper,
            )

    def compile_to_file(self, output_model_path: str | None = None):
        """
        Compiles to an output file. If an output file path is not provided,
        the output file path is generated based on the input model path by replacing
        '.onnx' with '_ctx.onnx'. Ex: The generated output file is 'model_ctx.onnx' for
        an input model with path 'model.onnx'.

        Raises an 'InvalidArgument' exception if the compilation options are invalid.

        :param output_model_path: Defaults to None. The path for the output/compiled model.
        """
        if output_model_path:
            if not isinstance(output_model_path, (str, os.PathLike)):
                raise TypeError(f"Output model's filepath is of unexpected type '{type(output_model_path)}'")
            output_model_path = os.fspath(output_model_path)
        self._model_compiler.compile_to_file(output_model_path)

    def set_ep_context_data_write_func(self, write_func: Callable[[str, C.OrtEpContextData], None]) -> None:
        """
        Registers a callback that receives external EPContext data during compilation.

        The ``OrtEpContextData`` view is valid only during the callback. Read large payloads in chunks with
        ``data.read(offset, length)``. ORT may invoke the callback concurrently, so synchronize shared state.

        :param write_func: Callback receiving the logical data name and a bounded data view.
        """
        self._model_compiler.set_ep_context_data_write_func(write_func)

    def clear_ep_context_data_write_func(self) -> None:
        """Clears a previously registered external EPContext data write callback."""
        self._model_compiler.clear_ep_context_data_write_func()

    def compile_to_bytes(self) -> bytes:
        """
        Compiles to bytes representing the serialized compiled ONNX model.

        Raises an 'InvalidArgument' exception if the compilation options are invalid.

        :return: A bytes object representing the compiled ONNX model.
        """
        return self._model_compiler.compile_to_bytes()

    def compile_to_stream(self, write_function: Callable[[bytes], None]):
        """
        Compiles the input model and writes the serialized ONNX bytes to a stream using the provided write function.
        Raises an 'InvalidArgument' exception if the compilation options are invalid.
        :param write_function: A callable that accepts a bytes buffer to write.
        """
        self._model_compiler.compile_to_stream(write_function)


class IOBinding:
    """
    This class provides API to bind input/output to a specified device, e.g. GPU.
    """

    def __init__(self, session: Session):
        self._session = session._sess
        self._is_webgpu_session = "WebGpuExecutionProvider" in session.get_providers()
        self._is_webgpu_graph_capture_enabled = self._session.is_webgpu_graph_capture_enabled()
        self._iobinding = C.SessionIOBinding(session._sess)
        self._numpy_obj_references = {}
        # Capture tracks fixed OrtValues; None marks host or raw-pointer bindings.
        self._bound_inputs: dict[str, OrtValue | None] = {}
        self._bound_outputs: dict[str, OrtValue | None] = {}
        # Captured graph IDs freeze these bindings until release.
        self._pinned_graph_ids: set[int] = set()

    def _reject_if_pinned(self, action: str) -> None:
        if self._pinned_graph_ids:
            pinned = ", ".join(str(graph_id) for graph_id in sorted(self._pinned_graph_ids))
            raise ValueError(
                f"Cannot {action} while captured WebGPU graph(s) [{pinned}] still reference this "
                "IOBinding. Replay re-issues the buffers recorded at capture, so a change here "
                "would not affect replay and releasing the buffers could invalidate it. Call "
                "release_captured_graph(id) first."
            )

    def _capture_signature(self):
        """Return bound OrtValues whose identity and lifetime must remain fixed during replay."""
        return (
            tuple(sorted(self._bound_inputs.items(), key=lambda item: item[0])),
            tuple(sorted(self._bound_outputs.items(), key=lambda item: item[0])),
        )

    def _validate_capture_bindings(self) -> None:
        """Every binding participating in capture must be a fixed WebGPU device OrtValue."""
        for kind, bound in (("input", self._bound_inputs), ("output", self._bound_outputs)):
            for name, value in bound.items():
                if value is None or not value._is_webgpu_buffer:
                    raise ValueError(
                        f"WebGPU graph capture requires fixed WebGPU device OrtValues; {kind} "
                        f"'{name}' is not one. Bind values created by this session or backed by an "
                        f"environment-registered shared allocator, or set the "
                        f"'{_GPU_GRAPH_ID_RUN_CONFIG_KEY}' run option to {_GRAPH_ANNOTATION_SKIP} "
                        "to run without capture."
                    )

    def bind_cpu_input(self, name, arr_on_cpu):
        """
        bind an input to array on CPU
        :param name: input name
        :param arr_on_cpu: input values as a python array on CPU
        """
        self._reject_if_pinned("rebind inputs")

        # Hold a reference to the numpy object as the bound OrtValue is backed
        # directly by the data buffer of the numpy object and so the numpy object
        # must be around until this IOBinding instance is around
        self._numpy_obj_references[name] = arr_on_cpu
        self._iobinding.bind_input(name, arr_on_cpu)
        self._bound_inputs[name] = None

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        """
        :param name: input name
        :param device_type: e.g. cpu, cuda, cann
        :param device_id: device id, e.g. 0
        :param element_type: input element type. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16)
        :param shape: input shape
        :param buffer_ptr: memory pointer to input data
        """
        self._reject_if_pinned("rebind inputs")
        self._iobinding.bind_input(
            name,
            OrtDevice.make(device_type, device_id)._get_c_device(),
            element_type,
            shape,
            buffer_ptr,
        )
        self._bound_inputs[name] = None

    def bind_ortvalue_input(self, name, ortvalue):
        """
        :param name: input name
        :param ortvalue: OrtValue instance to bind
        """
        _validate_ortvalue_session_compatibility(ortvalue, self._session, "bound to")
        self._reject_if_pinned("rebind inputs")
        self._iobinding.bind_ortvalue_input(name, ortvalue._ortvalue)
        self._bound_inputs[name] = ortvalue

    def synchronize_inputs(self):
        self._iobinding.synchronize_inputs()

    def bind_output(
        self,
        name,
        device_type="cpu",
        device_id=0,
        element_type=None,
        shape=None,
        buffer_ptr=None,
    ):
        """
        :param name: output name
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        :param element_type: output element type. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16)
        :param shape: output shape
        :param buffer_ptr: memory pointer to output data
        """

        self._reject_if_pinned("rebind outputs")

        # Follow the `if` path when the user has not provided any pre-allocated buffer but still
        # would like to bind an output to a specific device (e.g. cuda).
        # Pre-allocating an output buffer may not be an option for the user as :
        # (1) They may not want to use a custom allocator specific to the device they want to bind the output to,
        # in which case ORT will allocate the memory for the user
        # (2) The output has a dynamic shape and hence the size of the buffer may not be fixed across runs
        if buffer_ptr is None:
            self._iobinding.bind_output(
                name,
                OrtDevice.make(device_type, device_id)._get_c_device(),
            )
        else:
            if element_type is None or shape is None:
                raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")
            self._iobinding.bind_output(
                name,
                OrtDevice.make(device_type, device_id)._get_c_device(),
                element_type,
                shape,
                buffer_ptr,
            )
        self._bound_outputs[name] = None

    def bind_ortvalue_output(self, name, ortvalue):
        """
        :param name: output name
        :param ortvalue: OrtValue instance to bind
        """
        _validate_ortvalue_session_compatibility(ortvalue, self._session, "bound to")
        self._reject_if_pinned("rebind outputs")
        self._iobinding.bind_ortvalue_output(name, ortvalue._ortvalue)
        self._bound_outputs[name] = ortvalue

    def synchronize_outputs(self):
        self._iobinding.synchronize_outputs()

    def get_outputs(self):
        """
        Returns the output OrtValues from the Run() that preceded the call.
        The data buffer of the obtained OrtValues may not reside on CPU memory
        """
        outputs = self._iobinding.get_outputs()
        if not isinstance(outputs, C.OrtValueVector):
            raise TypeError("get_outputs() must return an instance of type 'OrtValueVector'.")
        result = []
        for index in range(len(outputs)):
            ortvalue = outputs[index]
            result.append(
                OrtValue(
                    ortvalue,
                    session=self._session if ortvalue._is_webgpu_buffer() else None,
                )
            )
        return result

    def get_outputs_as_ortvaluevector(self):
        """Return the raw OrtValueVector of outputs from the Run() that preceded the call.

        The vector is a reference into this IOBinding (pybind ``reference_internal``), so it keeps the
        IOBinding alive, which in turn keeps the session alive. Device-resident outputs are therefore
        safe to hold past the session going out of scope.
        """
        return self._iobinding.get_outputs()

    def copy_outputs_to_cpu(self):
        """Copy output contents to CPU."""
        return self._iobinding.copy_outputs_to_cpu()

    def clear_binding_inputs(self):
        self._reject_if_pinned("clear input bindings")
        self._iobinding.clear_binding_inputs()
        self._bound_inputs.clear()

    def clear_binding_outputs(self):
        self._reject_if_pinned("clear output bindings")
        self._iobinding.clear_binding_outputs()
        self._bound_outputs.clear()


class OrtValue:
    """
    A data structure that supports all ONNX data formats (tensors and non-tensors) that allows users
    to place the data backing these on a device, for example, on a CUDA supported device.
    This class provides APIs to construct and deal with OrtValues.
    """

    def __init__(
        self,
        ortvalue: C.OrtValue,
        numpy_obj: np.ndarray | None = None,
        session: C.InferenceSession | None = None,
    ):
        if isinstance(ortvalue, C.OrtValue):
            self._ortvalue = ortvalue
            # Hold a ref count to the numpy object if the OrtValue is backed directly
            # by its data buffer so that it isn't destroyed when the OrtValue is in use
            self._numpy_obj = numpy_obj
            # Session-scoped device allocators can be invalidated when their session is destroyed.
            self._session = session
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided ortvalue` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.OrtValue`"
            )

    @property
    def _is_webgpu_buffer(self) -> bool:
        """Whether this value is backed by a WebGPU buffer.

        Resolved from the native OrtValue on every access and deliberately read-only: graph-capture
        validation and the copy-path checks below refuse host memory, so a settable attribute would
        let a CPU tensor pass itself off as a device tensor.
        """
        return self._ortvalue._is_webgpu_buffer()

    def _get_c_value(self) -> C.OrtValue:
        return self._ortvalue

    @classmethod
    def ortvalue_from_numpy(
        cls, numpy_obj: np.ndarray, /, device_type="cpu", device_id=0, vendor_id: int | OrtDeviceVendorId = -1
    ) -> OrtValue:
        """
        Factory method to construct an OrtValue (which holds a Tensor) from a given Numpy object
        A copy of the data in the Numpy object is held by the OrtValue only if the device is NOT cpu

        :param numpy_obj: The Numpy object to construct the OrtValue from
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        :param vendor_id: The device's PCI vendor id as an int or OrtDeviceVendorId. If provided, the device_type should be "gpu" or "npu".
        """
        # Hold a reference to the numpy object (if device_type is 'cpu') as the OrtValue
        # is backed directly by the data buffer of the numpy object and so the numpy object
        # must be around until this OrtValue instance is around
        return cls(
            C.OrtValue.ortvalue_from_numpy(
                numpy_obj,
                OrtDevice.make(device_type, device_id, vendor_id)._get_c_device(),
            ),
            numpy_obj if device_type.lower() == "cpu" else None,
        )

    @classmethod
    def ortvalue_from_numpy_with_onnx_type(cls, data: np.ndarray, /, onnx_element_type: int) -> OrtValue:
        """
        This method creates an instance of OrtValue on top of the numpy array.
        No data copy is made and the lifespan of the resulting OrtValue should never
        exceed the lifespan of bytes object. The API attempts to reinterpret
        the data type which is expected to be the same size. This is useful
        when we want to use an ONNX data type that is not supported by numpy.

        :param data: numpy.ndarray.
        :param onnx_element_type: a valid onnx TensorProto::DataType enum value
        """
        return cls(C.OrtValue.ortvalue_from_numpy_with_onnx_type(data, onnx_element_type), data)

    @classmethod
    def ortvalue_from_shape_and_type(
        cls,
        shape: Sequence[int],
        element_type,
        device_type: str = "cpu",
        device_id: int = 0,
        vendor_id: int | OrtDeviceVendorId = -1,
        memory_info: C.OrtMemoryInfo | None = None,
    ) -> OrtValue:
        """
        Factory method to construct an OrtValue (which holds a Tensor) from given shape and element_type

        :param shape: List of integers indicating the shape of the OrtValue
        :param element_type: The data type of the elements. It can be either numpy type (like numpy.float32) or an integer for onnx type (like onnx.TensorProto.BFLOAT16).
        :param device_type: e.g. cpu, cuda, cann, cpu by default
        :param device_id: device id, e.g. 0
        :param vendor_id: The device's PCI vendor id as an int or OrtDeviceVendorId. If provided, the device type should be "gpu" or "npu".
        :param memory_info: An OrtMemoryInfo from an OrtEpDevice (e.g. via ep_device.memory_info(OrtDeviceMemoryType.HOST_ACCESSIBLE)). When provided, the allocator matching this memory info is used directly, which allows allocating HOST_ACCESSIBLE memory for zero-copy numpy interop. The device_type, device_id, and vendor_id parameters are ignored when memory_info is provided.
        """

        if memory_info is not None:
            if device_type != "cpu" or device_id != 0 or vendor_id != -1:
                warnings.warn(
                    "device_type, device_id, and vendor_id are ignored when memory_info is provided.",
                    stacklevel=2,
                )
            if isinstance(element_type, int):
                return cls(
                    C.OrtValue.ortvalue_from_shape_and_onnx_type_for_memory_info(
                        shape,
                        element_type,
                        memory_info,
                    )
                )
            return cls(
                C.OrtValue.ortvalue_from_shape_and_type_for_memory_info(
                    shape,
                    element_type,
                    memory_info,
                )
            )

        device = OrtDevice.make(device_type, device_id, vendor_id)._get_c_device()

        # Integer for onnx element type (see https://onnx.ai/onnx/api/mapping.html).
        # For bfloat16, callers can also pass ml_dtypes.bfloat16 as a numpy-style
        # element_type; this integer path remains available for uses that don't
        # want to depend on ml_dtypes at the Python level.
        if isinstance(element_type, int):
            return cls(
                C.OrtValue.ortvalue_from_shape_and_onnx_type(
                    shape,
                    element_type,
                    device,
                )
            )

        return cls(
            C.OrtValue.ortvalue_from_shape_and_type(
                shape,
                element_type,
                device,
            )
        )

    @classmethod
    def ort_value_from_sparse_tensor(cls, sparse_tensor: SparseTensor) -> OrtValue:
        """
        The function will construct an OrtValue instance from a valid SparseTensor
        The new instance of OrtValue will assume the ownership of sparse_tensor
        """
        return cls(C.OrtValue.ort_value_from_sparse_tensor(sparse_tensor._get_c_tensor()))

    def as_sparse_tensor(self) -> SparseTensor:
        """
        The function will return SparseTensor contained in this OrtValue
        """
        return SparseTensor(self._ortvalue.as_sparse_tensor())

    def data_ptr(self) -> int:
        """
        Returns the address of the first element in the OrtValue's data buffer

        WebGPU buffers are opaque handles and do not expose a data pointer.
        """
        return self._ortvalue.data_ptr()

    def device_name(self) -> str:
        """
        Returns the name of the device where the OrtValue's data buffer resides e.g. cpu, cuda, cann, webgpu
        """
        return self._ortvalue.device_name().lower()

    def shape(self) -> Sequence[int]:
        """
        Returns the shape of the data in the OrtValue
        """
        return self._ortvalue.shape()

    def data_type(self) -> str:
        """
        Returns the data type of the data in the OrtValue. E.g. 'tensor(int64)'
        """
        return self._ortvalue.data_type()

    def element_type(self) -> int:
        """
        Returns the proto type of the data in the OrtValue
        if the OrtValue is a tensor.
        """
        return self._ortvalue.element_type()

    def tensor_size_in_bytes(self) -> int:
        """
        Returns the size of the data in the OrtValue in bytes
        if the OrtValue is a tensor.
        """
        return self._ortvalue.tensor_size_in_bytes()

    def has_value(self) -> bool:
        """
        Returns True if the OrtValue corresponding to an
        optional type contains data, else returns False
        """
        return self._ortvalue.has_value()

    def is_tensor(self) -> bool:
        """
        Returns True if the OrtValue contains a Tensor, else returns False
        """
        return self._ortvalue.is_tensor()

    def is_sparse_tensor(self) -> bool:
        """
        Returns True if the OrtValue contains a SparseTensor, else returns False
        """
        return self._ortvalue.is_sparse_tensor()

    def is_tensor_sequence(self) -> bool:
        """
        Returns True if the OrtValue contains a Tensor Sequence, else returns False
        """
        return self._ortvalue.is_tensor_sequence()

    def numpy(self) -> np.ndarray:
        """
        Returns a Numpy object from the OrtValue.
        Valid only for OrtValues holding Tensors. Throws for OrtValues holding non-Tensors.
        Use accessors to gain a reference to non-Tensor objects such as SparseTensor
        WebGPU device values require explicit readback with
        :meth:`IOBinding.copy_outputs_to_cpu`.
        """
        return self._ortvalue.numpy()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        Supports ``numpy.asarray(ortvalue)`` and ``numpy.array(ortvalue)`` via the
        `numpy __array__ protocol <https://numpy.org/devdocs/user/basics.interoperability.html>`_.

        Valid only for OrtValues holding Tensors on CPU.

        :param dtype: Optional numpy dtype to cast the result to.
        :param copy: Optional bool (numpy >= 2.0). If ``False``, a copy will
            only be made if necessary. If ``True``, a copy is always forced.
            If ``None`` (default), a copy will be made only if needed.
        :return: A numpy array with the same data as the OrtValue.
        """
        arr = self.numpy()

        if copy is not None:
            # numpy >= 2.0 added the copy kwarg to np.asarray;
            # np.array has always accepted it but with weaker semantics pre-2.0.
            arr = np.array(arr, dtype=dtype, copy=copy)
        elif dtype is not None:
            # np.asarray avoids a copy when the dtype already matches,
            # preserving memory sharing with the underlying OrtValue.
            arr = np.asarray(arr, dtype=dtype)

        return arr

    def __dlpack__(self, *, stream=None):
        """
        Returns a DLPack capsule representing the tensor (part of the
        `DLPack protocol <https://dmlc.github.io/dlpack/latest/>`_).

        This enables interoperability with other frameworks via
        ``from_dlpack(ortvalue)`` (e.g. ``torch.from_dlpack``,
        ``jax.dlpack.from_dlpack``, ``numpy.from_dlpack``).

        The OrtValue must hold a contiguous tensor. No data is copied;
        the consumer shares memory with this OrtValue, which must remain
        alive while the capsule is in use.
        WebGPU OrtValues cannot be exported through DLPack because their data
        is stored in opaque WebGPU buffers rather than CUDA-addressable memory.

        :param stream: Optional stream on which the tensor data is accessible.
            Currently unused; included for protocol compliance.
        :return: A PyCapsule holding a DLManagedTensor.
        """
        return self._ortvalue.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        """
        Returns ``(device_type, device_id)`` indicating where the tensor data
        resides (part of the `DLPack protocol
        <https://dmlc.github.io/dlpack/latest/>`_).

        WebGPU OrtValues do not expose a DLPack device.

        :return: Tuple of ``(device_type, device_id)`` as ints following DLPack
            ``DLDeviceType`` enum values.
        """
        return self._ortvalue.__dlpack_device__()

    @classmethod
    def from_dlpack(cls, data, /) -> OrtValue:
        """
        Construct an OrtValue from an object that implements the DLPack protocol.

        Accepts either:

        * An object with ``__dlpack__`` / ``__dlpack_device__`` methods
          (e.g. a PyTorch tensor, JAX array, or numpy array).
        * A raw DLPack PyCapsule (legacy path).

        Boolean tensors are automatically detected when the source object
        exposes a ``dtype`` attribute (numpy, PyTorch, etc.) or is an
        ``OrtValue``. For raw DLPack capsules where the original dtype cannot
        be inspected, bool tensors encoded as uint8 by older DLPack versions
        are not distinguishable from true uint8 tensors and will be imported
        as uint8.

        No data is copied; the new OrtValue shares memory with the source.

        :param data: A tensor object supporting the DLPack protocol, or a raw
            DLPack PyCapsule.
        :return: An OrtValue wrapping the tensor data.
        :raises TypeError: If the object does not provide a valid DLPack capsule.
        """
        # Detect boolean dtype from the source object before consuming it,
        # because DLPack encodes bool as uint8 and the capsule alone cannot
        # distinguish between the two.
        is_bool = False
        if isinstance(data, OrtValue):
            is_bool = data.data_type() == "tensor(bool)"
        elif hasattr(data, "dtype"):
            dtype_obj = data.dtype
            # Use .name when available (numpy, cupy, tensorflow all expose it).
            # Fall back to str() for frameworks that don't (e.g. PyTorch).
            dtype_name = getattr(dtype_obj, "name", str(dtype_obj))
            is_bool = dtype_name in ("bool", "bool_", "torch.bool")

        # If the input supports the __dlpack__ protocol, call it to get the capsule.
        if hasattr(data, "__dlpack__"):
            capsule = data.__dlpack__()
        else:
            capsule = data

        return cls(C.OrtValue.from_dlpack(capsule, is_bool))

    def update_inplace(self, data) -> None:
        """
        Update the OrtValue in place. The source data is copied over to the device
        memory backing the OrtValue. It can be used to update the input values for
        an InferenceSession with CUDA graph enabled or other scenarios where the
        OrtValue needs to be updated while the memory address can not be changed.

        :param data: The source data, which can be a Numpy array or another OrtValue.
            When an OrtValue is provided, data can be copied between devices (e.g.,
            GPU to GPU) without going through the CPU.
        """
        if isinstance(data, OrtValue):
            if self._is_webgpu_buffer != data._is_webgpu_buffer:
                raise ValueError(
                    "WebGPU OrtValue copies require WebGPU source and destination values; "
                    "use IOBinding.copy_outputs_to_cpu for readback."
                )
            if not _is_ortvalue_session_compatible(data, self._session):
                raise ValueError("Session-scoped OrtValues must originate from the same session.")

            self._ortvalue.update_inplace(data._ortvalue)
            return

        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray or an OrtValue.")

        # Every copy path requires contiguous source storage.
        if not data.flags.c_contiguous:
            data = np.ascontiguousarray(data)

        self._ortvalue.update_inplace(data)


_DEFAULT_WEBGPU_CONTEXT_ID = 0


def _session_webgpu_context_id(session: C.InferenceSession) -> int:
    """Return a session's WebGPU context id, or -1 when it has no WebGPU EP."""
    return session.webgpu_context_id()


def _has_foreign_webgpu_context(
    values: Sequence[OrtValue],
    context_id_getter=_session_webgpu_context_id,
) -> bool:
    """Whether any value provably belongs to a WebGPU context other than the default one.

    The environment data transfer resolves the default WebGPU context, so copying a buffer that
    belongs to a caller-supplied context would silently run on the wrong device. A value with no
    owning session cannot be attributed to a context and is treated as default-context, as is a
    session with no WebGPU EP.

    TODO: this can only attribute a value that carries session provenance, because every WebGPU
    allocation reports OrtDevice(GPU, VendorIds::NONE, 0) regardless of its context. Supporting a
    custom external WebGPU context requires the shared allocator and its data transfer to retain and
    use that external device instead of resolving WebGpuContextFactory::DefaultContext(), which
    would also let the transfer itself reject the copy instead of relying on this Python-side guard.
    """
    for value in values:
        if not value._is_webgpu_buffer or value._session is None:
            continue
        if context_id_getter(value._session) not in (-1, _DEFAULT_WEBGPU_CONTEXT_ID):
            return True
    return False


def copy_tensors(src: Sequence[OrtValue], dst: Sequence[OrtValue], stream=None) -> None:
    """
    Copy tensor data from source OrtValue sequence to destination OrtValue sequence.

    WebGPU values belonging to the default context are supported in either direction. A value from a
    caller-supplied WebGPU context is not, because the shared data transfer is bound to the default
    context.
    """
    if _has_foreign_webgpu_context([*src, *dst]):
        raise ValueError(
            "copy_tensors cannot copy a WebGPU OrtValue that belongs to a custom WebGPU context "
            "because the shared data transfer is bound to the default context; use IOBinding "
            "readback instead."
        )
    c_sources = [s._get_c_value() for s in src]
    c_dsts = [d._get_c_value() for d in dst]
    C.copy_tensors(c_sources, c_dsts, stream)


class OrtDevice:
    """
    A data structure that exposes the underlying C++ OrtDevice
    """

    def __init__(self, c_ort_device):
        """
        Internal constructor
        """
        if isinstance(c_ort_device, C.OrtDevice):
            self._ort_device = c_ort_device
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided object` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.OrtDevice`"
            )

    def _get_c_device(self):
        """
        Internal accessor to underlying object
        """
        return self._ort_device

    @staticmethod
    def make(ort_device_name, device_id, vendor_id: int | OrtDeviceVendorId = -1):
        if vendor_id < 0:
            # Preserve the historical convenience aliases ("cuda", "dml", "cann")
            # while making them work with plugin EP shared allocators. Those
            # allocators are keyed by vendor-specific OrtDevice values even when the
            # Python package itself was built without the corresponding built-in EP.
            alias_vendor_id = get_vendor_id_for_device_type(ort_device_name)
            if alias_vendor_id is not None:
                return OrtDevice(
                    C.OrtDevice(
                        get_ort_device_type(ort_device_name),
                        C.OrtDevice.default_memory(),
                        int(alias_vendor_id),
                        device_id,
                    )
                )

            # backwards compatibility with generic predefined OrtDevice names
            return OrtDevice(
                C.OrtDevice(
                    get_ort_device_type(ort_device_name),
                    C.OrtDevice.default_memory(),
                    device_id,
                )
            )
        else:
            # generic. use GPU or NPU for ort_device_name and provide a vendor id.
            # vendor id of 0 is valid in some cases (e.g. webgpu is generic and does not have a vendor id)
            return OrtDevice(
                C.OrtDevice(
                    get_ort_device_type(ort_device_name),
                    C.OrtDevice.default_memory(),
                    int(vendor_id),
                    device_id,
                )
            )

    def device_id(self):
        return self._ort_device.device_id()

    def device_type(self):
        return self._ort_device.device_type()

    def device_vendor_id(self):
        return self._ort_device.vendor_id()

    def device_mem_type(self):
        return self._ort_device.mem_type()


class SparseTensor:
    """
    A data structure that project the C++ SparseTensor object
    The class provides API to work with the object.
    Depending on the format, the class will hold more than one buffer
    depending on the format
    """

    def __init__(self, sparse_tensor: C.SparseTensor):
        """
        Internal constructor
        """
        if isinstance(sparse_tensor, C.SparseTensor):
            self._tensor = sparse_tensor
        else:
            # An end user won't hit this error
            raise ValueError(
                "`Provided object` needs to be of type `onnxruntime.capi.onnxruntime_pybind11_state.SparseTensor`"
            )

    def _get_c_tensor(self) -> C.SparseTensor:
        return self._tensor

    @classmethod
    def sparse_coo_from_numpy(
        cls,
        dense_shape: npt.NDArray[np.int64],
        values: np.ndarray,
        coo_indices: npt.NDArray[np.int64],
        ort_device: OrtDevice,
    ) -> SparseTensor:
        """
        Factory method to construct a SparseTensor in COO format from given arguments

        :param dense_shape: 1-D  numpy array(int64) or a python list that contains a dense_shape of the sparse tensor
            must be on cpu memory
        :param values: a homogeneous, contiguous 1-D numpy array that contains non-zero elements of the tensor
            of a type.
        :param coo_indices:  contiguous numpy array(int64) that contains COO indices for the tensor. coo_indices may
            have a 1-D shape when it contains a linear index of non-zero values and its length must be equal to
            that of the values. It can also be of 2-D shape, in which has it contains pairs of coordinates for
            each of the nnz values and its length must be exactly twice of the values length.
        :param ort_device: - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
            suppored for non-numeric data types.

        For primitive types, the method will map values and coo_indices arrays into native memory and will use
        them as backing storage. It will increment the reference count for numpy arrays and it will decrement it
        on GC. The buffers may reside in any storage either CPU or GPU.
        For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
        on other devices and their memory can not be mapped.
        """
        return cls(C.SparseTensor.sparse_coo_from_numpy(dense_shape, values, coo_indices, ort_device._get_c_device()))

    @classmethod
    def sparse_csr_from_numpy(
        cls,
        dense_shape: npt.NDArray[np.int64],
        values: np.ndarray,
        inner_indices: npt.NDArray[np.int64],
        outer_indices: npt.NDArray[np.int64],
        ort_device: OrtDevice,
    ) -> SparseTensor:
        """
        Factory method to construct a SparseTensor in CSR format from given arguments

        :param dense_shape: 1-D numpy array(int64) or a python list that contains a dense_shape of the
            sparse tensor (rows, cols) must be on cpu memory
        :param values: a  contiguous, homogeneous 1-D numpy array that contains non-zero elements of the tensor
            of a type.
        :param inner_indices:  contiguous 1-D numpy array(int64) that contains CSR inner indices for the tensor.
            Its length must be equal to that of the values.
        :param outer_indices:  contiguous 1-D numpy array(int64) that contains CSR outer indices for the tensor.
            Its length must be equal to the number of rows + 1.
        :param ort_device: - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
            suppored for non-numeric data types.

        For primitive types, the method will map values and indices arrays into native memory and will use them as
        backing storage. It will increment the reference count and it will decrement then count when it is GCed.
        The buffers may reside in any storage either CPU or GPU.
        For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
        on other devices and their memory can not be mapped.
        """
        return cls(
            C.SparseTensor.sparse_csr_from_numpy(
                dense_shape,
                values,
                inner_indices,
                outer_indices,
                ort_device._get_c_device(),
            )
        )

    def values(self) -> np.ndarray:
        """
        The method returns a numpy array that is backed by the native memory
        if the data type is numeric. Otherwise, the returned numpy array that contains
        copies of the strings.
        """
        return self._tensor.values()

    def as_coo_view(self):
        """
        The method will return coo representation of the sparse tensor which will enable
        querying COO indices. If the instance did not contain COO format, it would throw.
        You can query coo indices as:

        ::

            coo_indices = sparse_tensor.as_coo_view().indices()

        which will return a numpy array that is backed by the native memory.
        """
        return self._tensor.get_coo_data()

    def as_csrc_view(self):
        """
        The method will return CSR(C) representation of the sparse tensor which will enable
        querying CRS(C) indices. If the instance dit not contain CSR(C) format, it would throw.
        You can query indices as:

        ::

            inner_ndices = sparse_tensor.as_csrc_view().inner()
            outer_ndices = sparse_tensor.as_csrc_view().outer()

        returning numpy arrays backed by the native memory.
        """
        return self._tensor.get_csrc_data()

    def as_blocksparse_view(self):
        """
        The method will return coo representation of the sparse tensor which will enable
        querying BlockSparse indices. If the instance did not contain BlockSparse format, it would throw.
        You can query coo indices as:

        ::

            block_sparse_indices = sparse_tensor.as_blocksparse_view().indices()

        which will return a numpy array that is backed by the native memory
        """
        return self._tensor.get_blocksparse_data()

    def to_cuda(self, ort_device):
        """
        Returns a copy of this instance on the specified cuda device

        :param ort_device: with name 'cuda' and valid gpu device id

        The method will throw if:

        - this instance contains strings
        - this instance is already on GPU. Cross GPU copy is not supported
        - CUDA is not present in this build
        - if the specified device is not valid
        """
        return SparseTensor(self._tensor.to_cuda(ort_device._get_c_device()))

    def format(self):
        """
        Returns a OrtSparseFormat enumeration
        """
        return self._tensor.format

    def dense_shape(self) -> npt.NDArray[np.int64]:
        """
        Returns a numpy array(int64) containing a dense shape of a sparse tensor
        """
        return self._tensor.dense_shape()

    def data_type(self) -> str:
        """
        Returns a string data type of the data in the OrtValue
        """
        return self._tensor.data_type()

    def device_name(self) -> str:
        """
        Returns the name of the device where the SparseTensor data buffers reside e.g. cpu, cuda
        """
        return self._tensor.device_name().lower()


# Type hint for user-specified function that allows the user to specify initializer locations when compiling a model.
GetInitializerLocationFunc = Callable[
    [str, OrtValue, C.OrtExternalInitializerInfo | None], C.OrtExternalInitializerInfo | None
]

# Type hint that adheres to the signature expected by ORT.
GetInitializerLocationWrapperFunc = Callable[
    [str, C.OrtValue, C.OrtExternalInitializerInfo | None], C.OrtExternalInitializerInfo | None
]
