# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import collections
import collections.abc
import os
import warnings

from onnxruntime.capi import _pybind_state as C


def get_ort_device_type(device):
    device = device.lower()
    if device == 'cuda':
        return C.OrtDevice.cuda()
    elif device == 'cpu':
        return C.OrtDevice.cpu()
    else:
        raise Exception('Unsupported device type: ' + device)


def check_and_normalize_provider_args(providers, provider_options, available_provider_names):
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
            warnings.warn("Specified provider '{}' is not in available provider names."
                          "Available providers: '{}'".format(name, ", ".join(available_provider_names)))

        if name in provider_name_to_options:
            warnings.warn("Duplicate provider '{}' encountered, ignoring.".format(name))
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

        if not all([isinstance(provider, str) for provider in providers]):
            raise ValueError("Only string values for 'providers' are supported if 'provider_options' is given.")

        if not all([isinstance(options_for_provider, dict) for options_for_provider in provider_options]):
            raise ValueError("'provider_options' values must be dicts.")

        for name, options in zip(providers, provider_options):
            set_provider_options(name, options)

    else:
        for provider in providers:
            if isinstance(provider, str):
                set_provider_options(provider, dict())
            elif isinstance(provider, tuple) and len(provider) == 2 and \
                    isinstance(provider[0], str) and isinstance(provider[1], dict):
                set_provider_options(provider[0], provider[1])
            else:
                raise ValueError("'providers' values must be either strings or (string, dict) tuples.")

    return list(provider_name_to_options.keys()), list(provider_name_to_options.values())


def numpy_array_to_cuda(ort_device, numpy_array_on_cpu):
    '''
    The function allocates a numpy array on the specified CUDA device and copies
    the content of numpy_array_on_cpu. It preserves the dtype and the shape.
    Only numeric types are supported and only in CUDA enabled builds.
    No attempt is made to verify that the source is indeed on CPU
    '''
    return C.numpy_array_to_cuda(numpy_array_on_cpu, ort_device._get_c_device())


def numpy_array_to_cpu(numpy_array_on_cuda):
    '''
    The function allocates a numpy array in the host memory and copies
    the content of numpy_array_on_cuda. It preserves the dtype and the shape.
    Only numeric types are supported and only in CUDA enabled builds.
    No attempt is made to verify that the source is indeed on CUDA
    '''
    return C.numpy_array_to_cpu(numpy_array_on_cuda)


class Session:
    """
    This is the main class used to run a model.
    """
    def __init__(self):

        # self._sess is managed by the derived class and relies on bindings from C.InferenceSession
        self._sess = None
        self._enable_fallback = True

    def get_session_options(self):
        "Return the session options. See :class:`onnxruntime.SessionOptions`."
        return self._sess_options

    def get_inputs(self):
        "Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._inputs_meta

    def get_outputs(self):
        "Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._outputs_meta

    def get_overridable_initializers(self):
        "Return the inputs (including initializers) metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._overridable_initializers

    def get_modelmeta(self):
        "Return the metadata. See :class:`onnxruntime.ModelMetadata`."
        return self._model_meta

    def get_providers(self):
        "Return list of registered execution providers."
        return self._providers

    def get_provider_options(self):
        "Return registered execution providers' configurations."
        return self._provider_options

    def set_providers(self, providers=None, provider_options=None):
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

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """
        # recreate the underlying C.InferenceSession
        self._reset_session(providers, provider_options)

    def disable_fallback(self):
        """
        Disable session.run() fallback mechanism.
        """
        self._enable_fallback = False

    def enable_fallback(self):
        """
        Enable session.Run() fallback mechanism. If session.Run() fails due to an internal Execution Provider failure,
        reset the Execution Providers enabled for this session.
        If GPU is enabled, fall back to CUDAExecutionProvider.
        otherwise fall back to CPUExecutionProvider.
        """
        self._enable_fallback = True

    def run(self, output_names, input_feed, run_options=None):
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: See :class:`onnxruntime.RunOptions`.

        ::

            sess.run([output_name], {input_name: x})
        """
        num_required_inputs = len(self._inputs_meta)
        num_inputs = len(input_feed)
        # the graph may have optional inputs used to override initializers. allow for that.
        if num_inputs < num_required_inputs:
            raise ValueError("Model requires {} inputs. Input Feed contains {}".format(num_required_inputs, num_inputs))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return self._sess.run(output_names, input_feed, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print("EP Error: {} using {}".format(str(err), self._providers))
                print("Falling back to {} and retrying.".format(self._fallback_providers))
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return self._sess.run(output_names, input_feed, run_options)
            else:
                raise

    def run_with_ort_values(self, output_names, input_dict_ort_values, run_options=None):
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_ort_value }``
         See ``OrtValue`` class how to create OrtValue from numpy array or SparseTensor
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return an array of OrtValues
        ::

            sess.run([output_name], {input_name: x})
        """
        def invoke(sess, output_names, input_dict_ort_values, run_options):
            input_dict = {}
            for n, v in input_dict_ort_values.items():
                input_dict[n] = v._get_c_value()
            result = sess.run_with_ort_values(input_dict, output_names, run_options)
            ort_values = [OrtValue(v) for v in result]
            return ort_values

        num_required_inputs = len(self._inputs_meta)
        num_inputs = len(input_dict_ort_values)
        # the graph may have optional inputs used to override initializers. allow for that.
        if num_inputs < num_required_inputs:
            raise ValueError("Model requires {} inputs. Input Feed contains {}".format(num_required_inputs, num_inputs))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
            return invoke(self._sess, output_names, input_dict_ort_values, run_options)
        except C.EPFail as err:
            if self._enable_fallback:
                print("EP Error: {} using {}".format(str(err), self._providers))
                print("Falling back to {} and retrying.".format(self._fallback_providers))
                self.set_providers(self._fallback_providers)
                # Fallback only once.
                self.disable_fallback()
                return invoke(self._sess, output_names, input_dict_ort_values, run_options)
            else:
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

    def io_binding(self):
        "Return an onnxruntime.IOBinding object`."
        return IOBinding(self)

    def run_with_iobinding(self, iobinding, run_options=None):
        """
         Compute the predictions.

         :param iobinding: the iobinding object that has graph inputs/outputs bind.
         :param run_options: See :class:`onnxruntime.RunOptions`.
        """
        self._sess.run_with_iobinding(iobinding._iobinding, run_options)


class InferenceSession(Session):
    """
    This is the main class used to run a model.
    """
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        """
        :param path_or_bytes: filename or serialized ONNX or ORT format model in a byte string
        :param sess_options: session options
        :param providers: Optional sequence of providers in order of decreasing
            precedence. Values can either be provider names or tuples of
            (provider name, options dict). If not provided, then all available
            providers are used with the default precedence.
        :param provider_options: Optional sequence of options dicts corresponding
            to the providers listed in 'providers'.

        The model type will be inferred unless explicitly set in the SessionOptions.
        To explicitly set:
          so = onnxruntime.SessionOptions()
          so.add_session_config_entry('session.load_model_format', 'ONNX') or
          so.add_session_config_entry('session.load_model_format', 'ORT') or

        A file extension of '.ort' will be inferred as an ORT format model.
        All other filenames are assumed to be ONNX format models.

        'providers' can contain either names or names and options. When any options
        are given in 'providers', 'provider_options' should not be used.

        The list of providers is ordered by precedence. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
        means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
        """

        Session.__init__(self)

        if isinstance(path_or_bytes, str):
            self._model_path = path_or_bytes
            self._model_bytes = None
        elif isinstance(path_or_bytes, bytes):
            self._model_path = None
            self._model_bytes = path_or_bytes  # TODO: This is bad as we're holding the memory indefinitely
        else:
            raise TypeError("Unable to load from type '{0}'".format(type(path_or_bytes)))

        self._sess_options = sess_options
        self._sess_options_initial = sess_options
        self._enable_fallback = True
        self._read_config_from_model = os.environ.get('ORT_LOAD_CONFIG_FROM_MODEL') == '1'

        # internal parameters that we don't expect to be used in general so aren't documented
        disabled_optimizers = kwargs['disabled_optimizers'] if 'disabled_optimizers' in kwargs else None

        try:
            self._create_inference_session(providers, provider_options, disabled_optimizers)
        except ValueError:
            if self._enable_fallback:
                print("EP Error using {}".format(providers))
                print("Falling back to {} and retrying.".format(self._fallback_providers))
                self._create_inference_session(self._fallback_providers, None)
                # Fallback only once.
                self.disable_fallback()
            else:
                raise

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA. All others fall back to CPU.
        if 'TensorrtExecutionProvider' in available_providers:
            self._fallback_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self._fallback_providers = ['CPUExecutionProvider']

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(providers,
                                                                        provider_options,
                                                                        available_providers)

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()
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
        self._model_meta = self._sess.model_meta
        self._providers = self._sess.get_providers()
        self._provider_options = self._sess.get_provider_options()
        self._profiling_start_time_ns = self._sess.get_profiling_start_time_ns

    def _reset_session(self, providers, provider_options):
        "release underlying session object."
        # meta data references session internal structures
        # so they must be set to None to decrement _sess reference count.
        self._sess_options = None
        self._inputs_meta = None
        self._outputs_meta = None
        self._overridable_initializers = None
        self._model_meta = None
        self._providers = None
        self._provider_options = None
        self._profiling_start_time_ns = None

        # create a new C.InferenceSession
        self._sess = None
        self._sess_options = self._sess_options_initial
        self._create_inference_session(providers, provider_options)


class IOBinding:
    '''
    This class provides API to bind input/output to a specified device, e.g. GPU.
    '''
    def __init__(self, session):
        self._iobinding = C.SessionIOBinding(session._sess)
        self._numpy_obj_references = {}

    def bind_cpu_input(self, name, arr_on_cpu):
        '''
        bind an input to array on CPU
        :param name: input name
        :param arr_on_cpu: input values as a python array on CPU
        '''
        # Hold a reference to the numpy object as the bound OrtValue is backed
        # directly by the data buffer of the numpy object and so the numpy object
        # must be around until this IOBinding instance is around
        self._numpy_obj_references[name] = arr_on_cpu
        self._iobinding.bind_input(name, arr_on_cpu)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        '''
        :param name: input name
        :param device_type: e.g. cpu, cuda
        :param device_id: device id, e.g. 0
        :param element_type: input element type
        :param shape: input shape
        :param buffer_ptr: memory pointer to input data
        '''
        self._iobinding.bind_input(name,
                                   C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                               device_id),
                                   element_type, shape, buffer_ptr)

    def bind_ortvalue_input(self, name, ortvalue):
        '''
        :param name: input name
        :param ortvalue: OrtValue instance to bind
        '''
        self._iobinding.bind_ortvalue_input(name, ortvalue._ortvalue)

    def bind_output(self, name, device_type='cpu', device_id=0, element_type=None, shape=None, buffer_ptr=None):
        '''
        :param name: output name
        :param device_type: e.g. cpu, cuda, cpu by default
        :param device_id: device id, e.g. 0
        :param element_type: output element type
        :param shape: output shape
        :param buffer_ptr: memory pointer to output data
        '''

        # Follow the `if` path when the user has not provided any pre-allocated buffer but still
        # would like to bind an output to a specific device (e.g. cuda).
        # Pre-allocating an output buffer may not be an option for the user as :
        # (1) They may not want to use a custom allocator specific to the device they want to bind the output to,
        # in which case ORT will allocate the memory for the user
        # (2) The output has a dynamic shape and hence the size of the buffer may not be fixed across runs
        if buffer_ptr is None:
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id))
        else:
            if element_type is None or shape is None:
                raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")
            self._iobinding.bind_output(name,
                                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(),
                                                    device_id),
                                        element_type, shape, buffer_ptr)

    def bind_ortvalue_output(self, name, ortvalue):
        '''
        :param name: output name
        :param ortvalue: OrtValue instance to bind
        '''
        self._iobinding.bind_ortvalue_output(name, ortvalue._ortvalue)

    def get_outputs(self):
        '''
        Returns the output OrtValues from the Run() that preceded the call.
        The data buffer of the obtained OrtValues may not reside on CPU memory
        '''
        returned_ortvalues = []

        for ortvalue in self._iobinding.get_outputs():
            returned_ortvalues.append(OrtValue(ortvalue))

        return returned_ortvalues

    def copy_outputs_to_cpu(self):
        '''Copy output contents to CPU (if on another device). No-op if already on the CPU.'''
        return self._iobinding.copy_outputs_to_cpu()

    def clear_binding_inputs(self):
        self._iobinding.clear_binding_inputs()

    def clear_binding_outputs(self):
        self._iobinding.clear_binding_outputs()


class OrtValue:
    '''
    A data structure that supports all ONNX data formats (tensors and non-tensors) that allows users
    to place the data backing these on a device, for example, on a CUDA supported device.
    This class provides APIs to construct and deal with OrtValues.
    '''
    def __init__(self, ortvalue, numpy_obj=None):
        if isinstance(ortvalue, C.OrtValue):
            self._ortvalue = ortvalue
            # Hold a ref count to the numpy object if the OrtValue is backed directly
            # by its data buffer so that it isn't destroyed when the OrtValue is in use
            self._numpy_obj = numpy_obj
        else:
            # An end user won't hit this error
            raise ValueError("`Provided ortvalue` needs to be of type " +
                             "`onnxruntime.capi.onnxruntime_pybind11_state.OrtValue`")

    def _get_c_value(self):
        return self._ortvalue

    @staticmethod
    def ortvalue_from_numpy(numpy_obj, device_type='cpu', device_id=0):
        '''
        Factory method to construct an OrtValue (which holds a Tensor) from a given Numpy object
        A copy of the data in the Numpy object is held by the OrtValue only if the device is NOT cpu
        :param numpy_obj: The Numpy object to construct the OrtValue from
        :param device_type: e.g. cpu, cuda, cpu by default
        :param device_id: device id, e.g. 0
        '''
        # Hold a reference to the numpy object (if device_type is 'cpu') as the OrtValue
        # is backed directly by the data buffer of the numpy object and so the numpy object
        # must be around until this OrtValue instance is around
        return OrtValue(C.OrtValue.ortvalue_from_numpy(numpy_obj, C.OrtDevice(get_ort_device_type(device_type),
                        C.OrtDevice.default_memory(), device_id)), numpy_obj if device_type.lower() == 'cpu' else None)

    @staticmethod
    def ortvalue_from_shape_and_type(shape=None, element_type=None, device_type='cpu', device_id=0):
        '''
        Factory method to construct an OrtValue (which holds a Tensor) from given shape and element_type
        :param shape: List of integers indicating the shape of the OrtValue
        :param element_type: The data type of the elements in the OrtValue (numpy type)
        :param device_type: e.g. cpu, cuda, cpu by default
        :param device_id: device id, e.g. 0
        '''
        if shape is None or element_type is None:
            raise ValueError("`element_type` and `shape` are to be provided if pre-allocated memory is provided")

        return OrtValue(C.OrtValue.ortvalue_from_shape_and_type(shape, element_type,
                        C.OrtDevice(get_ort_device_type(device_type), C.OrtDevice.default_memory(), device_id)))

    @staticmethod
    def ort_value_from_sparse_tensor(sparse_tensor):
        '''
        The function will construct an OrtValue instance from a valid SparseTensor
        The new instance of OrtValue will assume the ownership of sparse_tensor
        '''
        return OrtValue(C.OrtValue.ort_value_from_sparse_tensor(sparse_tensor._get_c_tensor()))

    def as_sparse_tensor(self):
        '''
        The function will return SparseTensor contained in this OrtValue
        '''
        return SparseTensor(self._ortvalue.as_sparse_tensor())

    def data_ptr(self):
        '''
        Returns the address of the first element in the OrtValue's data buffer
        '''
        return self._ortvalue.data_ptr()

    def device_name(self):
        '''
        Returns the name of the device where the OrtValue's data buffer resides e.g. cpu, cuda
        '''
        return self._ortvalue.device_name().lower()

    def shape(self):
        '''
        Returns the shape of the data in the OrtValue
        '''
        return self._ortvalue.shape()

    def data_type(self):
        '''
        Returns the data type of the data in the OrtValue
        '''
        return self._ortvalue.data_type()

    def is_tensor(self):
        '''
        Returns True if the OrtValue contains a Tensor, else returns False
        '''
        return self._ortvalue.is_tensor()

    def is_sparse_tensor(self):
        '''
        Returns True if the OrtValue contains a SparseTensor, else returns False
        '''
        return self._ortvalue.is_sparse_tensor()

    def is_tensor_sequence(self):
        '''
        Returns True if the OrtValue contains a Tensor Sequence, else returns False
        '''
        return self._ortvalue.is_tensor_sequence()

    def numpy(self):
        '''
        Returns a Numpy object from the OrtValue.
        Valid only for OrtValues holding Tensors. Throws for OrtValues holding non-Tensors.
        Use accessors to gain a reference to non-Tensor objects such as SparseTensor
        '''
        return self._ortvalue.numpy()


class OrtDevice:
    '''
    A data structure that exposes the underlying C++ OrtDevice
    '''
    def __init__(self, c_ort_device):
        '''
        Internal constructor
        '''
        if isinstance(c_ort_device, C.OrtDevice):
            self._ort_device = c_ort_device
        else:
            raise ValueError("`Provided object` needs to be of type " +
                             "`onnxruntime.capi.onnxruntime_pybind11_state.OrtDevice`")

    def _get_c_device(self):
        '''
        Internal accessor to underlying object
        '''
        return self._ort_device

    @staticmethod
    def make(ort_device_name, device_id):
        return OrtDevice(C.OrtDevice(get_ort_device_type(ort_device_name),
                                     C.OrtDevice.default_memory(), device_id))

    def device_id(self):
        return self._ort_device.device_id()

    def device_type(self):
        return self._ort_device.device_type()


class SparseTensor:
    '''
    A data structure that project the C++ SparseTensor object
    The class provides API to work with the object.
    Depending on the format, the class will hold more than one buffer
    depending on the format
    '''
    def __init__(self, sparse_tensor):
        '''
        Internal constructor
        '''
        if isinstance(sparse_tensor, C.SparseTensor):
            self._tensor = sparse_tensor
        else:
            # An end user won't hit this error
            raise ValueError("`Provided object` needs to be of type " +
                             "`onnxruntime.capi.onnxruntime_pybind11_state.SparseTensor`")

    def _get_c_tensor(self):
        return self._tensor

    @staticmethod
    def sparse_coo_from_numpy(shape, values, coo_indices, ort_device):
        '''
        Factory method to construct a SparseTensor in COO format from given arguments
        :param shape: 1-D  numpy array(int64) or a python list that contains a dense_shape of the sparse tensor
         must be on cpu memory
        :param values: a homogeneous, contiguous 1-D numpy array that contains non-zero elements of the tensor
         of a type.
        :param coo_indices:  contiguous numpy array(int64) that contains COO indices for the tensor. coo_indices may
         have a 1-D shape when it contains a linear index of non-zero values and its length must be equal to
         that of the values. It can also be of 2-D shape, in which has it contains pairs of coordinates for
         each of the nnz values and its length must be exactly twice of the values length.
        :param ort_device - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
         suppored for non-numeric data types.

         For primitive types, the method will map values and coo_indices arrays into native memory and will use
         them as backing storage. It will increment the reference count for numpy arrays and it will decrement it
         on GC. The buffers may reside in any storage either CPU or GPU.
         For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
         on other devices and their memory can not be mapped.
        '''
        return SparseTensor(C.SparseTensor.sparse_coo_from_numpy(shape, values, coo_indices,
                            ort_device._get_c_device()))

    @staticmethod
    def sparse_csr_from_numpy(shape, values, inner_indices, outer_indices, ort_device):
        '''
        Factory method to construct a SparseTensor in CSR format from given arguments
        :param shape: 1-D numpy array(int64) or a python list that contains a dense_shape of the
         sparse tensor (rows, cols) must be on cpu memory
        :param values: a  contiguous, homogeneous 1-D numpy array that contains non-zero elements of the tensor
         of a type.
        :param inner_indices:  contiguous 1-D numpy array(int64) that contains CSR inner indices for the tensor.
         Its length must be equal to that of the values.
        :param outer_indices:  contiguous 1-D numpy array(int64) that contains CSR outer indices for the tensor.
         Its length must be equal to the number of rows + 1.
        :param ort_device - describes the backing memory owned by the supplied nummpy arrays. Only CPU memory is
         suppored for non-numeric data types.

         For primitive types, the method will map values and indices arrays into native memory and will use them as
         backing storage. It will increment the reference count and it will decrement then count when it is GCed.
         The buffers may reside in any storage either CPU or GPU.
         For strings and objects, it will create a copy of the arrays in CPU memory as ORT does not support those
         on other devices and their memory can not be mapped.
        '''
        return SparseTensor(C.SparseTensor.sparse_csr_from_numpy(shape, values, inner_indices, outer_indices,
                            ort_device._get_c_device()))

    def values(self):
        '''
        The method returns a numpy array that is backed by the native memory
        if the data type is numeric. Otherwise, the returned numpy array that contains
        copies of the strings.
        '''
        return self._tensor.values()

    def as_coo_view(self):
        '''
        The method will return coo representation of the sparse tensor which will enable
        querying COO indices. If the instance did not contain COO format, it would throw.
        You can query coo indices as:
          coo_indices = sparse_tensor.as_coo_view().indices()
          which will return a numpy array that is backed by the native memory
        '''
        return self._tensor.get_coo_data()

    def as_csrc_view(self):
        '''
        The method will return CSR(C) representation of the sparse tensor which will enable
        querying CRS(C) indices. If the instance dit not contain CSR(C) format, it would throw.
        You can query indices as:
          inner_ndices = sparse_tensor.as_csrc_view().inner()
          outer_ndices = sparse_tensor.as_csrc_view().outer()
          returning numpy arrays backed by the native memory
        '''
        return self._tensor.get_csrc_data()

    def as_blocksparse_view(self):
        '''
        The method will return coo representation of the sparse tensor which will enable
        querying BlockSparse indices. If the instance did not contain BlockSparse format, it would throw.
        You can query coo indices as:
          block_sparse_indices = sparse_tensor.as_blocksparse_view().indices()
          which will return a numpy array that is backed by the native memory
        '''
        return self._tensor.get_blocksparse_data()

    def to_cuda(self, ort_device):
        '''
        Returns a copy of this instance on the specified cuda device
        :param ort_device with name 'cuda' and valid gpu device id
        The method will throw if:
        - this instance contains strings
        - this instance is already on GPU. Cross GPU copy is not supported
        - CUDA is not present in this build
        - if the specified device is not valid
        '''
        return SparseTensor(self._tensor.to_cuda(ort_device._get_c_device()))

    def format(self):
        '''
        Returns a OrtSparseFormat enumeration
        '''
        return self._tensor.format

    def shape(self):
        '''
        Returns a numpy array(int64) containing a dense shape of a sparse tensor
        '''
        return self._tensor.shape()

    def data_type(self):
        '''
        Returns a string data type of the data in the OrtValue
        '''
        return self._tensor.data_type()

    def device_name(self):
        '''
        Returns the name of the device where the SparseTensor data buffers reside e.g. cpu, cuda
        '''
        return self._tensor.device_name().lower()
