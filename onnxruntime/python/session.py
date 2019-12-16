#-------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#--------------------------------------------------------------------------

import sys
import os

from onnxruntime.capi import _pybind_state as C

def getOrtDeviceType(device):
    if device == 'cuda':
        return C.OrtDevice.cuda()
    elif device == 'cpu':
        return C.OrtDevice.cpu()
    else:
        raise Exception('Unsupported device type: ' + torch_device.type)

class Session:
    """
    This is the main class used to run a model.
    """
    def __init__(self, sess):
        """
        :param path_or_bytes: filename or serialized model in a byte string
        :param sess_options: session options
        """
        self._sess = sess;
        self._inputs_meta = self._sess.inputs_meta
        self._outputs_meta = self._sess.outputs_meta
        self._model_meta = self._sess.model_meta

    def get_inputs(self):
        "Return the inputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._inputs_meta

    def get_outputs(self):
        "Return the outputs metadata as a list of :class:`onnxruntime.NodeArg`."
        return self._outputs_meta

    def get_modelmeta(self):
        "Return the metadata. See :class:`onnxruntime.ModelMetadata`."
        return self._model_meta

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
        return self._sess.run(output_names, input_feed, run_options)

    def end_profiling(self):
        """
        End profiling and return results in a file.

        The results are stored in a filename if the option
        :meth:`onnxruntime.SessionOptions.enable_profiling`.
        """
        return self._sess.end_profiling()

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
    def __init__(self, path_or_bytes, sess_options=None):
        if sess_options:
            sess = C.InferenceSession(
                sess_options, C.get_session_initializer())
        else:
            sess = C.InferenceSession(
                C.get_session_initializer(), C.get_session_initializer())

        if isinstance(path_or_bytes, str):
            sess.load_model(path_or_bytes)
        elif isinstance(path_or_bytes, bytes):
            sess.read_bytes(path_or_bytes)
        elif isinstance(path_or_bytes, tuple):
            # to remove, hidden trick
            sess.load_model_no_init(path_or_bytes[0])
        else:
            raise TypeError("Unable to load from type '{0}'".format(type(path_or_bytes)))
        Session.__init__(self, sess);


class IOBinding:
    def __init__(self, session):
        self._iobinding = C.SessionIOBinding(session._sess)

    def bind_input(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        self._iobinding.bind_input(name,
                                   C.OrtDevice(getOrtDeviceType(device_type), C.OrtDevice.default_memory(), device_id),
                                   element_type, shape, buffer_ptr)

    def bind_output(self, name, device_type, device_id, element_type, shape, buffer_ptr):
        self._iobinding.bind_output(name,
                                    C.OrtDevice(getOrtDeviceType(device_type), C.OrtDevice.default_memory(), device_id),
                                    element_type, shape, buffer_ptr)
    
    def clear_binding_inputs(self):
        self._iobinding.clear_binding_inputs()

    def clear_binding_outputs(self):
        self._iobinding.clear_binding_outputs()


class TrainingSession(InferenceSession):
    def __init__(self, path_or_bytes, parameters, sess_options=None):
        if sess_options:
            sess = C.TrainingSession(
                sess_options, C.get_session_initializer())
        else:
            sess = C.TrainingSession(
                C.get_session_initializer(), C.get_session_initializer())

        if isinstance(path_or_bytes, str):
            sess.load_model(path_or_bytes, parameters)
        elif isinstance(path_or_bytes, bytes):
            sess.read_bytes(path_or_bytes, parameters)
        else:
            raise TypeError("Unable to load from type '{0}'".format(type(path_or_bytes)))
        Session.__init__(self, sess);

    def __del__(self):
        if self._sess:
            self._sess.finalize()

    def get_state(self):
        return self._sess.get_state()

    def load_state(self, dict, strict=False):
        self._sess.load_state(dict, strict)
