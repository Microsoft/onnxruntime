# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import flatbuffers

_TOOLS_PYTHON = str(Path(__file__).resolve().parents[2])
if _TOOLS_PYTHON not in sys.path:
    sys.path.insert(0, _TOOLS_PYTHON)

from util.ort_format_model import OperatorTypeUsageManager, OrtFormatModelProcessor  # noqa: E402

from dump_ort_model import OrtFormatModelDumper  # noqa: E402  # isort:skip
import ort_flatbuffers_py.fbs as fbs  # noqa: E402  # isort:skip


def _vector(builder, start_vector, offsets):
    start_vector(builder, len(offsets))
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _fp6_ort_model():
    builder = flatbuffers.Builder(512)
    names = [builder.CreateString(name) for name in ("input", "output")]
    node_args = []
    for name, elem_type in zip(names, (27, 28), strict=True):
        fbs.TensorTypeAndShape.TensorTypeAndShapeStart(builder)
        fbs.TensorTypeAndShape.TensorTypeAndShapeAddElemType(builder, elem_type)
        tensor_type = fbs.TensorTypeAndShape.TensorTypeAndShapeEnd(builder)

        fbs.TypeInfo.TypeInfoStart(builder)
        fbs.TypeInfo.TypeInfoAddValueType(builder, fbs.TypeInfoValue.TypeInfoValue.tensor_type)
        fbs.TypeInfo.TypeInfoAddValue(builder, tensor_type)
        type_info = fbs.TypeInfo.TypeInfoEnd(builder)

        fbs.ValueInfo.ValueInfoStart(builder)
        fbs.ValueInfo.ValueInfoAddName(builder, name)
        fbs.ValueInfo.ValueInfoAddType(builder, type_info)
        node_args.append(fbs.ValueInfo.ValueInfoEnd(builder))

    inputs = _vector(builder, fbs.Node.NodeStartInputsVector, names[:1])
    outputs = _vector(builder, fbs.Node.NodeStartOutputsVector, names[1:])
    op_type = builder.CreateString("Cast")
    domain = builder.CreateString("")
    node_name = builder.CreateString("fp6_cast")
    fbs.Node.NodeStart(builder)
    fbs.Node.NodeAddName(builder, node_name)
    fbs.Node.NodeAddDomain(builder, domain)
    fbs.Node.NodeAddOpType(builder, op_type)
    fbs.Node.NodeAddSinceVersion(builder, 28)
    fbs.Node.NodeAddInputs(builder, inputs)
    fbs.Node.NodeAddOutputs(builder, outputs)
    node = fbs.Node.NodeEnd(builder)

    graph_node_args = _vector(builder, fbs.Graph.GraphStartNodeArgsVector, node_args)
    graph_nodes = _vector(builder, fbs.Graph.GraphStartNodesVector, [node])
    fbs.Graph.GraphStart(builder)
    fbs.Graph.GraphAddNodeArgs(builder, graph_node_args)
    fbs.Graph.GraphAddNodes(builder, graph_nodes)
    graph = fbs.Graph.GraphEnd(builder)

    fbs.Model.ModelStart(builder)
    fbs.Model.ModelAddGraph(builder, graph)
    model = fbs.Model.ModelEnd(builder)

    version = builder.CreateString("test")
    fbs.InferenceSession.InferenceSessionStart(builder)
    fbs.InferenceSession.InferenceSessionAddOrtVersion(builder, version)
    fbs.InferenceSession.InferenceSessionAddModel(builder, model)
    session = fbs.InferenceSession.InferenceSessionEnd(builder)
    builder.Finish(session, file_identifier=b"ORTM")
    return bytes(builder.Output())


class TestOrtFormatModelFp6(unittest.TestCase):
    def test_dump_and_collect_operator_types(self):
        model_bytes = _fp6_ort_model()
        with patch("builtins.open", side_effect=[io.BytesIO(model_bytes), io.BytesIO(model_bytes)]):
            dumper = OrtFormatModelDumper("fp6.ort")
            required_ops = {}
            processors = OperatorTypeUsageManager()
            processor = OrtFormatModelProcessor("fp6.ort", required_ops, processors)

        output = io.StringIO()
        dumper.dump(output)
        self.assertIn("input type=Float6E2M3", output.getvalue())
        self.assertIn("output type=Float6E3M2", output.getvalue())

        processor.process()
        self.assertEqual(required_ops, {"ai.onnx": {28: {"Cast"}}})
        self.assertEqual(
            json.loads(processors.get_config_entry("ai.onnx", "Cast")),
            {"inputs": {"0": ["Float6E2M3"]}, "outputs": {"0": ["Float6E3M2"]}},
        )


if __name__ == "__main__":
    unittest.main()
