from .quant_utils import QuantizationMode
from .operators.base_operator import QuantOperatorBase
from .operators.qdq_base_operator import QDQOperatorBase
from .operators.matmul import MatMulInteger, QLinearMatMul
from .operators.attention import AttentionQuant
from .operators.embed_layernorm import EmbedLayerNormalizationQuant
from .operators.gather import GatherQuant
from .operators.conv import QLinearConv, ConvInteger, QDQConv
from .operators.activation import QLinearActivation, QDQRemovableActivation
from .operators.binary_op import QLinearBinaryOp
from .operators.maxpool import QDQMaxPool, QMaxPool
from .operators.gavgpool import QGlobalAveragePool
from .operators.lstm import LSTMQuant
from .operators.skip_layer_norm import SkipLayerNormalizationQuant
from .operators.split import QSplit
from .operators.pad import QPad
from .operators.direct_q8 import Direct8BitOp, QDQDirect8BitOp
from .operators.resize import QResize, QDQResize
from .operators.pooling import QLinearPool
from .operators.concat import QLinearConcat, QDQConcat

CommonOpsRegistry = {
    "Gather": GatherQuant,
    "EmbedLayerNormalization": EmbedLayerNormalizationQuant,
    "SkipLayerNormalization": SkipLayerNormalizationQuant,
}

IntegerOpsRegistry = {
    "Conv": ConvInteger,
    "MatMul": MatMulInteger,
    "Attention": AttentionQuant,
    "LSTM": LSTMQuant,
}
IntegerOpsRegistry.update(CommonOpsRegistry)

QLinearOpsRegistry = {
    "Conv": QLinearConv,
    "MatMul": QLinearMatMul,
    "Add": QLinearBinaryOp,
    "Mul": QLinearBinaryOp,
    "Relu": QLinearActivation,
    "Clip": QLinearActivation,
    "LeakyRelu": QLinearActivation,
    "Sigmoid": QLinearActivation,
    "MaxPool": QMaxPool,
    "GlobalAveragePool": QGlobalAveragePool,
    "Split": QSplit,
    "Pad": QPad,
    "Reshape": Direct8BitOp,
    "Transpose" : Direct8BitOp,
    "Squeeze" : Direct8BitOp,
    "Unsqueeze" : Direct8BitOp,
    "Resize": QResize,
    "AveragePool" : QLinearPool,
    "Concat": QLinearConcat,
}
QLinearOpsRegistry.update(CommonOpsRegistry)

QDQRegistry = {
    "Conv": QDQConv,
    "Clip": QDQRemovableActivation,
    "Relu": QDQRemovableActivation,
    "Reshape": QDQDirect8BitOp,
    "Transpose" : QDQDirect8BitOp,
    "Squeeze" : QDQDirect8BitOp,
    "Unsqueeze" : QDQDirect8BitOp,
    "Resize": QDQResize,
    "MaxPool": QDQMaxPool,
    "AveragePool" : QDQDirect8BitOp,
    "Concat": QDQConcat,
}


def CreateDefaultOpQuantizer(onnx_quantizer, node):
    return QuantOperatorBase(onnx_quantizer, node)


def CreateOpQuantizer(onnx_quantizer, node):
    registry = IntegerOpsRegistry if onnx_quantizer.mode == QuantizationMode.IntegerOps else QLinearOpsRegistry
    if node.op_type in registry.keys():
        return registry[node.op_type](onnx_quantizer, node)
    return QuantOperatorBase(onnx_quantizer, node)


def CreateQDQQuantizer(onnx_quantizer, node):
    if node.op_type in QDQRegistry.keys():
        return QDQRegistry[node.op_type](onnx_quantizer, node)
    return QDQOperatorBase(onnx_quantizer, node)
