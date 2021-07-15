# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Register pytorch symbolic for export using ONNX Runtime contrib ops

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args


_onnx_opset_version = 1


def register_custom_op(is_ortmodule=False):
    """
    This function registers symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    # Symbolic definition
    def inverse(g, self):
        return g.op("com.microsoft::Inverse", self).setType(self.type())

    def gelu(g, self):
        return g.op("com.microsoft::Gelu", self).setType(self.type())

    def triu(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=1).setType(self.type())

    def tril(g, self, diagonal):
        return g.op("com.microsoft::Trilu", self, diagonal, upper_i=0).setType(self.type())

    # Op Registration
    register_custom_op_symbolic('::inverse', inverse, _onnx_opset_version)
    register_custom_op_symbolic('::gelu', gelu, _onnx_opset_version)
    register_custom_op_symbolic('::triu', triu, _onnx_opset_version)
    register_custom_op_symbolic('::tril', tril, _onnx_opset_version)

    if is_ortmodule:
        @parse_args('v', 'v', 'v', 'i', 'v')
        def cross_entropy_loss(g, self, target, weight, reduction, ignore_index):
            # reduction: 0->none, 1->mean, 2->sum
            reduction = sym_help._maybe_get_const(reduction, 'i')
            reduction_vals = ['none', 'mean', 'sum']
            reduction = reduction_vals[reduction]
            output, log_prob = g.op("com.microsoft::SoftmaxCrossEntropyLossInternal",
                                    self, target, weight, ignore_index,
                                    reduction_s=reduction, outputs=2)
            output.setType(self.type())
            log_prob.setType(self.type())
            return output

        register_custom_op_symbolic('::cross_entropy_loss', cross_entropy_loss, _onnx_opset_version)

        @parse_args('v', 'v', 'v', 'i', 'v')
        def nll_loss(g, self, target, weight, reduction, ignore_index):
            # reduction: 0->none, 1->mean, 2->sum
            reduction = sym_help._maybe_get_const(reduction, 'i')
            reduction_vals = ['none', 'mean', 'sum']
            reduction = reduction_vals[reduction]
            output = g.op("com.microsoft::NegativeLogLikelihoodLossInternal",
                          self, target, weight, ignore_index, reduction_s=reduction)
            output.setType(self.type())
            return output

        register_custom_op_symbolic('::nll_loss', nll_loss, _onnx_opset_version)


def unregister_custom_op():
    """
    This function unregisters symbolic functions for
    custom ops that are implemented as part of ONNX Runtime
    """

    import torch.onnx.symbolic_registry as sym_registry

    # TODO: replace this once PyTorch supports unregister natively.
    def unregister(name, opset_version):
        ns, kind = name.split("::")
        from torch.onnx.symbolic_helper import _onnx_stable_opsets

        for version in _onnx_stable_opsets:
            if version >= opset_version and sym_registry.is_registered_op(kind, ns, version):
                del sym_registry._registry[(ns, version)][kind]

    unregister('::inverse', _onnx_opset_version)
    unregister('::gelu', _onnx_opset_version)
    unregister('::triu', _onnx_opset_version)
    unregister('::tril', _onnx_opset_version)
