from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def mean_squared_distance(input, target, reduction='mean', w=None):  # type: ignore
    out = np.square(input - target)
    if w is not None:
        out = np.multiply(out, w)
    if reduction == 'mean':
        out = np.mean(out)
    elif reduction == 'sum':
        out = np.sum(out)
    return out


class MeanSquaredDistance(Base):

    @staticmethod
    def export_mean_square_distance_none():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([1.2, 2.5], dtype=np.float32)
        t = np.array([1.1, 2.6], dtype=np.float32)

        # Compute Mean Square Distance
        msd = mean_squared_distance(r, t, reduction='none')

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_none')

    @staticmethod
    def export_mean_square_distance_none_weights():  # type: () -> None
        # Define operator attributes.
        reduction = 'none'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T', 'W'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([1.2, 2.5], dtype=np.float32)
        t = np.array([1.1, 2.6], dtype=np.float32)
        weights = np.array([0.8, 0.9], dtype=np.float32)

        # Compute Mean Square Distance
        msd = mean_squared_distance(r, t, reduction='none', w=weights)

        # Check results
        expect(node, inputs=[r, t, weights], outputs=[msd], name='test_mean_square_distance_none_weights')

    @staticmethod
    def export_mean_square_distance_sum():  # type: () -> None
        # Define operator attributes.
        reduction = 'sum'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[1.2, 2.5, 3.1], [1.3, 2.3, 3.4]], dtype=np.float32)
        t = np.array([[1.1, 2.6, 3.2], [1.4, 2.2, 3.3]], dtype=np.float32)

        # Compute Mean Square Distance
        msd = mean_squared_distance(r, t, reduction='sum')

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_sum')

    @staticmethod
    def export_mean_square_distance_mean():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[1.2, 2.5, 3.1], [1.3, 2.3, 3.4]], dtype=np.float32)
        t = np.array([[1.1, 2.6, 3.2], [1.4, 2.2, 3.3]], dtype=np.float32)

        # Compute Mean Square Distance
        sq = np.square(r - t)
        msd = np.mean(sq)

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_mean')

    @staticmethod
    def export_mean_square_distance_mean_3d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        r = np.array([[[1.2, 2.5], [3.1, 1.3]], [[2.3, 3.4], [1.1, 2.2]], [[3.6, 1.7], [2.5, 3.8]]], dtype=np.float32)
        t = np.array([[[1.1, 2.6], [3.2, 1.4]], [[2.2, 3.3], [1.2, 2.1]], [[3.5, 1.6], [2.5, 3.9]]], dtype=np.float32)

        # Compute Mean Square Distance
        msd = mean_squared_distance(r, t, reduction='mean')

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_mean_3d')

    @staticmethod
    def export_mean_square_distance_mean_4d():  # type: () -> None
        # Define operator attributes.
        reduction = 'mean'

        # Create operator.
        node = onnx.helper.make_node('MeanSquaredDistance',
                                     inputs=['R', 'T'],
                                     outputs=['X'],
                                     reduction=reduction
                                     )

        # Define operator inputs
        np.random.seed(0)
        r = np.random.rand(2, 4, 5, 7)
        t = np.random.rand(2, 4, 5, 7)

        # Compute Mean Square Distance
        msd = mean_squared_distance(r, t, reduction='mean')

        # Check results
        expect(node, inputs=[r, t], outputs=[msd], name='test_mean_square_distance_mean_4d')