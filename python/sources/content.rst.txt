This document describes ORTModule PyTorch frontend API for the ONNX Runtime (aka ORT) training acelerator.

What is new
===========

Version 0.1
-----------

#. Initial version

Overview
========
The aim of ORTModule is to provide a drop-in replacement for one or more torch.nn.Module objects in a user’s PyTorch program,
and execute the forward and backward passes of those modules using ORT.

As a result, the user will be able to accelerate their training script gradually using ORT,
without having to modify their training loop.

Users will be able to use standard PyTorch debugging techniques for convergence issues,
e.g. by probing the computed gradients on the model’s parameters.

The following code example illustrates how ORTModule would be used in a user’s training script,
in the simple case where the entire model can be offloaded to ONNX Runtime:

.. code-block:: python

    # Original PyTorch model
    class NeuralNet(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            ...
        def forward(self, x): 
            ...

    model = NeuralNet(input_size=784, hidden_size=500, num_classes=10)
    model = ORTModule(model) # Only change to original PyTorch script
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    # Training Loop is unchanged
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

API
===

.. automodule:: onnxruntime.training.ortmodule.ortmodule
    :members:
    :show-inheritance:
    :member-order: bysource
    :inherited-members:
