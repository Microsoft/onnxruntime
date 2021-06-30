`get_providers`: Return list of registered execution providers.
`get_provider_options`: Return the registered execution providers' configurations.
`set_providers`: Register the given list of execution providers. The underlying session is re-created. 
    The list of providers is ordered by Priority. For example ['CUDAExecutionProvider', 'CPUExecutionProvider']
    means execute a node using CUDAExecutionProvider if capable, otherwise execute using CPUExecutionProvider.
