import copy
import torch


def extract_tensors(m: torch.nn.Module) -> tuple[torch.nn.Module, list[dict]]:
    """
    Remove the tensors from a PyTorch model, convert them to NumPy
    arrays, and return the stripped model and tensors.
    """
    tensors = []
    for _, module in m.named_modules():
        # Store the tensors in Python dictionaries
        params = {
            name: torch.clone(param).detach().numpy()
            for name, param in module.named_parameters(recurse=False)
        }
        buffers = {
            name: torch.clone(buf).detach().numpy()
            for name, buf in module.named_buffers(recurse=False)
        }
        tensors.append({"params": params, "buffers": buffers})

    # Make a copy of the original model and strip all tensors and
    # buffers out of the copy.
    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in ([name for name, _ in module.named_parameters(recurse=False)]
                     + [name for name, _ in module.named_buffers(recurse=False)]):
            setattr(module, name, None)

            # Make sure the copy is configured for inference.
    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: list[dict]):
    """
    Restore the tensors that extract_tensors() stripped out of a
    PyTorch model.
    :param no_parameters_objects: Skip wrapping tensors in
     ``torch.nn.Parameters`` objects (~20% speedup, may impact
     some models)
    """
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(name,
                torch.nn.Parameter(torch.as_tensor(array)))
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array))
