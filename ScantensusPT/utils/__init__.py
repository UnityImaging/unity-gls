from collections import OrderedDict

import torch


def load_checkpoint(checkpoint_path, device="cpu"):
    if device == "cpu":
        print(f'Loading weights onto CPU')
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    elif device == "cuda":
        print(f'Loading weights onto GPU')
        checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    else:
        raise Exception("Device not recognised")

    return checkpoint_dict


def fix_state_dict(state_dict, remove_module_prefix=True):

    if remove_module_prefix:
        module_prefix = "module."
        module_prefix_len = len(module_prefix)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith(module_prefix):
                name = k[module_prefix_len:]
            else:
                name = k[:]  # want a copy
            new_state_dict[name] = v

        return new_state_dict
    else:
        return state_dict


def load_and_fix_state_dict(checkpoint_path, device="cpu", remove_module_prefix=True):
    checkpoint_dict = load_checkpoint(checkpoint_path=checkpoint_path, device=device)
    state_dict = fix_state_dict(state_dict=checkpoint_dict, remove_module_prefix=remove_module_prefix)
    return state_dict

