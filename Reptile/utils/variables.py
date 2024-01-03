import torch
import torch.nn.functional as F
from collections import OrderedDict

def interpolate_vars(old_vars, new_vars, epsilon):

    old_vars = [v.detach() for v in old_vars]
    new_vars = [v.detach() for v in new_vars]
    interpolated_vars = []
    for old_var, new_var in zip(old_vars, new_vars):
        interpolated_var = old_var + epsilon * (new_var - old_var)
        interpolated_vars.append(interpolated_var)
    return interpolated_vars


def subtract_vars(var_seq_1, var_seq_2):

    return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):

    return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):

    return [v * scale for v in var_seq]

def projection(param1, param2):
    """
    Project param1 onto param2.
    
    Args:
    - param1: First parameter tensor.
    - param2: Second parameter tensor.
    
    Returns:
    - Projected param1 onto param2.
    """
    param1 = torch.cat([p.flatten() for p in param1.values()])
    # param2 = torch.cat([p.flatten() for p in param2])
    dot_product = torch.dot(param1.view(-1), param2.view(-1))
    projection = dot_product / torch.norm(param2)**2
    projected_param1 = projection * param2
    return projected_param1
    

def apply_params(model, init_params, mean_params, epsilon):
    
    updated_params = OrderedDict()
    
    for i, (name, param) in enumerate(model.state_dict().items()):
        update_param = torch.stack(interpolate_vars(init_params[name], mean_params[name], epsilon))
        updated_params[name] = update_param 
    
    model.load_state_dict(updated_params)
    
    return model
