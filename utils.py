import argparse
from copy import deepcopy
from math import log2

#------------------------------ Argparse utilities ----------------------------------#
def argparse_str2list(args, delimiter=','):
    """
    Convert string containing delimiter to list of numbers
    (using lists with arparse don't work with wandb sweeps)
    """
    args_copy = deepcopy(args)
    for key, val in vars(args_copy).items():
        if type(val) is str and delimiter in val:
            setattr(args_copy, key, [float(item) if '.' in item else int(item) for item in val.split(delimiter)])
    return args_copy


def argparse_str2bool(v):
    """
    Relaxed boolean type checking for argparse
    (wandb sweeps requires flags to take explicit str values such as 'True' or 'False')
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#------------------------ Tensor duplication along batch axis ----------------------#
def reshape_4d_batch(batch):
    return batch.reshape(batch.shape[0] * batch.shape[1], *batch.shape[2:])


def expand_4d_batch(batch, n):
    if n == 0 or n == 1:
        return batch
    return reshape_4d_batch(batch.unsqueeze(0).expand(n, *([-1] * len(batch.shape))))


def restore_expanded_4d_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return expanded_batch.reshape(n, expanded_batch.shape[0] // n, *expanded_batch.shape[1:])


def mean_expanded_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return restore_expanded_4d_batch(expanded_batch, n).mean(0)


def std_expanded_batch(expanded_batch, n):
    if n == 0 or n == 1:
        return expanded_batch
    return restore_expanded_4d_batch(expanded_batch, n).std(0)

#------------------------ Architecture generation -------------------------------#
def get_block_scaling(in_dim, out_dim, max_scaling):
    """
    Generates consecutive scaling factors to go from high resolution to low resolution

    :param in_dim: max resolution
    :param out_dim: min resolution
    :param max_scaling: highest scaling factor
    :return: ex: get_block_scaling(64, 2, 4) --> [4, 4, 2]
    """
    log_res_ratio = int(log2(in_dim // out_dim))
    log_scale = int(log2(max_scaling))
    mapping = []
    while log_res_ratio > 0:
        mapping.extend([int(2 ** log_scale)] * (log_res_ratio // log_scale))
        log_res_ratio %= log_scale
        log_scale -= 1
    return mapping

#-------------------------------- Other ------------------------------------#
def hparams2desc(parser, args, hparam_names, delimiter='-', verbose='vvv'):
    desc = ''
    for action in parser._get_optional_actions():
        if len(set(hparam_names) & set(map(lambda s: s.replace('-', ''), action.option_strings))) > 0:
            arg_short_name, arg_name = action.option_strings[0].replace('-', ''), action.option_strings[-1].replace('-', '')
            desc += delimiter
            if verbose == 'v':
                pass
            elif verbose == 'vv':
                desc += arg_short_name
            elif verbose == 'vvv':
                desc += arg_name + '='
            else:
                raise ValueError(f"verbose choice is ['v', 'vv', 'vvv']. Provided verbose={verbose}")
            desc += str(getattr(args, arg_name))
    return desc
