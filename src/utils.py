import torch

def print_rank(message):
    """If distributed is initialized, print the rank."""
    if torch.distributed.is_initialized():
        print(f'rank{torch.distributed.get_rank()}: ' + message)
    else:
        print(message)


def print_master(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message)
    else:
        print(message)
