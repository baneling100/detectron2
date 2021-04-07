import pickle
import logging

import torch
import torch_xla.core.xla_model as xm

__all__ = [
    "custom_gather",
    "custom_all_gather",
]

def custom_gather(data, dst=0):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []
    """
    data_list = custom_all_gather(data)
    rank = xm.get_ordinal()
    return data_list if rank == dst else []

def custom_all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: A list, representing the replica group for the all_gather() operation.
            Example: [0, 1, 2, 3]

    Returns:
        list[data]: list of data gathered from each rank
    """
    if xm.xrt_world_size() == 1:
        return [data]
    #if group is None:
    #    group = _get_global_gloo_group()

    buffer = pickle.dumps(data)

    #size_list, tensor = custom_pad_to_largest_tensor(tensor)
    #max_size = max(size_list)

    # receiving Tensor from all ranks
    #tensor_list = [
    #    torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    #]
    #tensor_list = xm.all_gather(tensor).cpu().tolist()
    xdata = xm.rendezvous("all_gather", buffer)

    data_list = []
    for xd in xdata:
        #buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(xd))

    return data_list