import pickle
import logging

import torch
import torch_xla.core.xla_model as xm

__all__ = [
    "custom_gather",
    "custom_all_gather",
]

def custom_gather(data, dst=0, group=None):
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
    data_list = custom_all_gather(data, group)
    rank = xm.get_ordinal()
    return data_list if rank == dst else []

def custom_all_gather(data, group=None):
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
    if custom_get_world_size(group) == 1:
        return [data]

    tensor = custom_serialize_to_tensor(data)

    size_list, tensor = custom_pad_to_largest_tensor(tensor, group)
    #max_size = max(size_list)

    # receiving Tensor from all ranks
    #tensor_list = [
    #    torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    #]
    tensor_list = xm.all_gather(tensor, groups=None if group is None else [group]).tolist()

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def custom_get_world_size(group=None):
    if group is None:
        return xm.xrt_world_size()
    ordinal = xm.get_ordinal()
    if ordinal not in group:
        return -1
    return len(group)

def custom_serialize_to_tensor(data):
    #backend = dist.get_backend(group)
    #assert backend in ["gloo", "nccl"]
    #device = torch.device("cpu" if backend == "gloo" else "cuda")
    device = xm.xla_device()
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                xm.get_ordinal(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def custom_pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = custom_get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    #size_list = [
    #    torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    #]
    size_list = xm.all_gather(local_size, groups=None if group is None else [group]).tolist()
    #size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor