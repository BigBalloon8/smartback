from typing import Union, Callable, Sequence
from functools import wraps

import torch
import torch.distributed as dist

def data_frag(*args, in_axes: Union[int, tuple], num_devices: int):
    new_args = [[] for _ in range(num_devices)]
    in_axes = (in_axes,) * len(args) if isinstance(in_axes, int) else in_axes
    if len(in_axes) != len(args):
        raise ValueError("len of in_axes must match len of args")
    for d, a in zip(in_axes, args):
        if not isinstance(a, torch.Tensor):
            raise TypeError("Only tensors can be mapped")
        if d is not None:
            a = torch.chunk(a, chunks=num_devices, dim=d)
            for i in range(num_devices):
                new_args[i].append(a[i])
        else:
            for i in range(num_devices):
                new_args[i].append(a)

    return new_args


def data_to_device(*args):
    return [a.to(torch.cuda.current_device()) for a in args]


def scalar_to_vec(*args):
    # if output is scalar convert to vecotr of shape (1,1) for all_gather
    return tuple(map(lambda x: x.unsqueeze(0) if x.dim() == 0 else x, args))


def custom_all_gather(
    x: torch.Tensor,
    axis: int = 0,
    group: dist.ProcessGroup = dist.group.WORLD,
    tiled: bool = False,
    out=None,
):
    num_processes = group.size()
    tensor_in = x.contiguous() if axis == 0 else x.transpose(0, axis).contiguous()
    if out is None:
        tensor_out = [
            torch.empty(tensor_in.shape, dtype=tensor_in.dtype, device=tensor_in.device)
            for _ in range(num_processes)
        ]
    else:
        if isinstance(out, torch.Tensor):
            tensor_out = torch.chunk(out, chunks=num_processes)
        elif isinstance(out, list):
            tensor_out = out
        else:
            raise Exception("out must be list of tensors or tensor")

    dist.all_gather(tensor_out, tensor_in, group=group)
    tensor_out = torch.concat(tensor_out)
    tensor_out = tensor_out if axis == 0 else tensor_out.transpose(0, axis)

    if tiled:
        tensor_out = torch.chunk(tensor_out, chunks=num_processes, axis=axis)

    return tensor_out


def pmap(
    fn: Callable,
    *,
    in_axes: Union[int, Sequence[int]] = 0,
    out_axes: Union[int, Sequence[int]] = 0,
    group: Union[dist.group, None] = None,
    dst: int = 0
) -> Callable:
    rank = dist.get_rank(group=group)
    num_processes = dist.get_world_size(group=group)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if num_processes == 1:
            return torch.vmap(fn, in_axes=in_axes, out_axes=out_axes)(*args, **kwargs)

        # Run Function
        new_args = data_frag(*args, in_axes=in_axes, num_devices=num_processes)
        data_to_device(*new_args[rank])
        func_out = torch.vmap(fn, in_axes=in_axes, out_axes=out_axes)(
            *new_args[rank], **kwargs
        )
        func_out = scalar_to_vec(*func_out)

        out_axes_ = (
            (out_axes,) * len(func_out) if isinstance(out_axes, int) else out_axes
        )

        # collect function outputs
        # TODO cleanup (repetitive code)
        if dst == -1:
            if isinstance(func_out, tuple):
                out = []
                for i in range(len(func_out)):
                    out.append(
                        custom_all_gather(func_out[i], out_axes_[i], group=group)
                    )
                return tuple(out)
            else:
                return custom_all_gather(func_out, out_axes_[0], group=group)
        else:
            if isinstance(func_out, tuple):
                out = []
                for i in range(len(func_out)):
                    out.append(
                        custom_all_gather(
                            func_out[i], out_axes_[i], group=group, tiled=True, dst=dst
                        )
                    )

                    return tuple(out)
            else:
                return custom_all_gather(func_out, out_axes_[0], group=group, dst=dst)

    return wrapper