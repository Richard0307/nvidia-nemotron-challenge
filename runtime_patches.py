from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


IMPORT_STUBS = [
    "cutlass",
    "cutlass.cute",
    "mamba_ssm.modules.mamba3",
    "mamba_ssm.ops.cute",
    "mamba_ssm.ops.cute.mamba3",
    "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
]


def pure_rmsnorm_fn(
    x,
    weight,
    bias=None,
    z=None,
    eps: float = 1e-5,
    group_size=None,
    norm_before_gate: bool = True,
    upcast: bool = True,
):
    del group_size, norm_before_gate

    dtype = x.dtype
    if upcast:
        x = x.float()

    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()

    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())

    return out.to(dtype)


def patch_rmsnorm() -> None:
    for _, module in list(sys.modules.items()):
        if hasattr(module, "rmsnorm_fn"):
            module.rmsnorm_fn = pure_rmsnorm_fn


def install_import_stubs(enabled: bool) -> None:
    if not enabled:
        return

    for module_name in IMPORT_STUBS:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    sys.modules["mamba_ssm.modules.mamba3"].Mamba3 = None


def disable_nemotron_fast_path(enabled: bool) -> None:
    if not enabled:
        return

    for name, module in list(sys.modules.items()):
        if "modeling_nemotron_h" in name and hasattr(module, "is_fast_path_available"):
            module.is_fast_path_available = False

