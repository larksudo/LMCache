# SPDX-License-Identifier: Apache-2.0
"""Ascend NPU platform helpers."""

# First Party
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_pin_memory import PinMemoryBackend
from lmcache.v1.platform.npu.pin_memory import NpuPinMemoryBackend

# ---------------------------------------------------------------------------
# Device detection registry entry
# ---------------------------------------------------------------------------


class NpuDeviceSpec(DeviceSpec):
    """Ascend NPU device spec for the detection registry.

    The compiled ops backend (``lmcache_ascend.c_ops``) and NPU-specific
    transfer optimisation live in the external ``lmcache_ascend`` plugin; this
    entry wires device detection, host-memory pinning, and backend selection so
    ``import lmcache`` resolves the NPU pin backend eagerly when ``DeviceExt`` is
    constructed on an NPU host.
    """

    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def torch_module_name(self) -> str:
        return "npu"

    @property
    def ops_module(self) -> str | None:
        return "lmcache_ascend.c_ops"

    @property
    def pin_memory_backend(self) -> type[PinMemoryBackend] | None:
        return NpuPinMemoryBackend

    def is_available(self) -> bool:
        """Check NPU availability without importing lmcache.__init__."""
        try:
            # Third Party
            import torch

            return hasattr(torch, "npu") and torch.npu.is_available()
        except Exception:
            return False
