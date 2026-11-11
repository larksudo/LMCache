# SPDX-License-Identifier: Apache-2.0
"""Compatibility home for the page-buffer shape descriptor factory.

The function itself lives in :mod:`lmcache.v1.gpu_connector.utils`. It is
re-exported here because LMCache-Ascend patches it *in this module*
(``lmcache_ascend.v1.shape_desc.install_plane_geometry_fill`` rebinds
``lmcache.v1.common.utils.make_page_buffer_shape_desc`` to attach the
per-plane byte geometry its Ascend KV formats require). Every call site
must import the name from this module so the patch applies to it.
"""

# First Party
from lmcache.v1.gpu_connector.utils import make_page_buffer_shape_desc

__all__ = ["make_page_buffer_shape_desc"]
