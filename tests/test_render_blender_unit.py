"""Unit-level tests for the Blender backend that don't require a real bpy.

The expensive end-to-end tests (rendering, addon install) live in
``test_render_blender.py`` and are gated behind the ``blender_slow`` marker.
The checks here are pure-Python and run in the fast suite — they catch
regressions in the AOV plumbing constants and the AOVImages contract.
"""

from __future__ import annotations

import numpy as np

from src.render import AOVImages
from src.render_blender import _SHADER_AOV_INPUT, _SHADER_AOV_NAMES


def test_shader_aov_names_match_principled_sockets() -> None:
    """The AOV name table must map every declared AOV to a known BSDF socket."""
    assert set(_SHADER_AOV_INPUT.keys()) == set(_SHADER_AOV_NAMES)
    # These socket names are the canonical Cycles Principled-BSDF input names
    # (used both by bpy 3.x and bpy 4.x; the addon-compat patches deal with
    # any version-specific renames at addon level).
    assert _SHADER_AOV_INPUT["PixelRoughness"] == "Roughness"
    assert _SHADER_AOV_INPUT["PixelMetallic"] == "Metallic"


def test_aovimages_accepts_new_fields_as_optional() -> None:
    """Existing Mitsuba-only call-sites that pass 4 fields must still work."""
    h, w = 4, 5
    aov = AOVImages(
        depth=np.zeros((h, w), dtype=np.float32),
        normal=np.zeros((h, w, 3), dtype=np.float32),
        albedo=np.zeros((h, w, 3), dtype=np.float32),
        shape_index=np.zeros((h, w), dtype=np.int32),
    )
    assert aov.glossy_color is None
    assert aov.emission is None
    assert aov.material_index is None
    assert aov.roughness_per_pixel is None
    assert aov.metallic_per_pixel is None


def test_aovimages_carries_all_new_fields() -> None:
    h, w = 3, 4
    aov = AOVImages(
        depth=np.zeros((h, w), dtype=np.float32),
        normal=np.zeros((h, w, 3), dtype=np.float32),
        albedo=np.zeros((h, w, 3), dtype=np.float32),
        shape_index=np.zeros((h, w), dtype=np.int32),
        glossy_color=np.ones((h, w, 3), dtype=np.float32) * 0.04,
        emission=np.zeros((h, w, 3), dtype=np.float32),
        material_index=np.arange(h * w, dtype=np.int32).reshape(h, w),
        roughness_per_pixel=np.full((h, w), 0.5, dtype=np.float32),
        metallic_per_pixel=np.zeros((h, w), dtype=np.float32),
    )
    assert aov.glossy_color is not None
    assert aov.glossy_color.shape == (h, w, 3)
    assert aov.emission is not None
    assert aov.material_index is not None and aov.material_index.dtype == np.int32
    assert aov.roughness_per_pixel is not None
    assert aov.metallic_per_pixel is not None
    np.testing.assert_array_equal(
        aov.material_index, np.arange(h * w, dtype=np.int32).reshape(h, w)
    )
