"""Unit-level tests for the Blender backend that don't require a real bpy.

The expensive end-to-end tests (rendering, addon install) live in
``test_render_blender.py`` and are gated behind the ``blender_slow`` marker.
The checks here are pure-Python and run in the fast suite — they catch
regressions in the AOV plumbing constants and the AOVImages contract.
"""

from __future__ import annotations

import numpy as np

from src.render import AOVImages
from src.render_blender import (
    BLENDER_UNLIMITED_BOUNCES,
    _BSDF_METALLIC_DEFAULT,
    _BSDF_ROUGHNESS_DEFAULT,
    _BSDF_ROUGHNESS_USES_SOCKET,
    _SHADER_AOV_NAMES,
    _cycles_bounce_cap,
)


def test_shader_aov_names_are_unique() -> None:
    assert sorted(_SHADER_AOV_NAMES) == ["PixelMetallic", "PixelRoughness"]


def test_cycles_bounce_cap_maps_unlimited_sentinel() -> None:
    """Mitsuba's max_depth=-1 (unlimited) maps to a finite Cycles cap; finite
    values pass through unchanged."""
    assert _cycles_bounce_cap(-1) == BLENDER_UNLIMITED_BOUNCES
    assert _cycles_bounce_cap(8) == 8
    assert _cycles_bounce_cap(0) == 0
    assert _cycles_bounce_cap(64) == 64


def test_diffuse_bsdf_reports_microfacet_fully_rough() -> None:
    """Cycles Diffuse BSDF -> roughness = 1.0 under the microfacet convention."""
    assert _BSDF_ROUGHNESS_DEFAULT["ShaderNodeBsdfDiffuse"] == 1.0
    assert _BSDF_METALLIC_DEFAULT["ShaderNodeBsdfDiffuse"] == 0.0


def test_glossy_anisotropic_bsdf_treated_as_metal() -> None:
    """Cycles Anisotropic / Glossy nodes are conductors — metallic AOV = 1.0."""
    assert _BSDF_METALLIC_DEFAULT["ShaderNodeBsdfAnisotropic"] == 1.0
    assert _BSDF_METALLIC_DEFAULT["ShaderNodeBsdfGlossy"] == 1.0


def test_principled_bsdf_reads_socket() -> None:
    """For Principled, the Roughness socket value is taken directly."""
    assert "ShaderNodeBsdfPrincipled" in _BSDF_ROUGHNESS_USES_SOCKET
    assert "ShaderNodeBsdfAnisotropic" in _BSDF_ROUGHNESS_USES_SOCKET


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
