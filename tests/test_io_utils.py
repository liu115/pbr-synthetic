"""Unit tests for src/io_utils.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.io_utils import (
    METADATA_FILENAME,
    SUBDIRS,
    TRANSFORMS_FILENAME,
    build_frame_record,
    compute_intrinsics,
    existing_rendered_frames,
    frame_stem,
    make_output_layout,
    read_exr,
    write_exr,
    write_metadata_json,
    write_transforms_json,
)


def test_compute_intrinsics_known_fov() -> None:
    intr = compute_intrinsics(60.0, 640, 480)
    assert intr.width == 640 and intr.height == 480
    assert intr.cx == 320.0 and intr.cy == 240.0
    # fl_x for 60 deg horizontal on 640 wide: 640 / (2 tan 30°)
    expected = 640.0 / (2.0 * np.tan(np.deg2rad(30.0)))
    assert intr.fl_x == pytest.approx(expected)
    assert intr.fl_y == intr.fl_x


def test_make_output_layout_creates_all_subdirs(tmp_output_dir: Path) -> None:
    make_output_layout(tmp_output_dir)
    for sub in SUBDIRS:
        assert (tmp_output_dir / sub).is_dir()


def test_frame_stem_zero_pads() -> None:
    assert frame_stem(0) == "0000"
    assert frame_stem(42) == "0042"
    assert frame_stem(9999) == "9999"


def test_existing_rendered_frames_finds_pre_existing(tmp_output_dir: Path) -> None:
    make_output_layout(tmp_output_dir)
    (tmp_output_dir / "rgb_hdr" / "0003.exr").write_bytes(b"placeholder")
    (tmp_output_dir / "rgb_hdr" / "0007.exr").write_bytes(b"placeholder")
    (tmp_output_dir / "rgb_hdr" / "ignore_me.txt").write_bytes(b"x")
    result = existing_rendered_frames(tmp_output_dir)
    assert result == {3, 7}


def test_existing_rendered_frames_returns_empty_when_dir_missing(
    tmp_output_dir: Path,
) -> None:
    # No layout created
    assert existing_rendered_frames(tmp_output_dir) == set()


def test_write_and_read_exr_roundtrip(tmp_output_dir: Path) -> None:
    arr = np.random.RandomState(0).rand(8, 8, 3).astype(np.float32)
    path = tmp_output_dir / "rgb_hdr" / "0000.exr"
    write_exr(path, arr)
    assert path.exists()
    loaded = read_exr(path)
    assert loaded.shape == arr.shape
    assert np.allclose(loaded, arr, atol=1e-4)


def test_build_frame_record_paths_and_matrix() -> None:
    mat = np.eye(4)
    mat[:3, 3] = [1.0, 2.0, 3.0]
    rec = build_frame_record(5, mat)
    assert rec["file_path"] == "rgb/0005"
    assert rec["depth_path"] == "depth/0005.exr"
    assert rec["hdr_path"] == "rgb_hdr/0005.exr"
    assert len(rec["transform_matrix"]) == 4
    assert rec["transform_matrix"][0][0] == 1.0
    assert rec["transform_matrix"][2][3] == 3.0


def test_build_frame_record_rejects_bad_matrix() -> None:
    with pytest.raises(ValueError):
        build_frame_record(0, np.eye(3))


def test_write_transforms_json_payload(tmp_output_dir: Path) -> None:
    intr = compute_intrinsics(60.0, 640, 480)
    frames = [build_frame_record(0, np.eye(4))]
    out = tmp_output_dir / TRANSFORMS_FILENAME
    write_transforms_json(out, intr, "y", frames)
    loaded = json.loads(out.read_text())
    assert loaded["coordinate_convention"] == "opencv"
    assert loaded["up_axis"] == "y"
    assert loaded["w"] == 640
    assert loaded["h"] == 480
    assert loaded["camera_angle_x"] == pytest.approx(intr.fov_x_rad)
    assert len(loaded["frames"]) == 1


def test_write_metadata_json_writes_render_datetime(
    tmp_output_dir: Path,
) -> None:
    out = tmp_output_dir / METADATA_FILENAME
    write_metadata_json(out, {"seed": 42, "scene": "test"})
    loaded = json.loads(out.read_text())
    assert loaded["seed"] == 42
    assert loaded["scene"] == "test"
    assert "render_datetime_utc" in loaded
