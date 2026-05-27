"""Verify the renderer-free pose_to_c2w matches Mitsuba's ScalarTransform4f.look_at."""

from __future__ import annotations

import numpy as np
import pytest

from src.camera_sampling import CameraPose
from src.pose_utils import pose_to_c2w, pose_to_c2w_opencv


_POSES = [
    CameraPose(position=(0.0, 1.5, 0.0), yaw=0.0, pitch=0.0, up_axis="y"),
    CameraPose(position=(1.2, 1.0, -0.3), yaw=0.7, pitch=-0.2, up_axis="y"),
    CameraPose(position=(-0.5, 1.7, 0.4), yaw=-1.1, pitch=0.3, up_axis="y"),
    CameraPose(position=(0.0, 0.0, 1.5), yaw=0.0, pitch=0.0, up_axis="z"),
    CameraPose(position=(0.8, -0.4, 1.1), yaw=2.3, pitch=-0.1, up_axis="z"),
    CameraPose(position=(0.0, 0.5, 0.5), yaw=1.5, pitch=0.4, up_axis="x"),
]


def test_pose_to_c2w_is_rigid() -> None:
    """The matrix should always be a proper rigid SE(3) transform."""
    for pose in _POSES:
        m = pose_to_c2w(pose)
        assert m.shape == (4, 4)
        # Rotation part is orthogonal (R R^T = I).
        R = m[:3, :3]
        eye = R @ R.T
        np.testing.assert_allclose(eye, np.eye(3), atol=1e-9)
        # Right-handed (det = +1).
        assert np.linalg.det(R) > 0
        # Translation matches origin.
        np.testing.assert_allclose(m[:3, 3], np.asarray(pose.position), atol=1e-12)
        # Last row is [0,0,0,1].
        np.testing.assert_array_equal(m[3, :], np.array([0.0, 0.0, 0.0, 1.0]))


def test_pose_to_c2w_forward_axis_matches_pose_forward() -> None:
    """Column 2 of c2w should equal CameraPose.forward()."""
    for pose in _POSES:
        m = pose_to_c2w(pose)
        np.testing.assert_allclose(m[:3, 2], pose.forward(), atol=1e-10)


def test_pose_to_c2w_opencv_is_rigid() -> None:
    """OpenCV-convention c2w should also be a proper SE(3) transform."""
    for pose in _POSES:
        m = pose_to_c2w_opencv(pose)
        assert m.shape == (4, 4)
        R = m[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) > 0
        np.testing.assert_allclose(m[:3, 3], np.asarray(pose.position), atol=1e-12)


def test_pose_to_c2w_opencv_forward_axis_matches_pose_forward() -> None:
    """Column 2 of the OpenCV c2w is the look direction (= ``pose.forward()``)."""
    for pose in _POSES:
        m = pose_to_c2w_opencv(pose)
        np.testing.assert_allclose(m[:3, 2], pose.forward(), atol=1e-10)


def test_pose_to_c2w_opencv_basis_directions() -> None:
    """col 0 = right (cross(forward, up)), col 1 = down (= -col1 of look_at)."""
    for pose in _POSES:
        m_cv = pose_to_c2w_opencv(pose)
        forward = pose.forward()
        up = pose.world_up()
        expected_right = np.cross(forward, up)
        expected_right /= np.linalg.norm(expected_right)
        np.testing.assert_allclose(m_cv[:3, 0], expected_right, atol=1e-10)
        # col 1 should be ``cross(forward, col 0)`` — points in the image-down
        # direction (which is the negation of the look-at convention's +Y).
        m_lookat = pose_to_c2w(pose)
        np.testing.assert_allclose(m_cv[:3, 1], -m_lookat[:3, 1], atol=1e-10)


def test_pose_to_c2w_opencv_equivalent_to_diag_flip() -> None:
    """For non-degenerate poses: pose_to_c2w_opencv == pose_to_c2w @ diag(-1,-1,1,1)."""
    flip = np.diag([-1.0, -1.0, 1.0, 1.0])
    for pose in _POSES:
        np.testing.assert_allclose(
            pose_to_c2w_opencv(pose), pose_to_c2w(pose) @ flip, atol=1e-10
        )


@pytest.mark.mitsuba_slow
def test_pose_to_c2w_matches_mitsuba_look_at(mitsuba_variant: str) -> None:
    """Numpy implementation must reproduce Mitsuba's ScalarTransform4f.look_at."""
    import mitsuba as mi  # noqa: F401  (variant set by fixture)
    import mitsuba as mi_module

    for pose in _POSES:
        ours = pose_to_c2w(pose)
        T = mi_module.ScalarTransform4f.look_at(
            origin=list(pose.position),
            target=list(pose.target(distance=1.0)),
            up=pose.world_up().tolist(),
        )
        theirs = np.asarray(T.matrix, dtype=np.float64).reshape(4, 4)
        # Mitsuba's ScalarTransform4f.look_at works in float32 internally,
        # so we allow ~float32 tolerance instead of double precision.
        np.testing.assert_allclose(ours, theirs, atol=1e-6)
