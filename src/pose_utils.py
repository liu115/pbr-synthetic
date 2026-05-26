"""Renderer-free camera pose helpers.

``pose_to_c2w`` reproduces the exact 4x4 camera-to-world matrix that Mitsuba's
``ScalarTransform4f.look_at`` would build from a ``CameraPose``, but without
importing Mitsuba. Backends that don't have Mitsuba available (e.g. Blender)
can use this directly, and the Mitsuba-backed pipeline keeps the same output
bytes for ``transforms.json``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.camera_sampling import CameraPose


def pose_to_c2w(pose: CameraPose) -> NDArray[np.float64]:
    """Return the 4x4 camera-to-world matrix that Mitsuba's ``look_at`` produces.

    Columns 0/1/2 are the camera-frame X/Y/Z basis vectors in world space and
    column 3 is the origin. The convention matches Mitsuba: camera +Z is the
    look direction (target - origin), +Y is the world up vector passed in, and
    +X is ``cross(up, forward)``.
    """
    origin = np.asarray(pose.position, dtype=np.float64)
    target = np.asarray(pose.target(distance=1.0), dtype=np.float64)
    up = np.asarray(pose.world_up(), dtype=np.float64)

    forward = target - origin
    fn = float(np.linalg.norm(forward))
    if fn == 0.0:
        raise ValueError(f"Degenerate pose: target == origin ({origin})")
    forward = forward / fn

    x_axis = np.cross(up, forward)
    xn = float(np.linalg.norm(x_axis))
    if xn == 0.0:
        raise ValueError(
            f"Degenerate pose: forward parallel to world_up ({up}); "
            f"forward={forward}"
        )
    x_axis = x_axis / xn

    y_axis = np.cross(forward, x_axis)
    yn = float(np.linalg.norm(y_axis))
    if yn > 0.0:
        y_axis = y_axis / yn

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = x_axis
    c2w[:3, 1] = y_axis
    c2w[:3, 2] = forward
    c2w[:3, 3] = origin
    return c2w
