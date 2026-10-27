"""
4Seasons (TUM, https://cvg.cit.tum.de/data/datasets/4seasons-dataset):
recordings to the tcloc data format.

A recording, as downloaded and unzipped, holds
``undistorted_images/cam{0,1}/<ns>.png`` (a rectified greyscale stereo pair, 800
x 400, 30 cm baseline), ``GNSSPoses.txt`` (one pose per keyframe, about every
metre) and ``Transformations.txt``; the shared ``calibration/`` folder holds the
intrinsics.

Poses. ``GNSSPoses.txt`` gives each keyframe's camera-to-world pose (camera
axes: x right, y down, z forward) in the recording's SLAM frame, with a scale.
As in the dataset's own tools (libartipy), a camera position p reaches
Earth-centred (ECEF) coordinates as

    T_e_gpsw @ inverse(T_w_gpsw) @ scale @ T_S_AS @ p

(``transform_e_gpsw``, ``transform_w_gpsw``, ``GNSS scale`` and
``transform_S_AS`` in ``Transformations.txt``). These poses fuse visual-inertial
odometry with RTK-GNSS: metric and consistent across recordings to centimetres.
We express them in a local east-north-up frame shared by all recordings of a
dataset, and use them both as the visual odometry pose and the world position.

Images. The bottom rows show the bonnet, so the top 304 rows are kept (800 x
304). Depth comes from the stereo pair (tcloc.stereo) with settings scaled to
this camera.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from tcloc import lie
from tcloc.stereo import StereoConfig, depth, disparity

# keep the rows above the bonnet (it reaches up to row ~310); width and height
# are multiples of 16
CROP_ROWS = 304

# Nearest depth fx * baseline / 112 px = 1.3 m; blocks scaled from RobotCar's 15
# px at 1280 wide.
STEREO = StereoConfig(
    num_disparities=112, block_size=9, disparity_offset=0.0, max_depth=150.0
)


def quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to a unit quaternion.

    Stable for any angle (Shepperd's method).

    Args:
        R: (3, 3) float64 rotation matrix.

    Returns:
        (4,) float64 unit quaternion (x, y, z, w), with w >= 0.
    """
    t = np.trace(R)

    # Build from the largest component.
    i = int(np.argmax([t, R[0, 0], R[1, 1], R[2, 2]]))
    q = np.zeros(4)

    if i == 0:
        w = np.sqrt(1 + t) / 2
        q[:] = (
            (R[2, 1] - R[1, 2]) / (4 * w),
            (R[0, 2] - R[2, 0]) / (4 * w),
            (R[1, 0] - R[0, 1]) / (4 * w),
            w,
        )
    else:
        a, b, c = (i - 1, i % 3, (i + 1) % 3)
        q[a] = np.sqrt(1 + 2 * R[a, a] - t) / 2
        q[b] = (R[b, a] + R[a, b]) / (4 * q[a])
        q[c] = (R[c, a] + R[a, c]) / (4 * q[a])
        q[3] = (R[c, b] - R[b, c]) / (4 * q[a])

    return q if q[3] >= 0 else -q


def calibration(directory: Path) -> tuple[np.ndarray, float]:
    """
    Read the rectified left camera's intrinsics and the stereo baseline.

    Args:
        directory: The unzipped 4Seasons ``calibration`` folder.

    Returns:
        A tuple ``(K, baseline)``: K, (3, 3) float64 intrinsics of the
        undistorted 800 x 400 images; baseline in metres.
    """
    calib0_path = directory / "undistorted_calib_0.txt"
    fx, fy, cx, cy = map(float, calib0_path.read_text().split()[1:5])
    T = np.loadtxt(directory / "undistorted_calib_stereo.txt")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), float(abs(T[0, 3]))


def _transforms(recording: Path) -> dict[str, np.ndarray]:
    """
    Read the named entries of ``Transformations.txt``.

    Each value is on the line after its ``# name:`` header; the result maps
    each name to its values as a float64 array.
    """
    lines = (recording / "Transformations.txt").read_text().splitlines()

    return {
        line[1:].split(":")[0].strip(): np.array(lines[i + 1].split(","), float)
        for i, line in enumerate(lines)
        if line.startswith("#")
    }


def _pose(v: np.ndarray) -> np.ndarray:
    """(4, 4) float64 transform from (7,) ``tx, ty, tz, qx, qy, qz, qw``."""
    t, q = torch.from_numpy(v[None, :3]), torch.from_numpy(v[None, 3:7])
    return lie.from_quaternion(t, q)[0].numpy()


def enu_rotation(ecef: np.ndarray) -> np.ndarray:
    """
    Rotation from ECEF to the local east-north-up axes at an ECEF point.

    Uses the WGS84 ellipsoid (geodetic latitude).

    Args:
        ecef: (3,) float64 Earth-centred, Earth-fixed position, metres.

    Returns:
        (3, 3) float64 rotation R_enu_ecef.
    """
    x, y, z = ecef
    a, e2 = 6378137.0, 6.69437999014e-3
    lon, p = np.arctan2(y, x), np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - e2))

    for _ in range(5):  # fixed-point iteration for the geodetic latitude
        n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + e2 * n * np.sin(lat), p)

    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)

    return np.array(
        [[-so, co, 0], [-sl * co, -sl * so, cl], [cl * co, cl * so, sl]]
    )


def keyframes(
    recording: Path, origin_ecef: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a recording's keyframe timestamps and camera-to-ENU poses.

    Args:
        recording: The unzipped ``recording_<start time>`` folder.
        origin_ecef: (3,) float64 ECEF origin of the ENU frame, metres.

    Returns:
        A tuple ``(timestamps, poses)``: timestamps, (N,) int64 keyframe
        timestamps in nanoseconds; poses, (N, 4, 4) float64 transforms
        T_enu_cam (camera-to-ENU poses, metres) in the east-north-up frame
        at ``origin_ecef``.
    """

    rows = np.loadtxt(recording / "GNSSPoses.txt", delimiter=",", comments="#")
    tr = _transforms(recording)
    scale = np.diag([tr["GNSS scale"][0]] * 3 + [1.0])

    slam_to_ecef = (
        _pose(tr["transform_e_gpsw"])
        @ np.linalg.inv(_pose(tr["transform_w_gpsw"]))
        @ scale
        @ _pose(tr["transform_S_AS"])
    )

    to_enu = np.eye(4)
    to_enu[:3, :3] = enu_rotation(origin_ecef)
    to_enu[:3, 3] = -to_enu[:3, :3] @ origin_ecef
    poses = np.stack([_pose(r[1:8]) for r in rows])
    T = to_enu @ slam_to_ecef @ poses

    # The scale belongs to positions only: keep the rotations orthonormal.
    T[:, :3, :3] /= tr["GNSS scale"][0]
    return rows[:, 0].astype(np.int64), T


def origin(out: Path, recording: Path) -> np.ndarray:
    """
    The dataset's shared ENU origin.

    Stored in ``<out>/enu_origin.json``; if missing, it is set from the
    recording given (the first one prepared) and written.

    Args:
        out: Output root of the prepared dataset.
        recording: The unzipped ``recording_<start time>`` folder.

    Returns:
        (3,) float64 ECEF position of the origin, metres.
    """

    path = out / "enu_origin.json"

    if not path.exists():
        out.mkdir(parents=True, exist_ok=True)
        ecef = _transforms(recording)["transform_e_gpsw"][:3].tolist()
        path.write_text(json.dumps({"ecef": ecef}) + "\n")

    return np.array(json.loads(path.read_text())["ecef"])


def image_and_depth(
    left: np.ndarray,
    right: np.ndarray,
    fx: float,
    baseline: float,
    cfg: StereoConfig = STEREO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cropped greyscale image and its depth from a rectified stereo pair.

    Args:
        left: (400, 800) uint8 greyscale rectified left image.
        right: (400, 800) uint8 greyscale rectified right image.
        fx: Focal length of the left camera, pixels.
        baseline: Stereo baseline, metres.
        cfg: Stereo matching settings.

    Returns:
        A tuple ``(image, depth)``: image, (304, 800) uint8 greyscale left
        image cropped above the bonnet; depth, (304, 800) float32 depth of
        that image in metres, 0 where unknown.
    """
    left, right = left[:CROP_ROWS], right[:CROP_ROWS]
    disp = disparity(left, right, cfg)
    return left, depth(disp, fx, baseline, left.shape, cfg)
