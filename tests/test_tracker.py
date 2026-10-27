"""The tracker recovers a known pose between two views of a synthetic
textured wall."""

import torch
from torch.nn import functional as F

from tcloc import lie
from tcloc.tracker import Tracker, TrackerConfig


def _render(
    texture: torch.Tensor,
    K: torch.Tensor,
    T_cam_wall: torch.Tensor,
    h: int,
    w: int,
    wall_z: float,
):
    """Image of a textured plane z = wall_z (in the wall frame) seen from a
    camera at T_cam_wall."""
    rows, cols = torch.meshgrid(
        torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij"
    )
    rays = torch.stack(
        (
            (cols - K[0, 2]) / K[0, 0],
            (rows - K[1, 2]) / K[1, 1],
            torch.ones_like(cols),
        ),
        -1,
    )
    T_wall_cam = lie.inverse(T_cam_wall[None])[0]
    R, c = T_wall_cam[:3, :3], T_wall_cam[:3, 3]
    d = rays @ R.T
    s = (wall_z - c[2]) / d[..., 2]  # ray-plane intersection
    P = c + s[..., None] * d
    depth = s * 1.0  # rays have z = 1 in the camera frame, so s is the depth
    # the wall spans 16 m x 12 m
    grid = torch.stack((P[..., 0] / 8, P[..., 1] / 6), -1)[None]
    image = F.grid_sample(
        texture, grid, align_corners=False, padding_mode="reflection"
    )
    return image, depth[None, None]


def test_recovers_known_pose():
    torch.manual_seed(0)
    h, w, levels = 96, 128, 3
    K = torch.tensor([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    # smooth random "features"
    texture = F.avg_pool2d(torch.rand(1, 8, 256, 256), 5, stride=1, padding=2)
    ref_image, ref_depth = _render(texture, K, torch.eye(4), h, w, wall_z=10.0)
    T_q_r = lie.se3_exp(torch.tensor([[0.3, -0.1, 0.4, 0.01, 0.03, -0.01]]))[0]
    query_image, _ = _render(texture, K, T_q_r, h, w, wall_z=10.0)

    tracker = Tracker(TrackerConfig(iterations=(20, 20, 20)))

    def pyramid(x: torch.Tensor) -> list[torch.Tensor]:
        return [F.avg_pool2d(x, 2**lv) if lv else x for lv in range(levels)]

    ones = [torch.ones_like(x[:, :1]) for x in pyramid(ref_image)]
    reference = tracker.prepare_reference(
        pyramid(ref_image), ones, ref_depth, K[None]
    )
    T = tracker.track(
        reference, pyramid(query_image), ones, K[None], torch.eye(4)[None]
    )[-1][0]

    error = lie.se3_log((T @ lie.inverse(T_q_r[None]))[0])
    assert error[:3].norm() < 0.01  # metres
    assert error[3:].norm() < 1e-3  # radians
