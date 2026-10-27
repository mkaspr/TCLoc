"""
The exponential and logarithm maps are inverses, also near the identity where
Taylor series take over.
"""

import torch

from tcloc import lie


def _random_tangents(n: int, scale: float) -> torch.Tensor:
    return (torch.rand(n, 6, dtype=torch.float64) * 2 - 1) * scale


def test_se3_log_inverts_exp():
    # tiny angles use the Taylor series; |omega| < pi
    for scale in (1e-6, 1e-3, 0.1, 1.0, 1.7):
        xi = _random_tangents(100, scale)
        torch.testing.assert_close(
            lie.se3_log(lie.se3_exp(xi)), xi, rtol=1e-6, atol=1e-9
        )


def test_exp_gives_rigid_transforms():
    T = lie.se3_exp(_random_tangents(100, 1.0))
    R = T[:, :3, :3]
    torch.testing.assert_close(
        R @ R.transpose(1, 2), torch.eye(3, dtype=R.dtype).expand_as(R)
    )
    torch.testing.assert_close(
        torch.linalg.det(R), torch.ones(100, dtype=R.dtype)
    )


def test_inverse_and_points():
    T = lie.se3_exp(_random_tangents(10, 1.0))
    torch.testing.assert_close(
        T @ lie.inverse(T), torch.eye(4, dtype=T.dtype).expand_as(T)
    )
    X = torch.randn(10, 50, 3, dtype=T.dtype)
    torch.testing.assert_close(
        lie.transform_points(lie.inverse(T), lie.transform_points(T, X)), X
    )


def test_gradients_finite_at_identity():
    xi = torch.zeros(4, 6, dtype=torch.float64, requires_grad=True)
    lie.se3_log(lie.se3_exp(xi)).sum().backward()
    assert xi.grad is not None and torch.isfinite(xi.grad).all()


def test_quaternion_matches_exp():
    axis = torch.nn.functional.normalize(
        torch.randn(20, 3, dtype=torch.float64), dim=-1
    )
    angle = torch.rand(20, 1, dtype=torch.float64) * 3
    q = torch.cat((axis * torch.sin(angle / 2), torch.cos(angle / 2)), -1)
    t = torch.randn(20, 3, dtype=torch.float64)
    torch.testing.assert_close(
        lie.from_quaternion(t, q)[:, :3, :3], lie.so3_exp(axis * angle)
    )
