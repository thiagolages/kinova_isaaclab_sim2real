import pytest
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def dist_eff_to_target_z_axis(eff_pos, target):
    """
    Compute the distance from the end effector to a line defined by a target position and rotation.
    Position needs to be in the world frame, rotation is a 3x3 rotation matrix.
    Args:
        eff_pos: Effector position (3D vector).
        target: A tuple (target_position, target_rotation) where target_position is a 3D vector and target_rotation is a quaternion (4D vector) in scalar-last format.
    Returns:
        The distance from the end effector to the line defined by the target position and rotation.
    """
    target_pos, target_quat = target  # Unpack the target tuple
    
    # Get target pose in world frame
    if not target_quat.shape == (4,):
        raise ValueError("target_quat must be a quaternion (4,)")
    
    if not eff_pos.shape == (3,):
        raise ValueError("eff_pos must be a 3D vector (3,)")

    print("target_quat = ", target_quat)
    target_rot = torch.tensor(R.from_quat(target_quat).as_matrix(),dtype=torch.float)  # Convert quaternion to rotation matrix
    print("target_rot = ", target_rot)
    # Sanity check for rotation matrix
    if not target_rot.shape == (3, 3):
        raise ValueError("target_rot must be a rotation matrix (3, 3)")

    dz_unit = target_rot[:, 2]  # Extract the z-axis from the rotation matrix
    print("dz_unit =", dz_unit)
    print("eff_pos =", eff_pos)
    print("target_pos =", target_pos)
    vec = eff_pos - target_pos  # Vector from end effector to target    
    print("vec = eff_pos - target_pos = ", vec)
    proj_vec = torch.dot(vec, dz_unit) * dz_unit # Projection of vec onto the z-axis of the target frame
    print("proj_vec =", proj_vec)
    perp_vec = vec - proj_vec  # Perpendicular vector from effector to the line
    print("perp_vec =", perp_vec)
    dist = torch.norm(perp_vec)  # Norm of the perperdicular vector gives the distance
    print("dist =", dist)

    return dist

def dist_eff_to_target_xy_plane(eff_pos, target):
    """
    Compute the signed distance from the end effector to the XY plane defined by a target position and orientation.
    The plane normal is the Z axis of the target's rotation (from target_quat).
    Args:
        eff_pos: (num_envs, 3) tensor of end effector positions in world frame.
        target: tuple (target_pos, target_quat), where target_pos is (num_envs, 3), target_quat is (num_envs, 4) in scalar-last format.
    Returns:
        Signed distance (num_envs,) from eff_pos to the target's XY plane (positive if above, negative if below).
    """
    target_pos, target_quat = target  # (num_envs, 3), (num_envs, 4)
    
    # The Z axis in the target frame (in world coordinates)
    z_axis = torch.zeros_like(target_pos)
    z_axis[:, 2] = 1.0
    
    # Rotate z_axis by target_quat to get world normal direction
    z_axis_world = quat_apply(target_quat, z_axis)  # (num_envs, 3)
    
    # Vector from plane point to effector
    vec = eff_pos - target_pos  # (num_envs, 3)
    
    # Signed distance: projection onto normal
    signed_dist = torch.sum(vec * z_axis_world, dim=1)
    
    return signed_dist

# FIX TESTS
# @pytest.mark.parametrize(
#     "eff_pos,target_pos,target_quat,expected_dist",
#     [
#         # Aligned with world Z, eff_pos above plane
#         (torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([2.0])),
#         # Aligned with world Z, eff_pos below plane
#         (torch.tensor([[0.0, 0.0, -3.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([-3.0])),
#         # Offset plane, eff_pos above
#         (torch.tensor([[1.0, 2.0, 5.0]]), torch.tensor([[1.0, 2.0, 1.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([4.0])),
#         # Offset plane, eff_pos below
#         (torch.tensor([[1.0, 2.0, -2.0]]), torch.tensor([[1.0, 2.0, 1.0]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]]), torch.tensor([-3.0])),
#         # Rotated 180 deg about X (Z axis down)
#         (torch.tensor([[0.0, 0.0, 2.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([-2.0])),
#         # Rotated 90 deg about Y (Z axis to X)
#         (torch.tensor([[2.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0, 0.0]]), torch.tensor([2.0])),
#         # Rotated 90 deg about X (Z axis to Y)
#         (torch.tensor([[0.0, 3.0, 0.0]]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([[0.7071068, 0.7071068, 0.0, 0.0]]), torch.tensor([3.0])),
#     ]
# )
# def test_dist_eff_to_target_xy_plane_various(eff_pos, target_pos, target_quat, expected_dist):
#     result = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
#     assert torch.allclose(result, expected_dist, atol=1e-6)

# def test_dist_eff_to_target_xy_plane_invalid_shapes():
#     # eff_pos wrong shape
#     eff_pos = torch.tensor([0.0, 0.0])  # Should be (N, 3)
#     target_pos = torch.tensor([[0.0, 0.0, 0.0]])
#     target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
#     with pytest.raises(RuntimeError):
#         dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
#     # target_pos wrong shape
#     eff_pos = torch.tensor([[0.0, 0.0, 0.0]])
#     target_pos = torch.tensor([[0.0, 0.0]])  # Should be (N, 3)
#     with pytest.raises(RuntimeError):
#         dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
#     # target_quat wrong shape
#     target_pos = torch.tensor([[0.0, 0.0, 0.0]])
#     target_quat = torch.tensor([[0.0, 0.0, 0.0]])  # Should be (N, 4)
#     with pytest.raises(RuntimeError):
#         dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))

# Target position at origin, Z-axis aligned with world Z. No rotation.
@pytest.mark.parametrize(
    "eff_pos,expected_dist",
    [
        (torch.tensor([2.0, 0.0, 5.0]), 2.0),   # 2 units off in x
        (torch.tensor([0.0, 3.0, 5.0]), 3.0),   # 3 units off in y
        (torch.tensor([0.0, 0.0, 5.0]), 0.0),   # On axis
        (torch.tensor([-4.0, 0.0, 1.0]), 4.0),  # 4 units off in -x
        (torch.tensor([0.0, -2.0, -1.0]), 2.0), # 2 units off in -y
    ]
)
def test_dist_eff_to_target_z_axis_off_axis(eff_pos, expected_dist):
    target_pos = torch.tensor([0.0, 0.0, 0.0])
    target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    assert torch.isclose(dist, torch.tensor(expected_dist), atol=1e-6)

# Target position at (1, 2, 3), Z-axis aligned with world Z. No rotation.
@pytest.mark.parametrize(
    "eff_pos,expected_dist",
    [
        (torch.tensor([1.0, 2.0, 8.0]), 0.0),   # On z-axis, above target
        (torch.tensor([1.0, 2.0, -2.0]), 0.0),  # On z-axis, below target
        (torch.tensor([1.0, 2.0, 3.0]), 0.0),   # At target position
        (torch.tensor([2.0, 2.0, 8.0]), 1.0),   # 1 unit off in x
        (torch.tensor([1.0, 3.0, 8.0]), 1.0),   # 1 unit off in y
    ]
)
def test_dist_eff_to_target_z_axis_on_axis(eff_pos, expected_dist):
    target_pos = torch.tensor([1.0, 2.0, 3.0])
    target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    assert torch.isclose(dist, torch.tensor(expected_dist), atol=1e-6)

# Rotate 90-deg around X
@pytest.mark.parametrize(
    "eff_pos,expected_dist",
    [
        (torch.tensor([0.0, 0.0, 2.0]), 2.0),   # 2 units off in z (z-axis rotated to y)
        (torch.tensor([0.0, 0.0, -3.0]), 3.0),  # 3 units off in -z
        (torch.tensor([0.0, 0.0, 0.0]), 0.0),   # On axis
        (torch.tensor([0.0, 5.0, 0.0]), 0.0),   # On axis, different y
    ]
)
def test_dist_eff_to_target_z_axis_rotated_axis_x(eff_pos, expected_dist):
    # Rotate z-axis to y-axis (90 deg about x)
    target_pos = torch.tensor([0.0, 0.0, 0.0])
    rot = R.from_euler('x', 90, degrees=True)
    target_quat = torch.tensor(rot.as_quat())
    dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    assert torch.isclose(dist, torch.tensor(expected_dist), atol=1e-6)

# Rotate 90-deg around Y
@pytest.mark.parametrize(
    "eff_pos,expected_dist",
    [
        (torch.tensor([0.0, 2.0, 0.0]), 2.0),   # 2 units off in y (z-axis rotated to x)
        (torch.tensor([0.0, -3.0, 0.0]), 3.0),  # 3 units off in -y
        (torch.tensor([0.0, 0.0, 0.0]), 0.0),   # On axis
        (torch.tensor([0.0, 0.0, 5.0]), 5.0),   # On axis, different z
    ]
)
def test_dist_eff_to_target_z_axis_rotated_axis_y(eff_pos, expected_dist):
    # Rotate z-axis to x-axis (90 deg about y)
    target_pos = torch.tensor([0.0, 0.0, 0.0])
    rot = R.from_euler('y', 90, degrees=True)
    target_quat = torch.tensor(rot.as_quat())
    dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    assert torch.isclose(dist, torch.tensor(expected_dist), atol=1e-6)


# Rotate 90-deg around Z
@pytest.mark.parametrize(
    "eff_pos,expected_dist",
    [
        (torch.tensor([2.0, 0.0, 0.0]), 2.0),   # 2 units off in x (z-axis rotated in xy)
        (torch.tensor([-3.0, 0.0, 0.0]), 3.0),  # 3 units off in -x
        (torch.tensor([0.0, 0.0, 0.0]), 0.0),   # On axis
        (torch.tensor([0.0, 0.0, 5.0]), 0.0),   # On axis, different z
    ]
)
def test_dist_eff_to_target_z_axis_rotated_axis_z(eff_pos, expected_dist):
    # Rotate z-axis in xy-plane (90 deg about z)
    target_pos = torch.tensor([0.0, 0.0, 0.0])
    rot = R.from_euler('z', 90, degrees=True)
    target_quat = torch.tensor(rot.as_quat())
    dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    assert torch.isclose(dist, torch.tensor(expected_dist), atol=1e-6)

# Invalid shapes
def test_dist_eff_to_target_z_axis_invalid_shapes():
    target_pos = torch.tensor([0.0, 0.0, 0.0])
    target_quat = torch.tensor([0.0, 0.0, 0.0])  # Invalid shape
    eff_pos = torch.tensor([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
    target_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    eff_pos = torch.tensor([0.0, 0.0])  # Invalid shape
    with pytest.raises(ValueError):
        dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))