import torch
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.math import quat_apply

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
    if not target_quat.shape[1] == 4: # should be (num_envs, 4)
        raise ValueError("target_quat must be a quaternion (4,)")
    
    if not eff_pos.shape[1] == 3: # should be (num_envs, 3)
        raise ValueError("eff_pos must be a 3D vector (3,)")

    # Convert target_quat from scalar-last (x, y, z, w) to scalar-first (w, x, y, z)
    #print("target_quat before = ", target_quat)
    #target_quat = torch.cat([target_quat[:, 1:4], target_quat[:, 0:1]], dim=1)

    # Assuming target_quat is in scalar-first format (w, x, y, z)

    #print("target_quat after = ", target_quat)
    target_rot = torch.tensor(R.from_quat(target_quat.cpu()).as_matrix(),dtype=torch.float, device=eff_pos.device)  # Convert quaternion to rotation matrix
    #print("target_rot = ", target_rot)
    # Sanity check for rotation matrix
    if target_rot.shape[-2:] != (3, 3):
        raise ValueError("target_rot must have shape (num_envs, 3, 3)")

    dz_unit = target_rot[:, 2]  # Extract the z-axis from the rotation matrix
    #print("dz_unit =", dz_unit)
    #print("eff_pos =", eff_pos)
    #print("target_pos =", target_pos)
    vec = eff_pos - target_pos  # Vector from end effector to target    
    #print("vec = eff_pos - target_pos = ", vec)
    proj_length = torch.sum(vec * dz_unit, dim=1, keepdim=True)  # (N, 1)
    proj_vec = proj_length * dz_unit  # (N, 3)
    #print("proj_vec =", proj_vec)
    perp_vec = vec - proj_vec  # Perpendicular vector from effector to the line
    #print("perp_vec =", perp_vec)
    dist = torch.norm(perp_vec, dim=1, keepdim=True)  # (N, 1) norm vector
    #print("dist =", dist)
    dist = dist.squeeze(-1)

    return dist

def dist_eff_to_target_xy_plane(eff_pos_w, target_w):
    """
    Compute the signed distance from the end effector to the XY plane defined by a target_w position and orientation.
    The plane normal is the Z axis of the target_w's rotation (from target_quat).
    Args:
        eff_pos_w: (num_envs, 3) tensor of end effector positions in world frame.
        target_w: tuple (target_pos, target_quat), where target_pos is (num_envs, 3), target_quat is (num_envs, 4) in scalar-first format.
    Returns:
        Signed distance (num_envs,) from eff_pos_w to the target_w's XY plane (positive if above, negative if below).
    """
    target_pos_w, target_quat_w = target_w  # (num_envs, 3), (num_envs, 4)
    
    # The Z axis in the target_w frame (in world coordinates)
    z_axis = torch.zeros_like(target_pos_w)
    z_axis[:, 2] = 1.0
    
    # Rotate z_axis by target_quat_w to get world normal direction
    # print("target_quat_w = ", target_quat_w)
    # print("z_axis = ", z_axis)
    z_axis_world = quat_apply(target_quat_w, z_axis)  # (num_envs, 3)
    # print("z_axis_world = ", z_axis_world)
    
    # Vector from plane point to effector
    vec = eff_pos_w - target_pos_w  # (num_envs, 3)

    # print("vec = eff_pos_w - target_pos_w = ", vec)
    # print("vec * z_axis_world = ", vec * z_axis_world)
    
    # Signed distance: projection onto normal
    # The * operator is doing a dot product
    # print("vec * z_axis_world = ",vec * z_axis_world)
    signed_dist = torch.sum(vec * z_axis_world, dim=1)
    
    return signed_dist
