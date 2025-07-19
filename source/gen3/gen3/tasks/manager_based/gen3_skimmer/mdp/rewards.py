# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.assets import RigidObject
from isaaclab.utils.math import combine_frame_transforms, quat_mul, angle_between_vecs
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import dist_eff_to_target_xy_plane, dist_eff_to_target_z_axis

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(robot.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)

def get_z_axis(quat, effector: bool = True) -> torch.Tensor:
    if effector:
        eff_quat_w = quat
        eff_quat_w_xyzw = torch.cat([eff_quat_w[:, 1:], eff_quat_w[:, :1]], dim=1).cpu().numpy()  # (num_envs, 4)
        eff_quat_w_xyzw_rot_mat = R.from_quat(eff_quat_w_xyzw)
        eff_rotmat = torch.from_numpy(eff_quat_w_xyzw_rot_mat.as_matrix()).to(eff_quat_w.device).type(eff_quat_w.dtype)  # (num_envs, 3, 3)
        z_axis = eff_rotmat[:, :, 2]  # (num_envs, 3)
        # print("eff_quat_w[0]:", eff_quat_w[0])
        # print("eff_quat_w_xyzw[0]:", eff_quat_w_xyzw[0])
        # print("eff_quat_w_xyzw_rot_mat[0]:\n", eff_quat_w_xyzw_rot_mat[0].as_matrix())
        # print("eff_rotmat[0]:", eff_rotmat[0])
        # print("eff_z_axis[0]:", z_axis[0])
    else:
        des_quat_w = quat
        des_quat_w_xyzw = torch.cat([des_quat_w[:, 1:], des_quat_w[:, :1]], dim=1).cpu().numpy()  # (num_envs, 4)
        des_quat_w_xyzw_rot_mat = R.from_quat(des_quat_w_xyzw)
        target_rotmat = torch.from_numpy(des_quat_w_xyzw_rot_mat.as_matrix()).to(des_quat_w.device).type(des_quat_w.dtype)  # (num_envs, 3, 3)
        z_axis = target_rotmat[:, :, 2]  # (num_envs, 3)
        # print("des_quat_w[0]:", des_quat_w[0])
        # print("des_quat_w_xyzw[0]:", des_quat_w_xyzw[0])
        # print("des_quat_w_xyzw_rot_mat[0]:\n", des_quat_w_xyzw_rot_mat[0].as_matrix())
        # print("target_rotmat[0]:", target_rotmat[0])
        # print("target_z[0]:", z_axis[0])
    
    return z_axis

def orientation_tracking_sin(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    command_name: str,
    distance_threshold: float = 0.2,  # default threshold, can be overridden
) -> torch.Tensor:
    """
    Penalize the end-effector orientation deviation from Z pointing up using sin(theta) of the angle,
    but only if the end-effector is within a distance threshold from the target. Otherwise, reward is zero.
    """
    # Get the robot articulation
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Get the end-effector orientation in world frame
    eff_quat_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # (num_envs, 4), (w, x, y, z)
    eff_z_axis = get_z_axis(eff_quat_w, effector=True)

    # Target Z
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    target_z = get_z_axis(des_quat_w, effector=False)

    theta_deg = angle_between_vecs(eff_z_axis, target_z)

    sin_theta = torch.sin(torch.deg2rad(theta_deg))  # Convert degrees to radians and compute sin(theta)
    reward = 1.0 - sin_theta  # Return sin(theta) as the reward

    # --- Compute distance between end-effector and target ---
    # Get end-effector position in world frame
    eff_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # (num_envs, 3)
    # Get robot base pose
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7]
    # Get target pose in robot base frame
    target_pos_b = command[:, :3]
    target_quat_b = command[:, 3:7]
    # Convert target pose to world frame
    target_pos_w, _ = combine_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        target_pos_b,
        target_quat_b
    )  # (num_envs, 3)
    # Compute Euclidean distance
    dist = torch.linalg.norm(eff_pos_w - target_pos_w, dim=1)  # (num_envs,)

    # Only apply orientation penalty if within threshold, else reward is zero
    mask = dist <= distance_threshold
    reward = reward * mask.float()

    if torch.isnan(reward).any() or torch.isinf(reward).any():
        raise RuntimeError("ERROR: NaN or Inf detected in orientation_tracking_sin reward!")

    return reward


def end_effector_orientation_tracking_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float = 0.1,  # Standard deviation for normalization
) -> torch.Tensor:
    """Penalize the end-effector orientation deviation from Z pointing up."""

    reward = orientation_tracking_sin(env, asset_cfg, command_name)

    return 1.0 - torch.tanh(reward / std)  # Normalize the reward with a tanh kernel

def sphere_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sphere_r: float = None,
) -> torch.Tensor:
    """Penalize the end-effector for being outside the target area."""
    # Check that sphere_r is provided
    if sphere_r is None or sphere_r < 1e-2: # min 1cm
        raise ValueError("sphere_r must be provided and >= 1cm.")

    ### 1. Get target and effector poses
    
    ## 1.1. Get robot articulation and command
    robot: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    ## Get robot pose
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7] # (w, x, y, z) format

    ## 1.2. Get target pose
    target_pos_b = command[:, :3] # (num_envs, 3)
    target_quat_b = command[:, 3:7] # (num_envs, 4)

    # Convert target pose to world frame
    target_pos_w, target_quat_w = combine_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        target_pos_b,
        target_quat_b
    ) # (num_envs, 3)

    ## 1.3. Get effector pose
    # asset_cfg.body_ids[0] should be the index of the end-effector link
    eff_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3] # (num_envs, 3) # type: ignore

    rot = R.from_quat(target_quat_w.cpu().numpy())
    offset = torch.tensor([0.0, 0.0, sphere_r])
    offset_w = rot.apply(offset)
    sphere_center_w = target_pos_w + torch.from_numpy(offset_w).to(target_pos_w.device).type(target_pos_w.dtype)

    
    ## 1.4. Calculate distance between effector and target
    dist_eff_to_sphere_center = torch.norm(eff_pos_w - sphere_center_w, dim=1)  # (num_envs,)
    # print("dist_eff_to_sphere_center = ", dist_eff_to_sphere_center)
    # print("dist_eff_to_sphere_center.shape = ", dist_eff_to_sphere_center.shape)
    
    ## 1.5. Calculate penalty
    penalty = torch.minimum((dist_eff_to_sphere_center - sphere_r) / sphere_r, torch.zeros_like(dist_eff_to_sphere_center))
    
    # print("sphere penalty = ", penalty)
    # print("sphere penalty.shape = ", penalty.shape)
    
    ## 1.6. Check for NaNs
    if torch.isnan(penalty).any() or torch.isinf(penalty).any():
        raise RuntimeError("ERROR: NaN or Inf detected in sphere penalty!")
    
    ## 1.7. Check for unwanted shape (should be 1D: (num_envs,))
    
    ## 1.6. Return penalty
    return penalty
    
    

def cone_penalty(env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        cone_r: float = None,
        cone_h: float = None,
        delta_r: float = None,
        delta_h: float = None,
    ) -> torch.Tensor:
    """Penalize the end-effector for being outside the target area."""
    # Check that cone_r and cone_h are provided
    if cone_r is None or cone_h is None or cone_r <= 0 or cone_h <= 0:
        raise ValueError("cone_r and cone_h must be provided.")

    # Check that delta_r and delta_h are positive
    if delta_r is None or delta_h is None or delta_r <= 0 or delta_h <= 0:
        raise ValueError("delta_r and delta_h must be provided, and positive.")

    ### 1. Get target and effector poses
    
    ## 1.1. Get robot articulation and command
    robot: Articulation = env.scene[asset_cfg.name]
    # Obtained from isaaclab_tasks.manager_based.manipulation.reach.mdp
    # function called 'position_command_error'
    command = env.command_manager.get_command(command_name)
    
    
    ## Get robot pose
    robot_pos_w = robot.data.root_state_w[:, :3]
    robot_quat_w = robot.data.root_state_w[:, 3:7] # (w, x, y, z) format

    ## 1.2. Get target pose
    target_pos_b = command[:, :3] # (num_envs, 3)
    target_quat_b = command[:, 3:7] # (num_envs, 4)

    # Convert target pose to world frame
    target_pos_w, target_quat_w = combine_frame_transforms(
        robot_pos_w,
        robot_quat_w,
        target_pos_b,
        target_quat_b
    ) # (num_envs, 3)
    
    ## 1.3. Get effector pose
    # asset_cfg.body_ids[0] should be the index of the end-effector link
    eff_pos_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], :3] # (num_envs, 3) # type: ignore


    #print("Current effector position (eff_pos_w):", eff_pos_w)
    #print("Target position (target_pos_w):", target_pos_w)

    ### 2. Calculate cone's parameters

    ## 2.1. Calculate theta = 2 * atan(r/h)
    theta = 2 * math.degrees(math.atan2(cone_r, cone_h))
    #print("theta = ", theta)
    
    ## 2.2. Compute the effector height with respect to the target XY plane
    eff_cone_height = dist_eff_to_target_xy_plane(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    #print("eff_cone_height = ", eff_cone_height)
    #print("eff_cone_height.shape = ", eff_cone_height.shape)

    ## 2.3. Compute the effector distance to the cone Z axis
    dist_to_cone_axis = dist_eff_to_target_z_axis(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    #print("dist_to_cone_axis = ", dist_to_cone_axis) 
    #print("dist_to_cone_axis.shape = ", dist_to_cone_axis.shape)

    ## 2.4. Calculate the radius at the current height as r = h * tan(theta/2)
    r_at_current_h = torch.maximum(
        eff_cone_height * math.tan(math.radians(theta / 2.0)),
        torch.tensor(1e-5, device=eff_cone_height.device, dtype=eff_cone_height.dtype)
    ) # (num_envs,)
    #print("r_at_current_h = ", r_at_current_h)
    #print("r_at_current_h.shape = ", r_at_current_h.shape)

    ### 3. Define multipliers

    ## 3.1. delta_r scales penalty by how close the effector is to the center (r = 0), not normalized
    # Normalizes delta_r based on curent radius, so that this penalty doesn't diminish as we
    # approach the target.
    delta_r = torch.clamp(delta_r / r_at_current_h, 0.0, 10.0) # (num_envs,)

    #print("delta_r = ", delta_r)
    #print("delta_r.shape = ", delta_r.shape)

    ## 3.2. delta_h scales penalty by how close we are from the target, in the Z direction of the cone (its height)
    delta_h = torch.clamp(delta_h * (cone_h - eff_cone_height) / cone_h, 0.0, 10.0) # (num_envs,)
    #print("delta_h = ", delta_h)
    #print("delta_h.shape = ", delta_h.shape)
    # TODO: check if needs to be normalized

    ### 4. Check if effector is inside penalty cone
    # Don't penalize if outside cone max radius or max height
    # eff_cone_height: (num_envs,), cone_h: scalar

    # small value to don't penalize too much if outside the cone, 
    # but still make it so it's not a sparse reward
    eps_error = (1e-5 + torch.rand(1).item() * 4e-5)

    mask = (dist_to_cone_axis > r_at_current_h) | (eff_cone_height > cone_h) | (eff_cone_height < 0)
    #print("mask = ", mask)
    #print("mask.shape = ", mask.shape)
    r_error = torch.where(mask, torch.ones_like(dist_to_cone_axis) * eps_error, r_at_current_h - dist_to_cone_axis)
    #print("r_error = ", r_error)
    #print("r_error.shape = ", r_error.shape)
    # if dist_to_cone_axis > r_at_current_h or eff_cone_height > cone_h:
    #     r_error = 0
    # else:
    #     r_error = r_at_current_h - dist_to_cone_axis # (num_envs,)
    
    ### 4. Define final penalty
    # This penalty will be multiplied by the `weight` (some negative number)
    
    penalty = delta_r * delta_h * r_error
    # nonzero_percentage = (penalty != 0).float().mean() * 100
    #print(f"Percentage of nonzero penalty elements: {nonzero_percentage.item():.2f}%")
    # print(f"penalty for cone = {penalty}")
    # print("penalty.shape = ", penalty.shape)
    # print("NaNs/Infs in cone penalty {}/{}".format(torch.isnan(penalty).sum().item(), torch.isinf(penalty).sum().item()))
    # print("Penalty average:", penalty.mean().item())
    # print("Top 10 penalty values:", torch.topk(penalty, 10).values)

    # Check for NaNs
    if torch.isnan(penalty).any() or torch.isinf(penalty).any():
        raise RuntimeError("ERROR: NaN or Inf detected in penalty term output!")
    # Check for unwanted shape (should be 1D: (num_envs,))
    if penalty.ndim != 1:
        raise ValueError(f"Reward has unexpected shape: {penalty.shape}")
    
    return penalty

# def manipulability_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     eff_name: str,
# ) -> torch.Tensor:
#     """
#     Compute the manipulability reward based on the end-effector pose and the target pose.
#     The reward is scaled by the manipulability scale factor.
#     """
#     robot: Articulation = env.scene[asset_cfg.name]
#     # command = env.command_manager.get_command(command_name)

#     # Get the jacobian
#     geometric_jacobians = robot.root_physx_view.get_jacobians() #(num_envs, num_bodies, 6, total_dofs)

#     body_ids, _ = robot.find_bodies([eff_name], preserve_order=True)
#     ee_body_idx = body_ids[0]          # <- this is the index to use in jacobians
#     arm_joint_ids, _ = robot.find_joints(["^joint_[1-7]$"], preserve_order=True)

#     # print("body_ids = ", body_ids)
#     # print("ee_body_idx = ", ee_body_idx)
#     # print("arm_joint_ids = ", arm_joint_ids)
#     # PhysX index is (body_idx – 1) because it skips the root link
#     J = geometric_jacobians[:, ee_body_idx - 1, :, arm_joint_ids] # (num_envs, 6, num_joints)

#     # yoshikawa_manipulability
#     JJt = J @ J.transpose(-2, -1) # 'change second-to-last dimension with last dimension'

#     # Add small value to JJt to avoid NaNs close to singularities
#     JJt = JJt + 1e-12 * torch.eye(6, device=J.device)

#     det = torch.abs(torch.det(JJt))
#     # Ensure det is not extremely small negative due to precision before abs, then sqrt
#     # though abs should handle it. Clamping to a small positive epsilon before sqrt is safest.
#     manipulability = torch.sqrt(torch.clamp(det, min=1e-24)) # Clamp to a very small positive before sqrt
#     # print("JJt = ", JJt)  # Added print statement for debugging
#     # print("J = ", J)  # Added print statement for debugging
#     # print("J shape = ", J.shape)  # Added print statement for debugging
#     # print("JJt shape = ", JJt.shape)  # Added print statement for debugging
#     # print("det = ", det)  # Added print statement for debugging
#     # print("manipulability reward = ", manipulability)  # Added print statement for debugging
#     print("NaNs/Infs in manipulability {}/{}".format(torch.isnan(manipulability).sum().item(), torch.isinf(manipulability).sum().item()))
    
#     if torch.isnan(manipulability).any() or torch.isinf(manipulability).any():
#         raise RuntimeError("ERROR: NaN or Inf detected in manipulability reward!")

#     return manipulability
    

# def joint_limit_penalty(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
# ) -> torch.Tensor:
#     """
#     Penalize joints for being close to their limits.
#     The penalty increases as the joint approaches its lower or upper limit within the margin.
#     """
#     robot: Articulation = env.scene[asset_cfg.name]
#     joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]  # (num_envs, num_joints)
    
#     if torch.isnan(joint_pos).any() or torch.isinf(joint_pos).any():
#         raise RuntimeError("ERROR: NaN or Inf detected in joint_pos!")

#     # Get the limits from the first environment, since they are equal
#     joint_lower = robot.data.soft_joint_pos_limits[0, :, 0]
#     joint_upper = robot.data.soft_joint_pos_limits[0, :, 1]

#     geometric_avg = torch.sqrt(torch.abs(joint_lower * joint_upper))  # sqrt(|a*b|) for each joint
#     joint_range = torch.clamp(joint_upper - joint_lower, min=1e-6) # (num_joints,)

#     if torch.any(joint_upper == joint_lower):
#         raise ValueError("Some joint limits are equal, cannot compute joint limit penalty.")

#     # Sum over all joints
#     penalty = torch.sum(torch.square((joint_pos - geometric_avg) / joint_range), dim=1)

#     print("NaNs/Infs in joint limit penalty {}/{}".format(torch.isnan(penalty).sum().item(), torch.isinf(penalty).sum().item()))

#     if torch.isnan(penalty).any() or torch.isinf(penalty).any():
#         raise RuntimeError("ERROR: NaN or Inf detected in joint_limit_penalty!")
    