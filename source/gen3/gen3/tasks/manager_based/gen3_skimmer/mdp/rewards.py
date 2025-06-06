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
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import dist_eff_to_target_xy_plane, dist_eff_to_target_z_axis

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


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
    asset: Articulation = env.scene[asset_cfg.name]
    # Obtained from isaaclab_tasks.manager_based.manipulation.reach.mdp
    # function called 'position_command_error'
    command = env.command_manager.get_command(command_name)
    
    ## 1.2. Get target pose
    target_pos_b = command[:, :3] # (num_envs, 3)
    target_pos_w, target_quat_w = combine_frame_transforms(
        asset.data.root_state_w[:, :3],
        asset.data.root_state_w[:, 3:7],
        target_pos_b
    ) # (num_envs, 3)
    
    ## 1.3. Get effector pose
    # asset_cfg.body_ids[0] should be the index of the end-effector link
    eff_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # (num_envs, 3) # type: ignore

    ### 2. Calculate cone's parameters

    ## 2.1. Calculate theta = 2 * atan(r/h)
    theta = 2 * math.degrees(math.atan2(cone_r, cone_h))
    
    ## 2.2. Compute the effector height with respect to the target XY plane
    eff_cone_height = dist_eff_to_target_xy_plane(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    
    ## 2.3. Compute the effector distance to the cone Z axis
    dist_to_cone_axis = dist_eff_to_target_z_axis(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    
    ## 2.4. Calculate the radius at the current height as r = h * tan(theta/2)
    r_at_current_h = torch.maximum(
        eff_cone_height * math.tan(math.radians(theta / 2.0)),
        torch.tensor(1e-3, device=eff_cone_height.device, dtype=eff_cone_height.dtype)
    ) # (num_envs,)

    ### 3. Define multipliers

    ## 3.1. delta_r scales penalty by how close the effector is to the center (r = 0), not normalized
    # Normalizes delta_r based on curent radius, so that this penalty doesn't diminish as we
    # approach the target.
    delta_r = delta_r / r_at_current_h # (num_envs,)

    ## 3.2. delta_h scales penalty by how close we are from the target, in the Z direction of the cone (its height)
    delta_h = delta_h * (cone_h - eff_cone_height) / cone_h # (num_envs,)
    # TODO: check if needs to be normalized

    ### 4. Check if effector is inside penalty cone
    # Don't penalize if outside cone max radius or max height
    # eff_cone_height: (num_envs,), cone_h: scalar
    mask = (dist_to_cone_axis > r_at_current_h) | ((eff_cone_height > cone_h) & (eff_cone_height > 0))
    r_error = torch.where(mask, torch.zeros_like(dist_to_cone_axis), r_at_current_h - dist_to_cone_axis)
    # if dist_to_cone_axis > r_at_current_h or eff_cone_height > cone_h:
    #     r_error = 0
    # else:
    #     r_error = r_at_current_h - dist_to_cone_axis # (num_envs,)
    
    ### 4. Define final reward
    # This reward will be multiplied by the `weight` (some negative number)
    rew = delta_r * delta_h * r_error

    # compute the reward
    result = torch.sum(rew, dim=1)

    print("cone_penalty result = ", result)

    # Check for NaNs
    if torch.isnan(result).any():
        raise ValueError("Reward contains NaN values.")
    # Check for unwanted shape (should be 1D: (num_envs,))
    if result.ndim != 1:
        raise ValueError(f"Reward has unexpected shape: {result.shape}")
    return result

