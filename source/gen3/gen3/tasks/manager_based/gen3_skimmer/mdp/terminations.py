import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
from scipy.spatial.transform import Rotation as R

def reached_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    pos_threshold: float = 0.01,
    cos_theta_threshold: float = 0.01,
):
    """Termination condition for reaching the target position and orientation."""
    # Get the robot articulation
    robot: Articulation = env.scene[asset_cfg.name]

    ## POSITION REACHED
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    curr_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    position_reached = distance < pos_threshold

    ## ROTATION REACHED
    # Vector representing Z pointing up in world frame
    target_z = torch.tensor(
        [0.0, 0.0, 1.0], device=env.device
    )

    # Get the end-effector orientation in world frame
    eff_quat_w = robot.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # (num_envs, 4), (w, x, y, z)
    eff_quat_w_xyzw = torch.cat([eff_quat_w[:, 1:], eff_quat_w[:, :1]], dim=1).cpu().numpy()  # (num_envs, 4)
    eff_rotmat = torch.from_numpy(R.from_quat(eff_quat_w_xyzw).as_matrix()).to(eff_quat_w.device).type(eff_quat_w.dtype)  # (num_envs, 3, 3)
    # Extract the last column (Z axis in world frame)
    eff_z_axis = eff_rotmat[:, :, 2]  # (num_envs, 3)
    # Compute the orientation error
    cos_theta = (eff_z_axis * target_z).sum(-1).clamp(-1.0, 1.0)  # (num_envs,)
    
    orientation_reached = cos_theta >= cos_theta_threshold

    return position_reached & orientation_reached