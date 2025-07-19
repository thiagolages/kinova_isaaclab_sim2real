# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# from isaaclab.envs.rewards import rewards as isaaclab_rewards
import gen3.tasks.manager_based.gen3_skimmer.mdp as mdp_skimmer
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import RewardTermCfg as RewTerm

##
# Pre-defined configs
##
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


##
# Environment configuration
##

from isaaclab.envs import ManagerBasedRLEnv
    
def check_singularity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    eff_name: str,
    threshold: float = 0.01
    ):
    """
    Terminates the episode if the manipulability is below a threshold (i.e., near singularity).
    """
    # Compute manipulability for all envs
    reward = mdp_skimmer.manipulability_reward(env, asset_cfg, eff_name)  # (num_envs,)
    # Find envs where manipulability is below threshold
    done_mask = reward < threshold
    return done_mask

@configclass
class Gen3SkimmerEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        eff_link = "end_effector_link"
        ee_pose_cmd = "ee_pose"
        scene_entity_cfg = SceneEntityCfg("robot", body_names=eff_link)

        # 0. Set the scene spacing
        self.scene.env_spacing = 1.5 # Spacing between environments in the scene
        
        # 1. Switch robot to Kinova Gen3 N7
        self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
        self.scene.table.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.table.init_state.rot = (0.0, 0.0, 0.0, 1.0) # wxyz 
        
        # 2. Override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        
        # 3. Override rewards
        # 3.1. Fill in the body names for end-effector tracking
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [eff_link]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [eff_link]

        # self.rewards.end_effector_orientation_tracking_fine_grained = RewTerm(
        #     func=mdp_skimmer.end_effector_orientation_tracking_tanh,
        #     weight=0.1, # POSITIVE
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[eff_link]), 
        #         "command_name": "ee_pose",
        #         "std": 0.1,  # Standard deviation for normalization
        #     },
        # )
        #self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [eff_link]
        
        # Cone and sphere penalty params
        sphere_penalty_r = 0.15 # 10cm
        cone_penalty_h = 0.20 * 5 # 20cm
        cone_penalty_r = 0.10 * 5 # 10cm

        ## Orientation tracking params
        orientation_tracking_sin_weight = -0.2 # negative
        orientation_tracking_sin_dist_thresh = 0.15 # 15cm

        ## Position tracking params
        self.rewards.end_effector_position_tracking.weight = -0.8 # negative
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1e-5 #0.4 # positive !
        
        # Action rate params
        self.rewards.action_rate.weight = -0.005 # negative
        
        # Joint velocity params
        self.rewards.joint_vel.weight = -0.005 # negative

        self.rewards.end_effector_orientation_tracking = RewardTermCfg(
            func=mdp_skimmer.orientation_tracking_sin,
            weight=orientation_tracking_sin_weight,  # negative
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "distance_threshold": orientation_tracking_sin_dist_thresh,
            }
        )

        # 3.2. Add cone penalty for collision avoidance


        self.rewards.cone_penalty = RewardTermCfg(
            func=mdp_skimmer.cone_penalty,
            weight=-15,
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "cone_h": cone_penalty_h,
                "cone_r": cone_penalty_r,
                "delta_r": 1.0,
                "delta_h": 1.0,
            }
        )

        # 3.3. Add sphere penalty for collision avoidance
        self.rewards.sphere_penalty = RewardTermCfg(
            func=mdp_skimmer.sphere_penalty,
            weight=1, # positive
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "sphere_r": sphere_penalty_r,
            }
        )

        # ERRO DE std >= 0.0 ESTÁ AQUI
        # self.rewards.manipulability = RewardTermCfg(
        #     func=mdp_skimmer.manipulability_reward,
        #     weight=1,
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #         "eff_name": eff_link,
        #     }
        # )      
        # 
        #  # Add event to terminate episode on singularity
        # self.terminations.singularity = TerminationTermCfg(
        #     func=check_singularity,
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #         "eff_name": eff_link,
        #         "threshold": 0.01,  # Threshold for manipulability
        #     }
        # ) 

        # Add event to terminate episode when close to target
        # self.terminations.reached_target = TerminationTermCfg(
        #     func=mdp_skimmer.reached_target,
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #         "command_name": ee_pose_cmd,
        #         "pos_threshold": 0.01,
        #         "cos_theta_threshold": 0.26, # 0.26 represents ~cos(75deg), 15deg tolerance
        #     }
        # ) 

        # self.rewards.joint_limits = RewardTermCfg(
        #     func=mdp_skimmer.joint_limit_penalty,
        #     weight=-0.1,
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #     }
        # )
        
        # Note: Cone visualization will be handled dynamically in the environment
        # to follow the target position for each environment

        # 4. Override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        
        # 5. Override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = eff_link
        
        # Makes the Reach env sample from these command ranges (in 'UniformPoseCommandCfg')
        self.commands.ee_pose.ranges.pos_x = (0.2, 0.8)
        self.commands.ee_pose.ranges.pos_y = (-0.40, 0.40)
        self.commands.ee_pose.ranges.pos_z = (0.20, 1.0)
        self.commands.ee_pose.ranges.roll = (math.radians(-20), math.radians(20)) # Rotates a bit around the X axis
        self.commands.ee_pose.ranges.pitch = (math.radians(-20), math.radians(20)) # Keeps Z axis up  with some tilt
        self.commands.ee_pose.ranges.yaw = (-math.pi, math.pi) # Rotate around Z axis

        # Apply the properties to the goal and current pose visualizers
        # self.commands.ee_pose.goal_pose_visualizer_cfg.markers["frame"].usd_path = "/workspace/kinova_isaaclab_sim2real/arrow_z.usd"
        # self.commands.ee_pose.current_pose_visualizer_cfg.markers["frame"].usd_path = "/workspace/kinova_isaaclab_sim2real/arrow_z.usd"
        # # Set the overall scale to 1.0 since we are controlling individual parts
        # self.commands.ee_pose.goal_pose_visualizer_cfg.markers["frame"].scale = (0.025, 0.025, 0.15) # Z axis 10x bigger
        # self.commands.ee_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.025, 0.025, 0.20) # Z axis 10x bigger
