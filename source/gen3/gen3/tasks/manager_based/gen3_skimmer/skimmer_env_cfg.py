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
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

##
# Pre-defined configs
##
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import SceneEntityCfg
from gen3.tasks.manager_based.gen3_skimmer.robots.kinova_custom import THIAGO_KINOVA_GEN3_N7_CFG
##
# Environment configuration
##

from isaaclab.envs import ManagerBasedRLEnv

SKIMMER_USD_PATH = "/workspace/research/assets/usd/skimmer.usd"
    
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
    # Find envs where manipulability is below wthreshold
    done_mask = reward < threshold
    return done_mask

@configclass
class Gen3SkimmerEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        eff_link = "end_effector_link" #"skimmer_tool"
        ee_pose_cmd = "ee_pose"
        scene_entity_cfg = SceneEntityCfg("robot", body_names=eff_link)

        # 0. Set the scene spacing
        self.scene.env_spacing = 1.5 # Spacing between environments in the scene
        # Get ASSETS_PATH from environment variable or set default
        
        # 1. Switch robot to Kinova Gen3 N7
        # self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = THIAGO_KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        print(f"THIAGO_KINOVA_GEN3_N7_CFG: {THIAGO_KINOVA_GEN3_N7_CFG}")
        print(f"self.scene.robot: {self.scene.robot}")

        self.scene.table.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
        self.scene.table.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.table.init_state.rot = (0.0, 0.0, 0.0, 1.0) # wxyz 

        # ------------------------------------------------------------------
        # 2.a Add the skimmer *tool* mesh under the gripper link
        #     We create it as a kinematic rigid object so that physics is not
        #     affected, but the visual is updated every frame automatically by
        #     USD because it is a child prim of the link.  The tool frame is
        #     rotated +90° about Y so that its Z-axis points *forward* in the
        #     gripper frame; the gripper itself is oriented such that its X is
        #     forward, therefore we apply the offset quaternion
        #     q_off = [cos(π/4), 0, sin(π/4), 0].  This matches the transform
        #     assumption used in the reward function changes.
        # ------------------------------------------------------------------

        # self.scene.tool = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/end_effector_link/skimmer_tool",
        #     spawn=sim_utils.UsdFileCfg(
        #         usd_path=SKIMMER_USD_PATH,
        #         scale=(1.0, 1.0, 1.0),
        #         visual_material=sim_utils.PreviewSurfaceCfg(opacity=1.0),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(
        #         pos=(0.0, 0.0, 0.0),
        #         rot=(0.70710678, 0.0, 0.70710678, 0.0),  # +90° about Y
        #     ),
        # )
        
        # 2.b Override events
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
        
        # Sphere penalty params
        sphere_penalty_r = 0.15 # 15cm
        sphere_penalty_weight = 5.0  # positive
        
        # Cone penalty params
        cone_penalty_h = 0.20 # 20cm
        cone_penalty_r = 0.10 # 10cm
        cone_penalty_weight = -150 # negative

        # Action rate params
        self.rewards.action_rate.weight = -0.005 # negative
        
        # Joint velocity params
        self.rewards.joint_vel.weight = -0.005 # negative

        ## Orientation tracking params
        orientation_tracking_sin_dist_thresh = 0.10 # 10cm
        orientation_tracking_sin_weight = -0.50 # negative

        ## Position tracking params
        self.rewards.end_effector_position_tracking_fine_grained.weight = 0.25 #0.4 # positive !
        self.rewards.end_effector_position_tracking.weight = -0.75 # negative
        

        # ALIGN ONLY Z AXIS, WITHOUT END EFFECTOR TRANSFORM
        # self.rewards.end_effector_orientation_tracking = RewardTermCfg(
        #     func=mdp_skimmer.orientation_tracking_sin,
        #     weight=orientation_tracking_sin_weight,  # negative
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #         "command_name": ee_pose_cmd,
        #         "distance_threshold": orientation_tracking_sin_dist_thresh,
        #     }
        # )

        orientation_tracking_eff_transform_z_target_dist_thresh = 0.10 # 10cm
        orientation_tracking_eff_transform_z_target_weight = -0.50 # negative
        
        self.rewards.end_effector_orientation_tracking = RewardTermCfg(
            func=mdp_skimmer.orientation_tracking_eff_transform_z_target,
            weight=orientation_tracking_eff_transform_z_target_weight,  # negative
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "distance_threshold": orientation_tracking_eff_transform_z_target_dist_thresh,
            }
        )

        # 3.2. Add cone penalty for collision avoidance


        self.rewards.cone_penalty = RewardTermCfg(
            func=mdp_skimmer.cone_penalty,
            weight=cone_penalty_weight,
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
            weight=sphere_penalty_weight,
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "sphere_r": sphere_penalty_r,
            }
        )

         # Add event to terminate episode if touching spherical shell from top
        # self.terminations.collision_from_top = TerminationTermCfg(
        #     func=mdp_skimmer.collision_from_top,
        #     params={
        #         "asset_cfg": scene_entity_cfg,
        #         "command_name": ee_pose_cmd,
        #         "sphere_r": 0.10, # 12cm
        #         # height of the plane as a function of the sphere radius
        #         # 0.0 means the plane is at the bottom of the sphere
        #         # 0.5 means the plane is at the middle of the sphere
        #         # 1.0 means the plane is at the top of the sphere
        #         "plane_h_percentage": 0.7,
        #     }
        # ) 

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


        # viz_cfg = self.commands.ee_pose
        # print("viz_cfg.goal_pose_visualizer_cfg: ", viz_cfg.goal_pose_visualizer_cfg)
        # print("viz_cfg.current_pose_visualizer_cfg: ", viz_cfg.current_pose_visualizer_cfg)
        # print("viz_cfg.goal_pose_visualizer_cfg.markers: ", viz_cfg.goal_pose_visualizer_cfg.markers)
        # print("viz_cfg.current_pose_visualizer_cfg.markers: ", viz_cfg.current_pose_visualizer_cfg.markers)
        # print("viz_cfg.goal_pose_visualizer_cfg.markers['frame']: ", viz_cfg.goal_pose_visualizer_cfg.markers["frame"])
        # print("viz_cfg.current_pose_visualizer_cfg.markers['frame']: ", viz_cfg.current_pose_visualizer_cfg.markers["frame"])

        # # make Z-axis long & visible, hide X/Y
        # thin_axes_props = sim_utils.UsdPropertiesCfg(
        #     prop_double={
        #         # X-axis
        #         "Frame/X_line.radius": 0.01,
        #         "Frame/X_tip.radius": 0.05,
        #         # Y-axis
        #         "Frame/Y_line.radius": 0.01,
        #         "Frame/Y_tip.radius": 0.05,
        #         # Z-axis
        #         "Frame/Z_line.radius": 0.01,
        #         "Frame/Z_tip.radius": 0.05,
        #     }
        # )
        # exit()
        # viz_cfg.goal_pose_visualizer_cfg.markers["frame"].usd_props    = thin_axes_props
        # viz_cfg.current_pose_visualizer_cfg.markers["frame"].usd_props = thin_axes_props