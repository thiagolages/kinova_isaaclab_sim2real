# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
import gen3.tasks.manager_based.gen3_skimmer.mdp as mdp_skimmer
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
# from isaaclab.scene import RigidObjectCfg, InitialStateCfg, UsdPropertiesCfg
# from isaaclab.sim.spawners.materials.materials_cfg import VisualMaterialCfg # For visual appearance

##
# Pre-defined configs
##
from isaaclab_assets import KINOVA_GEN3_N7_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class Gen3SkimmerEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        eff_link = "end_effector_link"
        ee_pose_cmd = "ee_pose"
        scene_entity_cfg = SceneEntityCfg("robot", body_names=eff_link)

        # 1. Switch robot to Kinova Gen3 N7
        self.scene.robot = KINOVA_GEN3_N7_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 2. Override events
        self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        
        # 3. Override rewards
        # 3.1. Fill in the body names for end-effector tracking
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [eff_link]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [eff_link]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [eff_link]
        
        # 3.2. Add cone penalty for collision avoidance
        # This means that at `cone_h` height from the target, the cone base has radius `cone_r`
        cone_h = 0.50 # 50cm
        cone_r = 0.15 # 15cm

        self.rewards.cone_penalty = RewTerm(
            func=mdp_skimmer.cone_penalty,
            weight=-5.0,
            params={
                "asset_cfg": scene_entity_cfg,
                "command_name": ee_pose_cmd,
                "cone_h": cone_h,
                "cone_r": cone_r,
                "delta_r": 1.0,
                "delta_h": 1.0,
            }
        )

        # # Add a visualization cone to the scene
        # # The USD Cone primitive has its origin at the center of its base.
        # # We'll place this base at a fixed position for visualization.
        # self.scene.penalty_cone_visualizer = RigidObjectCfg(
        #     prim_path="{ENV_REGEX_NS}/PenaltyConeVisual", # Unique prim path
        #     prim_type="Cone", # Specify the primitive type
        #     init_state=InitialStateCfg(
        #         pos=(0.5, 0.0, 0.0),  # Position of the cone's base center in world frame (e.g., in front of robot)
        #         rot=(1.0, 0.0, 0.0, 0.0)  # Orientation (w,x,y,z quaternion), identity for now
        #     ),
        #     usd_props=UsdPropertiesCfg(
        #         prop_double={
        #             "height": cone_h, # Height of the cone
        #             "radius": cone_r, # Radius of the cone's base
        #         },
        #         prop_token={
        #             "axis": "Z"  # Cone extends along its local Z-axis
        #         }
        #     ),
        #     visual_material=VisualMaterialCfg(
        #         prim_path="{ENV_REGEX_NS}/PenaltyConeVisual/Looks/Material", # Path for the new material
        #         diffuse_color=(0.0, 0.8, 0.0),  # Green color
        #         opacity=0.4  # Semi-transparent
        #     ),
        #     collision=False, # No collision for visualization
        #     physics_material=None # No physics properties
        # )

        # 4. Override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        
        # 5. Override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = eff_link
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
