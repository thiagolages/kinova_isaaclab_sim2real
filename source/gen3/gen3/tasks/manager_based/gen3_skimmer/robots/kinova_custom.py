import math
import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ASSETS_PATH = os.environ.get("ASSETS_PATH", "/workspace/research/assets")

"""Configuration of Kinova Gen3 (7-Dof) arm with an attached Strainer."""
THIAGO_KINOVA_GEN3_N7_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSETS_PATH}/usd/robot_test.usd", #robot_skimmer.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Define a proper home pose for the Kinova Gen3
            # This pose should be safe, reachable, and provide good workspace access
            "joint_1": 0.0,
            "joint_2": math.radians(-7.1),
            "joint_3": math.radians(10),
            "joint_4": math.radians(110.0),
            "joint_5": math.radians(-19.4),
            "joint_6": math.radians(36.8),
            "joint_7": math.radians(-90.0),
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-7]"],
            velocity_limit=100.0,
            effort_limit={
                "joint_[1-4]": 39.0,
                "joint_[5-7]": 9.0,
            },
            stiffness={
                "joint_[1-4]": 40.0,
                "joint_[5-7]": 15.0,
            },
            damping={
                "joint_[1-4]": 1.0,
                "joint_[5-7]": 0.5,
            },
        ),
    },
)
