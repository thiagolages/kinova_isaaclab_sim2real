import sys
import argparse

from isaaclab.app import AppLauncher

# if __name__ == "__main__" and not any(arg.startswith("unittest") for arg in sys.argv[0:2]):
#     # Only parse arguments and launch app if not running under unittest
#     parser = argparse.ArgumentParser(description="Tutorial on running IsaacSim via the AppLauncher.")
#     parser.add_argument("--size", type=float, default=1.0, help="Side-length of cuboid")
#     parser.add_argument(
#         "--width", type=int, default=1280, help="Width of the viewport and generated images. Defaults to 1280"
#     )
#     parser.add_argument(
#         "--height", type=int, default=720, help="Height of the viewport and generated images. Defaults to 720"
#     )
#     AppLauncher.add_app_launcher_args(parser)
#     args_cli = parser.parse_args()
#     app_launcher = AppLauncher(args_cli)
#     simulation_app = app_launcher.app
# else:
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import torch
import numpy as np
import unittest

# Need to run IsaacSim to import the rewards module
# Launch the IsaacSim application

# Ensure the app is launched before importing rewards
from gen3.tasks.manager_based.gen3_skimmer.mdp import rewards
from gen3.tasks.manager_based.gen3_skimmer.mdp.utils import (
    dist_eff_to_target_xy_plane,
    dist_eff_to_target_z_axis
)

# Dummy classes to simulate environment, robot, and asset configuration
class DummyData:
    def __init__(self, eff_quat_w):
        num_envs = eff_quat_w.shape[0]
        num_bodies = 1
        self.body_state_w = torch.zeros((num_envs, num_bodies, 7), dtype=eff_quat_w.dtype)
        self.body_state_w[:, 0, 3:7] = eff_quat_w

class DummyRobot:
    def __init__(self, eff_quat_w):
        self.data = DummyData(eff_quat_w)

class DummyEnv:
    def __init__(self, eff_quat_w, device="cpu"):
        self.device = device
        self.scene = {"robot": DummyRobot(eff_quat_w)}

class DummyAssetCfg:
    def __init__(self, body_ids=[0], name="robot"):
        self.body_ids = body_ids
        self.name = name

class TestEndEffectorOrientationTracking(unittest.TestCase):
    # Test basic cases for end effector orientation tracking
    def test_end_effector_orientation_tracking_basic(self):
        angles_deg = np.array([0, 30, 45, 60, 90, 120, 135, 150, 180])
        angles_deg = np.concatenate((angles_deg, -1 * angles_deg))  # Include negative angles
        axes = [
            [1, 0, 0],  # X
            [0, 1, 0],  # Y
            [0, 0, 1],  # Z
        ]
        for angle_deg in angles_deg:
            theta = np.deg2rad(angle_deg)
            cos_theta = np.cos(theta)
            expected_reward = -(1 - cos_theta)
            for idx, axis in enumerate(axes):
                axis_str = ["X", "Y", "Z"][idx]
                if axis_str == "Z" or angle_deg == 0:
                    expected_reward = 0.0  # For Z-axis and 0 degrees, the reward should be 0
                axis = np.array(axis)
                axis = axis / np.linalg.norm(axis)
                qw = np.cos(theta / 2)
                qx = axis[0] * np.sin(theta / 2)
                qy = axis[1] * np.sin(theta / 2)
                qz = axis[2] * np.sin(theta / 2)
                quat_wxyz = [qw, qx, qy, qz]
                eff_quat_w = torch.tensor([quat_wxyz], dtype=torch.float32)
                env = DummyEnv(eff_quat_w)
                asset_cfg = DummyAssetCfg()
                reward = rewards.end_effector_orientation_tracking(env, asset_cfg)
                reward *= -1.0  # Negate the reward as per the function definition
                # self.assertEqual(reward.shape, (1,))
                print("Angle: {}({}), Quat: {}, Expected: {}, Computed: {}".format(angle_deg, axis_str, quat_wxyz, expected_reward, reward.item()))
                self.assertTrue(torch.allclose(reward, torch.tensor([expected_reward], dtype=reward.dtype), atol=1e-7))

class TestDistEffToTargetXYPlane(unittest.TestCase):
    def test_zero_distance_on_plane(self):
        # eff_pos exactly on the plane
        eff_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        target_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        # Identity quaternion (no rotation), scalar-first (w, x, y, z)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([0.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_positive_distance_above_plane(self):
        # eff_pos above the plane (along +z)
        eff_pos = torch.tensor([[0.0, 0.0, 5.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([2.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_negative_distance_below_plane(self):
        # eff_pos below the plane (along -z)
        eff_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 3.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([-2.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_rotated_plane_90deg_about_x(self):
        # Plane normal is along +y after 90deg rotation about x
        eff_pos = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        # 90 deg about x: quaternion (w, x, y, z) = (0.7071, 0.7071, 0, 0)
        theta = np.pi / 2
        qw = np.cos(theta/2)
        qx = np.sin(theta/2)
        target_quat = torch.tensor([[qw, qx, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([2.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-5))

    def test_batch_inputs(self):
        # Test with batch of environments
        eff_pos = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_xy_plane(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([2.0, -2.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

class TestDistEffToTargetZAxis(unittest.TestCase):
    def test_zero_distance_on_line(self):
        # eff_pos is on the line defined by target_pos and target z-axis
        eff_pos = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        # Identity quaternion (no rotation), scalar-first (w, x, y, z)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([0.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_distance_off_line_xy(self):
        # eff_pos is offset in x and y, but same z as target_pos
        eff_pos = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([5.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_rotated_z_axis(self):
        # Rotate 90 deg about y, so z-axis points along x
        eff_pos = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        theta = np.pi / 2
        qw = np.cos(theta / 2)
        qy = np.sin(theta / 2)
        target_quat = torch.tensor([[qw, 0.0, qy, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([0.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_batch_inputs(self):
        eff_pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
        target_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        dist = dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        expected = torch.tensor([1.0, 2.0], dtype=dist.dtype)
        print("Expected:", expected, "Returned:", dist)
        self.assertTrue(torch.allclose(dist, expected, atol=1e-6))

    def test_invalid_shapes(self):
        eff_pos = torch.tensor([[1.0, 2.0]], dtype=torch.float32)  # shape (1,2)
        target_pos = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        try:
            dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        except ValueError as e:
            print("Expected: ValueError, Returned:", str(e))
        eff_pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        target_quat = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)  # shape (1,3)
        try:
            dist_eff_to_target_z_axis(eff_pos, (target_pos, target_quat))
        except ValueError as e:
            print("Expected: ValueError, Returned:", str(e))


if __name__ == "__main__":
    unittest.main()
