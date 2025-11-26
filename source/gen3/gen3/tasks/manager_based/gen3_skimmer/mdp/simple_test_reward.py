import torch
from scipy.spatial.transform import Rotation as R

# --- Mock classes to simulate the environment and dependencies ---

class DummyRobotData:
    def __init__(self, eff_pos, root_state, body_state, root_quat):
        self.body_state_w = body_state  # (num_envs, num_bodies, 7)
        self.root_state_w = root_state  # (num_envs, 7)
        self.root_quat_w = root_quat    # (num_envs, 4)

class DummyRobot:
    def __init__(self, eff_pos, root_state, body_state, root_quat):
        self.data = DummyRobotData(eff_pos, root_state, body_state, root_quat)

class DummyCommandManager:
    def __init__(self, command):
        self._command = command
    def get_command(self, name):
        return self._command

class DummyEnv:
    def __init__(self, eff_pos, root_state, body_state, root_quat, command):
        self.scene = {'robot': DummyRobot(eff_pos, root_state, body_state, root_quat)}
        self.command_manager = DummyCommandManager(command)

class DummyAssetCfg:
    def __init__(self, name, body_ids):
        self.name = name
        self.body_ids = body_ids

# --- The function to test (copy your collision_from_top here if needed) ---
# from rewards import collision_from_top

# --- Simple test function ---
def simple_test_collision_from_top():
    # Test case: effector inside the sphere and above the plane
    eff_pos = torch.tensor([[0.0, 0.0, 0.0]])
    sphere_center = torch.tensor([[0.0, 0.0, 1.0]])
    sphere_r = 0.1
    plane_h_percentage = 0.5
    num_envs = eff_pos.shape[0]
    root_state = torch.cat([sphere_center, torch.tensor([[1.0, 0.0, 0.0, 0.0]])], dim=1)
    body_state = torch.zeros((num_envs, 1, 7))
    body_state[:, 0, :3] = eff_pos
    body_state[:, 0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    command = torch.cat([torch.zeros((num_envs, 3)), torch.tensor([[1.0, 0.0, 0.0, 0.0]])], dim=1)
    env = DummyEnv(eff_pos, root_state, body_state, root_quat, command)
    asset_cfg = DummyAssetCfg('robot', [0])

    # Import your function here if not in the same file
    from rewards import collision_from_top

    result = collision_from_top(env, asset_cfg, 'dummy', sphere_r=sphere_r, plane_h_percentage=plane_h_percentage)
    print("Test 1 (should be False):", result)

    # Test case: effector outside the sphere
    eff_pos2 = torch.tensor([[0.0, 0.0, 2.0]])
    body_state2 = torch.zeros((num_envs, 1, 7))
    body_state2[:, 0, :3] = eff_pos2
    body_state2[:, 0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    env2 = DummyEnv(eff_pos2, root_state, body_state2, root_quat, command)
    result2 = collision_from_top(env2, asset_cfg, 'dummy', sphere_r=sphere_r, plane_h_percentage=plane_h_percentage)
    print("Test 2 (should be False):", result2)

if __name__ == "__main__":
    simple_test_collision_from_top()
