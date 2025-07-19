# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from typing import Optional, Dict, Any

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv


class ConeVisualizer:
    """
    Dynamic cone visualizer that creates and manages cones for multiple environments.
    Each cone follows its corresponding target position.
    """
    
    def __init__(self, env: ManagerBasedRLEnv, cone_radius: float, cone_height: float):
        """
        Initialize the cone visualizer.
        
        Args:
            env: The environment instance
            cone_radius: Radius of the cone base
            cone_height: Height of the cone
        """
        self.env = env
        self.cone_radius = cone_radius
        self.cone_height = cone_height
        self.num_envs = env.num_envs
        self.device = env.device
        
        # Dictionary to store cone objects for each environment
        self.cones: Dict[int, RigidObject] = {}
        
        # Create cones for each environment
        self._create_cones()
    
    def _create_cones(self):
        """Create cone objects for each environment."""
        for env_id in range(self.num_envs):
            cone_cfg = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/ConeVisual_{env_id}",
                spawn=sim_utils.ConeCfg(
                    radius=self.cone_radius,
                    height=self.cone_height,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.8, 0.0),  # Green color
                        opacity=0.4  # Semi-transparent
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),  # Static cone
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),  # Will be updated dynamically
                    rot=(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
                ),
            )
            
            # Create the cone object
            cone = RigidObject(cfg=cone_cfg)
            self.cones[env_id] = cone
    
    def update_cones(self, target_positions: torch.Tensor, target_orientations: Optional[torch.Tensor] = None):
        """
        Update cone positions to follow target positions.
        
        Args:
            target_positions: Tensor of shape (num_envs, 3) with target positions
            target_orientations: Optional tensor of shape (num_envs, 4) with target orientations
        """
        if target_orientations is None:
            # Use identity quaternions if not provided
            target_orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(self.num_envs, 1)
        
        # Update each cone's position and orientation
        for env_id in range(self.num_envs):
            if env_id in self.cones:
                cone = self.cones[env_id]
                
                # Get target position and orientation for this environment
                target_pos = target_positions[env_id]
                target_quat = target_orientations[env_id]
                
                # Set cone position to target position
                cone.set_world_poses(
                    positions=target_pos.unsqueeze(0),  # Add batch dimension
                    orientations=target_quat.unsqueeze(0)
                )
    
    def get_cone_positions(self) -> torch.Tensor:
        """
        Get current positions of all cones.
        
        Returns:
            Tensor of shape (num_envs, 3) with cone positions
        """
        positions = []
        for env_id in range(self.num_envs):
            if env_id in self.cones:
                cone = self.cones[env_id]
                pos = cone.get_world_poses()[0]  # Get position (first element)
                positions.append(pos)
            else:
                positions.append(torch.zeros(3, device=self.device))
        
        return torch.stack(positions)
    
    def cleanup(self):
        """Clean up cone objects."""
        for cone in self.cones.values():
            if hasattr(cone, 'destroy'):
                cone.destroy()
        self.cones.clear()


def create_cone_visualizer(env: ManagerBasedRLEnv, cone_radius: float, cone_height: float) -> ConeVisualizer:
    """
    Factory function to create a cone visualizer.
    
    Args:
        env: The environment instance
        cone_radius: Radius of the cone base
        cone_height: Height of the cone
    
    Returns:
        ConeVisualizer instance
    """
    return ConeVisualizer(env, cone_radius, cone_height)


# Example usage and testing
def test_cone_visualizer():
    """Test function for the cone visualizer."""
    print("ConeVisualizer module loaded successfully.")
    print("This module provides dynamic cone visualization for multiple environments.")
    print("Usage:")
    print("1. Create visualizer: visualizer = ConeVisualizer(env, cone_radius, cone_height)")
    print("2. Update cones: visualizer.update_cones(target_positions, target_orientations)")
    print("3. Cleanup: visualizer.cleanup()")


if __name__ == "__main__":
    test_cone_visualizer() 