# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.use('TkAgg') # Or 'Qt5Agg', 'GTK3Agg', etc.
except ImportError:
    print("Warning: TkAgg backend not available. Plot might not show.")

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial.transform import Rotation


def plot_trajectory_with_cone(
    end_effector_positions: torch.Tensor,  # Shape: (num_steps, 3) - x, y, z positions
    target_position: torch.Tensor,         # Shape: (3,) - target x, y, z position
    target_orientation: torch.Tensor,      # Shape: (4,) - target quaternion (w, x, y, z)
    cone_radius: float,                    # Cone base radius
    cone_height: float,                    # Cone height
    save_path: str = None,                 # Path to save the plot
    show_plot: bool = True                 # Whether to display the plot
):
    """
    Plot the end-effector trajectory with YZ and XZ projections showing the cone penalty region.

    Args:
        end_effector_positions: Tensor of end-effector positions over time
        target_position: Target position (origin of plots)
        target_orientation: Target orientation quaternion
        cone_radius: Radius of the cone base
        cone_height: Height of the cone
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """

    # Convert to numpy for plotting
    if isinstance(end_effector_positions, torch.Tensor):
        ee_positions = end_effector_positions.detach().cpu().numpy()
    else:
        ee_positions = np.array(end_effector_positions)

    if isinstance(target_position, torch.Tensor):
        target_pos = target_position.detach().cpu().numpy()
    else:
        target_pos = np.array(target_position)

    if isinstance(target_orientation, torch.Tensor):
        target_quat = target_orientation.detach().cpu().numpy()
    else:
        target_quat = np.array(target_orientation)

    # Calculate relative positions (origin is target)
    relative_positions = ee_positions - target_pos

    print(f"target_pos = {target_pos}")
    print(f"ee_positions = {ee_positions}")
    print(f"relative_positions.shape = {relative_positions.shape}")
    print(f"relative_positions[0] = {relative_positions[0]}")
    print(f"relative_positions[-1] = {relative_positions[-1]}")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  # Make figure taller

    # Plot YZ projection (left subplot)
    ax1.scatter(relative_positions[:, 1], relative_positions[:, 2],
                c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax1.plot(relative_positions[:, 1], relative_positions[:, 2], 'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax1.scatter(relative_positions[0, 1], relative_positions[0, 2], color='green', marker='s', s=45, zorder=5)
    ax1.scatter(relative_positions[-1, 1], relative_positions[-1, 2], color='red', marker='s', s=45, zorder=5)

    # Add cone boundary to YZ plot
    z_values = np.linspace(0, cone_height, 100)
    y_values = z_values * (cone_radius / cone_height)  # Linear interpolation
    ax1.plot(y_values, z_values, 'r--', linewidth=2, label='Cone boundary')
    ax1.plot(-y_values, z_values, 'r--', linewidth=2)

    ax1.set_xlabel('Y (relative to target)')
    ax1.set_ylabel('Z (relative to target)')
    ax1.set_title('YZ Projection of End-Effector Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Fix aspect ratio: make Y and Z axes have the same scale, but allow the plot to fill the subplot
    # This avoids the "collapsed" Y axis problem
    y_min, y_max = np.min(relative_positions[:, 1]), np.max(relative_positions[:, 1])
    z_min, z_max = np.min(relative_positions[:, 2]), np.max(relative_positions[:, 2])
    # Also include cone boundary in limits
    y_min = min(y_min, -cone_radius)
    y_max = max(y_max, cone_radius)
    z_min = min(z_min, 0)
    z_max = max(z_max, cone_height, np.max(z_values))
    ax1.set_xlim([y_min, y_max])
    ax1.set_ylim([z_min, z_max])
    ax1.set_aspect((y_max - y_min) / (z_max - z_min))  # Set aspect ratio to match data range

    # Plot XZ projection (right subplot)
    ax2.scatter(relative_positions[:, 0], relative_positions[:, 2],
                c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax2.plot(relative_positions[:, 0], relative_positions[:, 2], 'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax2.scatter(relative_positions[0, 0], relative_positions[0, 2], color='green', marker='s', s=45, zorder=5)
    ax2.scatter(relative_positions[-1, 0], relative_positions[-1, 2], color='red', marker='s', s=45, zorder=5)

    # Add cone boundary to XZ plot
    ax2.plot(y_values, z_values, 'r--', linewidth=2, label='Cone boundary')
    ax2.plot(-y_values, z_values, 'r--', linewidth=2)

    ax2.set_xlabel('X (relative to target)')
    ax2.set_ylabel('Z (relative to target)')
    ax2.set_title('XZ Projection of End-Effector Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Fix aspect ratio for XZ plot as well
    x_min, x_max = np.min(relative_positions[:, 0]), np.max(relative_positions[:, 0])
    x_min = min(x_min, -cone_radius)
    x_max = max(x_max, cone_radius)
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([z_min, z_max])
    ax2.set_aspect((x_max - x_min) / (z_max - z_min))

    # Add colorbar to the right of the subplots
    # cbar = fig.colorbar(ax1.collections[0], ax=[ax1, ax2], location='right', shrink=0.8, pad=0.02)
    # cbar.set_label('Time step')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig

def plot_3d_trajectory_with_cone(
    end_effector_positions: torch.Tensor,  # Shape: (num_steps, 3) - x, y, z positions
    target_position: torch.Tensor,         # Shape: (3,) - target x, y, z position
    target_orientation: torch.Tensor,      # Shape: (4,) - target quaternion (w, x, y, z)
    cone_radius: float,                    # Cone base radius
    cone_height: float,                    # Cone height
    save_path: str = None,                 # Path to save the plot
    show_plot: bool = True                 # Whether to display the plot
):
    """
    Plot the 3D end-effector trajectory with the cone penalty region.
    
    Args:
        end_effector_positions: Tensor of end-effector positions over time
        target_position: Target position (origin of plots)
        target_orientation: Target orientation quaternion
        cone_radius: Radius of the cone base
        cone_height: Height of the cone
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    
    # Convert to numpy for plotting
    if isinstance(end_effector_positions, torch.Tensor):
        ee_positions = end_effector_positions.detach().cpu().numpy()
    else:
        ee_positions = np.array(end_effector_positions)
    
    if isinstance(target_position, torch.Tensor):
        target_pos = target_position.detach().cpu().numpy()
    else:
        target_pos = np.array(target_position)
    
    if isinstance(target_orientation, torch.Tensor):
        target_quat = target_orientation.detach().cpu().numpy()
    else:
        target_quat = np.array(target_orientation)
    
    # Calculate relative positions (origin is target)
    relative_positions = ee_positions - target_pos

    print(f"target_pos = {target_pos}")
    print(f"ee_positions = {ee_positions}")
    print(f"relative_positions[0] = {relative_positions[0]}")
    print(f"relative_positions[-1] = {relative_positions[-1]}")
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.scatter(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
               c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax.plot(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
            'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax.scatter(relative_positions[0, 0], relative_positions[0, 1], relative_positions[0, 2], color='green', marker='s', s=45, zorder=5)
    ax.scatter(relative_positions[-1, 0], relative_positions[-1, 1], relative_positions[-1, 2], color='red', marker='s', s=45, zorder=5)
    
    # Create cone surface
    z_values = np.linspace(0, cone_height, 50)
    theta_values = np.linspace(0, 2*np.pi, 50)
    Z, Theta = np.meshgrid(z_values, theta_values)
    
    # Calculate radius at each height
    R = Z * (cone_radius / cone_height)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Plot cone surface
    ax.plot_surface(X, Y, Z, alpha=0.3, color='red', edgecolor='red', linewidth=0.5)
    
    ax.set_xlabel('X (relative to target)')
    ax.set_ylabel('Y (relative to target)')
    ax.set_zlabel('Z (relative to target)')
    ax.set_title('3D End-Effector Trajectory with Cone Penalty Region')
    
    # Add colorbar to the right of the 3D plot
    scatter = ax.scatter([], [], [], c=[], cmap='viridis')
    # cbar = fig.colorbar(scatter, ax=ax, location='right', shrink=0.8, pad=0.02)
    # cbar.set_label('Time step')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    if show_plot:
        plt.show()

    plt.close(fig)
    return fig

def analyze_trajectory_cone_penalty(
    end_effector_positions: torch.Tensor,  # Shape: (num_steps, 3) - x, y, z positions
    target_position: torch.Tensor,         # Shape: (3,) - target x, y, z position
    target_orientation: torch.Tensor,      # Shape: (4,) - target quaternion (w, x, y, z)
    cone_radius: float,                    # Cone base radius
    cone_height: float,                    # Cone height
):
    """
    Analyze how much of the trajectory was inside the cone penalty region.
    
    Args:
        end_effector_positions: Tensor of end-effector positions over time
        target_position: Target position
        target_orientation: Target orientation quaternion
        cone_radius: Radius of the cone base
        cone_height: Height of the cone
    
    Returns:
        Dictionary with analysis results
    """
    
    # Convert to numpy for analysis
    if isinstance(end_effector_positions, torch.Tensor):
        ee_positions = end_effector_positions.detach().cpu().numpy()
    else:
        ee_positions = np.array(end_effector_positions)
    
    if isinstance(target_position, torch.Tensor):
        target_pos = target_position.detach().cpu().numpy()
    else:
        target_pos = np.array(target_position)
    
    if isinstance(target_orientation, torch.Tensor):
        target_quat = target_orientation.detach().cpu().numpy()
    else:
        target_quat = np.array(target_orientation)
    
    # Calculate relative positions
    relative_positions = ee_positions - target_pos
    
    # Calculate distance to cone axis (Z-axis in target frame)
    distances_to_axis = np.sqrt(relative_positions[:, 0]**2 + relative_positions[:, 1]**2)
    
    # Calculate height relative to target
    heights = relative_positions[:, 2]
    
    # Calculate cone radius at each height
    cone_radii = np.maximum(heights * (cone_radius / cone_height), 1e-6)
    
    # Check if points are inside cone
    inside_cone = (distances_to_axis <= cone_radii) & (heights >= 0) & (heights <= cone_height)
    
    # Calculate statistics
    total_steps = len(ee_positions)
    steps_inside_cone = np.sum(inside_cone)
    percentage_inside_cone = (steps_inside_cone / total_steps) * 100
    
    # Calculate average distance to cone boundary when inside
    distances_to_boundary = cone_radii - distances_to_axis
    avg_distance_to_boundary = np.mean(distances_to_boundary[inside_cone]) if np.any(inside_cone) else np.inf
    
    return {
        'total_steps': total_steps,
        'steps_inside_cone': steps_inside_cone,
        'percentage_inside_cone': percentage_inside_cone,
        'avg_distance_to_boundary': avg_distance_to_boundary,
        'inside_cone_mask': inside_cone
    }

# Example usage function
def example_usage():
    """Example of how to use the plotting functions."""
    
    # Generate sample trajectory data
    num_steps = 100
    t = np.linspace(0, 2*np.pi, num_steps)
    
    # Create a spiral trajectory that approaches the target
    x = 0.1 * np.cos(t) * np.exp(-t/4)
    y = 0.1 * np.sin(t) * np.exp(-t/4)
    z = 0.5 * (1 - np.exp(-t/4))  # Approaches 0.5 from 0
    
    end_effector_positions = torch.tensor(np.column_stack([x, y, z]))
    target_position = torch.tensor([0.0, 0.0, 0.0])
    target_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    cone_radius = 0.1
    cone_height = 0.5
    
    # Plot 2D projections
    plot_trajectory_with_cone(
        end_effector_positions, target_position, target_orientation,
        cone_radius, cone_height, save_path="trajectory_2d.png"
    )
    
    # Plot 3D trajectory
    plot_3d_trajectory_with_cone(
        end_effector_positions, target_position, target_orientation,
        cone_radius, cone_height, save_path="trajectory_3d.png"
    )
    
    # Analyze trajectory
    analysis = analyze_trajectory_cone_penalty(
        end_effector_positions, target_position, target_orientation,
        cone_radius, cone_height
    )
    
    print("Trajectory Analysis:")
    print(f"Total steps: {analysis['total_steps']}")
    print(f"Steps inside cone: {analysis['steps_inside_cone']}")
    print(f"Percentage inside cone: {analysis['percentage_inside_cone']:.2f}%")
    print(f"Average distance to boundary: {analysis['avg_distance_to_boundary']:.4f}")

if __name__ == "__main__":
    example_usage() 