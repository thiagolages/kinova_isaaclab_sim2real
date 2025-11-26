# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
try:
    matplotlib.use('TkAgg') # Or 'Qt5Agg', 'GTK3Agg', etc.
except ImportError:
    print("Warning: TkAgg backend not available. Plot might not show.")

def quaternion_to_rotation_matrix(quat):
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
    
    Args:
        quat: Quaternion as (w, x, y, z)
    
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    
    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    
    # Convert to rotation matrix
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])
    
    return R

def get_z_axis_direction(quat):
    """
    Extract Z-axis direction from quaternion.
    
    Args:
        quat: Quaternion as (w, x, y, z)
    
    Returns:
        Z-axis direction as (x, y, z) unit vector
    """
    R = quaternion_to_rotation_matrix(quat)
    # Z-axis is the third column of the rotation matrix
    #z_axis = R[:, 2]
    ## CORRECTION: SINCE WE ROTATE -90DEG IN Y, LET'S GET (minus) THE X-AXIS
    z_axis = -R[:, 0]

    return z_axis

def plot_trajectory_with_cone(
    end_effector_positions: torch.Tensor,  # Shape: (num_steps, 3) - x, y, z positions
    end_effector_orientations: torch.Tensor,  # Shape: (num_steps, 4) - end-effector quaternions (w, x, y, z)
    target_position: torch.Tensor,         # Shape: (3,) - target x, y, z position
    target_orientation: torch.Tensor,      # Shape: (4,) - target quaternion (w, x, y, z)
    cone_radius: float,                    # Cone base radius
    cone_height: float,                    # Cone height
    sphere_radius: float = None,           # Sphere radius (optional)
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
        sphere_radius: Optional radius of sphere to plot (center at (0, 0, +sphere_radius))
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """

    # Convert to numpy for plotting
    if isinstance(end_effector_positions, torch.Tensor):
        ee_positions = end_effector_positions.detach().cpu().numpy()
    else:
        ee_positions = np.array(end_effector_positions)

    if end_effector_orientations is None:
        ee_orientations = None
    else:
        if isinstance(end_effector_orientations, torch.Tensor):
            ee_orientations = end_effector_orientations.detach().cpu().numpy()
        else:
            ee_orientations = np.array(end_effector_orientations)

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

    # print(f"target_pos = {target_pos}")
    # print(f"ee_positions = {ee_positions}")
    # print(f"relative_positions.shape = {relative_positions.shape}")
    # print(f"relative_positions[0] = {relative_positions[0]}")
    # print(f"relative_positions[-1] = {relative_positions[-1]}")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))  # Make figure taller

    # Calculate cone boundary points
    z_values = np.linspace(0, cone_height, 100)
    y_values = z_values * (cone_radius / cone_height)  # Linear interpolation

    # Calculate unified axis limits for both plots
    # Get trajectory limits
    x_min, x_max = np.min(relative_positions[:, 0]), np.max(relative_positions[:, 0])
    y_min, y_max = np.min(relative_positions[:, 1]), np.max(relative_positions[:, 1])
    z_min, z_max = np.min(relative_positions[:, 2]), np.max(relative_positions[:, 2])
    
    # Include cone boundary in limits
    x_min = min(x_min, -cone_radius)
    x_max = max(x_max, cone_radius)
    y_min = min(y_min, -cone_radius)
    y_max = max(y_max, cone_radius)
    z_min = min(z_min, 0)
    z_max = max(z_max, cone_height, np.max(z_values))
    
    # Include sphere in limits if provided
    if sphere_radius is not None:
        x_min = min(x_min, -sphere_radius)
        x_max = max(x_max, sphere_radius)
        y_min = min(y_min, -sphere_radius)
        y_max = max(y_max, sphere_radius)
        z_max = max(z_max, sphere_radius * 2)  # Sphere extends from 0 to 2*sphere_radius
    
    # Calculate unified limits for both plots to ensure consistency
    # Use the maximum range across X and Y dimensions only (not Z)
    max_x_range = x_max - x_min
    max_y_range = y_max - y_min
    max_xy_range = max(max_x_range, max_y_range)
    
    # Center the X and Y ranges
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Apply unified limits for X and Y, but keep Z tight
    unified_x_min = x_center - max_xy_range / 2
    unified_x_max = x_center + max_xy_range / 2
    unified_y_min = y_center - max_xy_range / 2
    unified_y_max = y_center + max_xy_range / 2
    
    # Keep Z limits tight to reduce white space
    unified_z_min = z_min
    unified_z_max = z_max

    # Plot YZ projection (left subplot)
    ax1.scatter(relative_positions[:, 1], relative_positions[:, 2],
                c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax1.plot(relative_positions[:, 1], relative_positions[:, 2], 'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax1.scatter(relative_positions[0, 1], relative_positions[0, 2], color='red', marker='s', s=45, zorder=5)
    ax1.scatter(relative_positions[-1, 1], relative_positions[-1, 2], color='green', marker='s', s=45, zorder=5)
    
    # Add Z-axis arrows for end-effector orientation
    arrow_length = 0.005  # Small arrow length
    arrow_width = 0.002  # Arrow width
    
    # Sample every few points to avoid overcrowding
    step = max(1, len(relative_positions) // 20)  # Show ~20 arrows max
    if ee_orientations is not None:
        for i in range(0, len(relative_positions), step):
            # Get the actual end-effector orientation for this point
            ee_quat = ee_orientations[i]
            
            # Extract Z-axis direction in world coordinates
            z_axis_world = get_z_axis_direction(ee_quat)
            
            # In YZ projection, we show Y and Z components of the Z-axis
            z_direction_y = z_axis_world[1]  # Y component of Z-axis
            z_direction_z = z_axis_world[2]  # Z component of Z-axis
            
            # Draw arrow
            ax1.arrow(relative_positions[i, 1], relative_positions[i, 2],
                    z_direction_y * arrow_length, z_direction_z * arrow_length,
                    head_width=arrow_width, head_length=arrow_width*2,
                    fc='blue', ec='blue', alpha=0.6, zorder=4)

    # Add cone boundary to YZ plot
    ax1.plot(y_values, z_values, 'r--', linewidth=2, label='Cone boundary')
    ax1.plot(-y_values, z_values, 'r--', linewidth=2)

    # Add sphere to YZ plot if sphere_radius is provided
    if sphere_radius is not None:
        sphere_center_z = sphere_radius  # Sphere center at (0, 0, +sphere_radius)
        sphere_center_y = 0.0
        
        # Create sphere boundary (circle in YZ plane)
        theta = np.linspace(0, 2*np.pi, 100)
        sphere_y = sphere_center_y + sphere_radius * np.cos(theta)
        sphere_z = sphere_center_z + sphere_radius * np.sin(theta)
        
        # Plot the full sphere (no masking needed)
        ax1.plot(sphere_y, sphere_z, 'g--', linewidth=2, label='Sphere boundary')

    ax1.set_xlabel('Y (relative to target)')
    ax1.set_ylabel('Z (relative to target)')
    ax1.set_title('YZ Projection of End-Effector Trajectory')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Apply unified limits to YZ plot
    ax1.set_xlim([unified_y_min, unified_y_max])
    ax1.set_ylim([unified_z_min, unified_z_max])
    ax1.set_aspect('equal')

    # Plot XZ projection (right subplot)
    ax2.scatter(relative_positions[:, 0], relative_positions[:, 2],
                c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax2.plot(relative_positions[:, 0], relative_positions[:, 2], 'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax2.scatter(relative_positions[0, 0], relative_positions[0, 2], color='red', marker='s', s=45, zorder=5)
    ax2.scatter(relative_positions[-1, 0], relative_positions[-1, 2], color='green', marker='s', s=45, zorder=5)
    
    # Add Z-axis arrows for end-effector orientation (same as YZ plot)
    if ee_orientations is not None:
        for i in range(0, len(relative_positions), step):
            # Get the actual end-effector orientation for this point
            ee_quat = ee_orientations[i]
            
            # Extract Z-axis direction in world coordinates
            z_axis_world = get_z_axis_direction(ee_quat)
            
            # In XZ projection, we show X and Z components of the Z-axis
            z_direction_x = z_axis_world[0]  # X component of Z-axis
            z_direction_z = z_axis_world[2]  # Z component of Z-axis
            
            # Draw arrow
            ax2.arrow(relative_positions[i, 0], relative_positions[i, 2],
                    z_direction_x * arrow_length, z_direction_z * arrow_length,
                    head_width=arrow_width, head_length=arrow_width*2,
                    fc='blue', ec='blue', alpha=0.6, zorder=4)

    # Add cone boundary to XZ plot
    ax2.plot(y_values, z_values, 'r--', linewidth=2, label='Cone boundary')
    ax2.plot(-y_values, z_values, 'r--', linewidth=2)

    # Add sphere to XZ plot if sphere_radius is provided
    if sphere_radius is not None:
        sphere_center_x = 0.0
        
        # Create sphere boundary (circle in XZ plane)
        sphere_x = sphere_center_x + sphere_radius * np.cos(theta)
        sphere_z = sphere_center_z + sphere_radius * np.sin(theta)
        
        # Plot the full sphere (no masking needed)
        ax2.plot(sphere_x, sphere_z, 'g--', linewidth=2, label='Sphere boundary')

    ax2.set_xlabel('X (relative to target)')
    ax2.set_ylabel('Z (relative to target)')
    ax2.set_title('XZ Projection of End-Effector Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Apply unified limits to XZ plot
    ax2.set_xlim([unified_x_min, unified_x_max])
    ax2.set_ylim([unified_z_min, unified_z_max])
    ax2.set_aspect('equal')

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
    end_effector_orientations: torch.Tensor,  # Shape: (num_steps, 4) - end-effector quaternions (w, x, y, z)
    target_position: torch.Tensor,         # Shape: (3,) - target x, y, z position
    target_orientation: torch.Tensor,      # Shape: (4,) - target quaternion (w, x, y, z)
    cone_radius: float,                    # Cone base radius
    cone_height: float,                    # Cone height
    sphere_radius: float = None,           # Sphere radius (optional)
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
        sphere_radius: Optional radius of sphere to plot (center at (0, 0, +sphere_radius))
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    
    # Convert to numpy for plotting
    if isinstance(end_effector_positions, torch.Tensor):
        ee_positions = end_effector_positions.detach().cpu().numpy()
    else:
        ee_positions = np.array(end_effector_positions)
    
    if end_effector_orientations is None:
        ee_orientations = None
    else:
        if isinstance(end_effector_orientations, torch.Tensor):
            ee_orientations = end_effector_orientations.detach().cpu().numpy()
        else:
            ee_orientations = np.array(end_effector_orientations)
    
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

    # print(f"target_pos = {target_pos}")
    # print(f"ee_positions = {ee_positions}")
    # print(f"relative_positions[0] = {relative_positions[0]}")
    # print(f"relative_positions[-1] = {relative_positions[-1]}")
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.scatter(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
               c=range(len(relative_positions)), cmap='viridis', alpha=0.7, s=20)
    ax.plot(relative_positions[:, 0], relative_positions[:, 1], relative_positions[:, 2], 
            'b-', alpha=0.5, linewidth=1)
    # Highlight the first point as a red square and last as a green square
    ax.scatter(relative_positions[0, 0], relative_positions[0, 1], relative_positions[0, 2], color='red', marker='s', s=45, zorder=5)
    ax.scatter(relative_positions[-1, 0], relative_positions[-1, 1], relative_positions[-1, 2], color='green', marker='s', s=45, zorder=5)
    
    # Add Z-axis arrows for end-effector orientation
    arrow_length = 0.02  # Small arrow length
    
    # Sample every few points to avoid overcrowding
    # step = max(1, len(relative_positions))  # Show ~20 arrows max
    step = 1
    
    if ee_orientations is not None:
        for i in range(0, len(relative_positions), step):
            # Get the actual end-effector orientation for this point
            ee_quat = ee_orientations[i]
            
            # Extract Z-axis direction in world coordinates
            z_axis_world = get_z_axis_direction(ee_quat)
            
            # Draw 3D arrow
            start_point = relative_positions[i]
            end_point = start_point + z_axis_world * arrow_length
            
            ax.quiver(start_point[0], start_point[1], start_point[2],
                    z_axis_world[0], z_axis_world[1], z_axis_world[2],
                    length=arrow_length, color='blue', alpha=0.6, arrow_length_ratio=0.3)
    
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
    
    # Add sphere surface if sphere_radius is provided
    if sphere_radius is not None:
        sphere_center_z = sphere_radius  # Sphere center at (0, 0, +sphere_radius)
        
        # Create sphere surface (full sphere)
        phi_values = np.linspace(0, 2*np.pi, 50)  # Full sphere
        theta_values = np.linspace(0, np.pi, 50)
        Phi, Theta = np.meshgrid(phi_values, theta_values)
        
        # Calculate sphere coordinates
        X_sphere = sphere_radius * np.sin(Theta) * np.cos(Phi)
        Y_sphere = sphere_radius * np.sin(Theta) * np.sin(Phi)
        Z_sphere = sphere_center_z + sphere_radius * np.cos(Theta)
        
        # Plot sphere surface (full sphere)
        ax.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.3, color='green', edgecolor='green', linewidth=0.5)
    
    ax.set_xlabel('X (relative to target)')
    ax.set_ylabel('Y (relative to target)')
    ax.set_zlabel('Z (relative to target)')
    ax.set_title('3D End-Effector Trajectory with Cone Penalty Region', fontsize=22, fontweight='bold')
    
    # Calculate unified axis limits for undistorted cone
    # Get trajectory limits
    x_min, x_max = np.min(relative_positions[:, 0]), np.max(relative_positions[:, 0])
    y_min, y_max = np.min(relative_positions[:, 1]), np.max(relative_positions[:, 1])
    z_min, z_max = np.min(relative_positions[:, 2]), np.max(relative_positions[:, 2])
    
    # Include cone boundary in limits
    x_min = min(x_min, -cone_radius)
    x_max = max(x_max, cone_radius)
    y_min = min(y_min, -cone_radius)
    y_max = max(y_max, cone_radius)
    z_min = min(z_min, 0)
    z_max = max(z_max, cone_height)
    
    # Include sphere bounds if provided
    if sphere_radius is not None:
        x_min = min(x_min, -sphere_radius)
        x_max = max(x_max, sphere_radius)
        y_min = min(y_min, -sphere_radius)
        y_max = max(y_max, sphere_radius)
        z_max = max(z_max, sphere_radius * 2)
    
    # Calculate unified X and Y limits to prevent cone distortion
    # Use the maximum range across X and Y dimensions
    max_x_range = x_max - x_min
    max_y_range = y_max - y_min
    max_xy_range = max(max_x_range, max_y_range)
    
    # Center the X and Y ranges
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Apply unified limits for X and Y, keep Z tight
    unified_x_min = x_center - max_xy_range / 2
    unified_x_max = x_center + max_xy_range / 2
    unified_y_min = y_center - max_xy_range / 2
    unified_y_max = y_center + max_xy_range / 2
    
    # Set axis limits
    ax.set_xlim([unified_x_min, unified_x_max])
    ax.set_ylim([unified_y_min, unified_y_max])
    ax.set_zlim([z_min, z_max])
    
    # Set equal aspect ratio for all axes to make sphere appear spherical
    ax.set_box_aspect([1, 1, 1])
    
    # Add colorbar to the right of the 3D plot
    scatter = ax.scatter([], [], [], c=[], cmap='viridis')
    # cbar = fig.colorbar(scatter, ax=ax, location='right', shrink=0.8, pad=0.02)
    # cbar.set_label('Time step')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    if show_plot:
        plt.show()
        # plt.tight_layout()

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
    cone_height = 0.2
    sphere_radius = None #0.15  # Sphere radius for demonstration
    
    # Plot 2D projections
    plot_trajectory_with_cone(
        end_effector_positions, target_position, target_orientation,
        cone_radius, cone_height, sphere_radius=sphere_radius, save_path="trajectory_2d.png"
    )
    
    # Plot 3D trajectory
    plot_3d_trajectory_with_cone(
        end_effector_positions, target_position, target_orientation,
        cone_radius, cone_height, sphere_radius=sphere_radius, save_path="trajectory_3d.png"
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