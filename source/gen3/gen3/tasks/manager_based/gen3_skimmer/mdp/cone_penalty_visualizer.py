#!/usr/bin/env python3
"""
Standalone 3D Cone Penalty Visualizer

This script creates an interactive 3D visualization of the penalty values inside a cone.
The penalty increases as points get closer to the center and closer to the cone vertex.
Color gradient: green (low penalty) to red (high penalty).

Usage:
    python3 cone_penalty_visualizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cone_penalty_3d(
    cone_radius: float = 0.1,           # Cone base radius (10cm)
    cone_height: float = 0.2,           # Cone height (20cm)
    sphere_radius: float = 0.15,        # Sphere radius (15cm)
    show_sphere: bool = True,           # Whether to show the sphere
    resolution: int = 50,                # Grid resolution
    save_path: str = None,              # Optional path to save the plot
    show_plot: bool = True              # Whether to display the plot
):
    """
    Create an interactive 3D visualization of the penalty values inside the cone.
    
    Args:
        cone_radius: Radius of the cone base
        cone_height: Height of the cone
        sphere_radius: Optional radius of sphere to plot (center at (0, 0, +sphere_radius))
        show_sphere: Whether to show the sphere
        resolution: Grid resolution for visualization
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    
    print(f"Creating 3D cone penalty visualization...")
    print(f"Cone radius: {cone_radius}m")
    print(f"Cone height: {cone_height}m")
    print(f"Sphere radius: {sphere_radius}m")
    print(f"Show sphere: {show_sphere}")
    
    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a grid of points inside the cone
    # Note: cone vertex is at (0,0,0) and base is at z = cone_height
    z_values = np.linspace(0, cone_height, resolution//6)
    r_values = np.linspace(0, cone_radius, resolution//2)
    theta_values = np.linspace(0, 2*np.pi, resolution)
    
    Z, R, Theta = np.meshgrid(z_values, r_values, theta_values)
    
    # Convert to Cartesian coordinates
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # Calculate the maximum radius at each height (cone shape)
    # At z=0 (vertex): radius = 0
    # At z=cone_height (base): radius = cone_radius
    max_radius_at_height = Z * (cone_radius / cone_height)
    
    # Only keep points that are inside the cone
    # A point is inside if its distance from center (r) is <= max_radius_at_height
    inside_cone = R <= max_radius_at_height
    
    # Filter points to only include those inside the cone
    X_flat = X[inside_cone]
    Y_flat = Y[inside_cone]
    Z_flat = Z[inside_cone]
    R_flat = R[inside_cone]
    max_radius_flat = max_radius_at_height[inside_cone]

    # Calculate penalty values
    # Penalty increases as points get closer to center (r=0) and closer to vertex (z=0)
    # Use a realistic penalty function that matches the environment logic
    distance_to_boundary = max_radius_flat - R_flat
    max_distance = max_radius_flat
    
    # Avoid division by zero
    # Increase penalty as Z decreases (closer to cone vertex)
    # Normalize Z so that z=0 (vertex) -> 1, z=cone_height (base) -> 0
    z_penalty = 1 - (Z_flat / cone_height)
    base_penalty = np.where(max_distance > 0, 
                            np.clip(distance_to_boundary / max_distance, 0, 1), 
                            0)
    # Combine: multiply base penalty by z_penalty to increase penalty as Z decreases
    # penalty = base_penalty * (0.5 + 0.5 * z_penalty)

    for i in range(len(np.arange(0, 0.2, 0.025))):
        base_penalty[Z_flat < (cone_height-i*0.025)] += 0.02
        # base_penalty[Z_flat < (cone_height-i)] += 0.25
    # base_penalty[Z_flat < 0.10] += 0.75
    # base_penalty[Z_flat < 0.05] += 0.75

    penalty = base_penalty

    
    # Invert so that points closer to boundary have lower penalty (green)
    # and points closer to center have higher penalty (red)
    # penalty = 1 - penalty
    
    # Create scatter plot with color based on penalty
    scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=penalty, 
                        cmap='RdYlGn_r', alpha=0.8, s=15)
    
    # Add cone boundary surface
    z_boundary = np.linspace(0, cone_height, 100)
    theta_boundary = np.linspace(0, 2*np.pi, 100)
    Z_boundary, Theta_boundary = np.meshgrid(z_boundary, theta_boundary)
    
    # Calculate radius at each height for cone boundary
    R_boundary = Z_boundary * (cone_radius / cone_height)
    X_boundary = R_boundary * np.cos(Theta_boundary)
    Y_boundary = R_boundary * np.sin(Theta_boundary)
    
    # Plot cone boundary as wireframe
    ax.plot_wireframe(X_boundary, Y_boundary, Z_boundary, 
                     color='black', alpha=0.5, linewidth=1.0)
    
    # Add sphere surface if sphere_radius is provided and show_sphere is True
    if sphere_radius is not None and show_sphere:
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
        ax.plot_surface(X_sphere, Y_sphere, Z_sphere, alpha=0.3, color='blue', edgecolor='blue', linewidth=0.5)
    
    # Set labels and title
    ax.set_xlabel('X (relative to target)', fontsize=12)
    ax.set_ylabel('Y (relative to target)', fontsize=12)
    ax.set_zlabel('Z (relative to target)', fontsize=12)
    title = '3D Cone Penalty Visualization\n(Red: High Penalty, Green: Low Penalty)'
    if show_sphere and sphere_radius is not None:
        title += '\n(Blue: Sphere)'
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set equal axis limits for proper visualization (no distortion)
    max_range = max(cone_radius, cone_height)
    if sphere_radius is not None and show_sphere:
        max_range = max(max_range, sphere_radius * 2)
    
    # Ensure XY axes have the same limits to prevent distortion
    xy_limit = max(cone_radius, sphere_radius if sphere_radius and show_sphere else 0)
    ax.set_xlim([-xy_limit, xy_limit])
    ax.set_ylim([-xy_limit, xy_limit])
    ax.set_zlim([0, max_range])
    
    # Set equal aspect ratio for all axes to maintain proper proportions
    ax.set_box_aspect([1, 1, 1])
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label('Penalty Value', fontsize=12)
    
    # Add text annotations
    ax.text2D(0.02, 0.98, f'Cone Radius: {cone_radius}m', transform=ax.transAxes, fontsize=10)
    ax.text2D(0.02, 0.94, f'Cone Height: {cone_height}m', transform=ax.transAxes, fontsize=10)
    if sphere_radius is not None and show_sphere:
        ax.text2D(0.02, 0.90, f'Sphere Radius: {sphere_radius}m', transform=ax.transAxes, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D penalty visualization saved to {save_path}")
    
    if show_plot:
        print("Displaying interactive 3D plot...")
        print("You can rotate, zoom, and pan the visualization.")
        print("Press 'q' or close the window to exit.")
        plt.show()

    plt.close(fig)
    return fig

def main():
    """Main function to run the cone penalty visualization."""
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D Cone Penalty Visualizer')
    parser.add_argument('--cone-radius', type=float, default=0.1, help='Cone base radius in meters (default: 0.1)')
    parser.add_argument('--cone-height', type=float, default=0.2, help='Cone height in meters (default: 0.2)')
    parser.add_argument('--sphere-radius', type=float, default=0.15, help='Sphere radius in meters (default: 0.15)')
    parser.add_argument('--show-sphere', action='store_true', default=True, help='Show the sphere (default: True)')
    parser.add_argument('--no-sphere', dest='show_sphere', action='store_false', help='Hide the sphere')
    parser.add_argument('--resolution', type=int, default=60, help='Grid resolution (default: 60)')
    parser.add_argument('--save', type=str, help='Save plot to file (optional)')
    
    args = parser.parse_args()
    
    # Default parameters (matching the environment configuration)
    cone_radius = args.cone_radius    # 10cm
    cone_height = args.cone_height    # 20cm
    sphere_radius = args.sphere_radius # 15cm
    show_sphere = args.show_sphere
    resolution = args.resolution
    save_path = args.save
    
    print("=== 3D Cone Penalty Visualizer ===")
    print("This visualization shows the penalty values inside the cone.")
    print("Red areas indicate high penalty (closer to center/vertex).")
    print("Green areas indicate low penalty (closer to boundary/base).")
    print(f"Cone vertex is at (0,0,0) and base is at z={cone_height}")
    print()
    
    # Create the visualization
    visualize_cone_penalty_3d(
        cone_radius=cone_radius,
        cone_height=cone_height,
        sphere_radius=sphere_radius,
        show_sphere=show_sphere,
        resolution=resolution,
        save_path=save_path,
        show_plot=True
    )

if __name__ == "__main__":
    main() 