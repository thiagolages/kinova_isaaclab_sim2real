import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial.transform import Rotation

# --- Helper function to calculate penalty for a single point ---
def calculate_single_point_penalty(
    eff_pos_w: torch.Tensor,       # Current point in world (1D tensor with 3 elements)
    target_pos_w: torch.Tensor,    # Target position in world (1D tensor with 3 elements)
    target_quat_w: torch.Tensor,   # Target orientation in world (w,x,y,z) (1D tensor with 4 elements)
    cone_r: float,                 # Cone radius at base
    cone_h: float,                 # Cone height
    delta_r_param: float,          # Scalar parameter for radial penalty scaling
    delta_h_param: float           # Scalar parameter for height penalty scaling
) -> float:
    """
    Calculates the penalty magnitude for a single 3D point based on the cone_penalty logic.
    Returns a scalar penalty value (rew component before weight multiplication).
    """
    if cone_r <= 0 or cone_h <= 0:
        raise ValueError("cone_r and cone_h must be positive.")
    if delta_r_param <= 0 or delta_h_param <= 0:
        raise ValueError("delta_r_param and delta_h_param must be positive.")

    # Ensure inputs are torch tensors
    eff_pos_w = torch.as_tensor(eff_pos_w, dtype=torch.float32)
    target_pos_w = torch.as_tensor(target_pos_w, dtype=torch.float32)
    target_quat_w = torch.as_tensor(target_quat_w, dtype=torch.float32)

    # 1. Calculate target frame axes in world
    # Scipy Rotation uses (x,y,z,w) for quaternion
    rot = Rotation.from_quat(target_quat_w.numpy()[1:] # x,y,z
                             .tolist() + [target_quat_w.numpy()[0]]) # w
    
    target_z_axis_w = torch.tensor(rot.apply([0., 0., 1.]), dtype=torch.float32)

    # 2. Calculate geometric quantities relative to the target frame
    vec_eff_to_target_origin = eff_pos_w - target_pos_w
    
    # Effector height with respect to the target XY plane (along target_z_axis_w)
    eff_cone_height = torch.dot(vec_eff_to_target_origin, target_z_axis_w)
    
    # Effector distance to the cone's Z-axis
    projection_on_z_axis = eff_cone_height * target_z_axis_w
    vec_in_xy_plane = vec_eff_to_target_origin - projection_on_z_axis
    dist_to_cone_axis = torch.norm(vec_in_xy_plane)

    # 3. Calculate cone's parameters at current height
    # theta = 2 * atan(r/h)
    theta_rad = 2 * math.atan2(cone_r, cone_h) # math.atan2 takes (y,x)
    
    # Radius of the cone at the current effector height: r = h_eff * tan(theta/2)
    # Ensure r_at_current_h is positive and non-zero
    # eff_cone_height can be negative if below the cone base plane
    r_at_current_h_val = eff_cone_height * math.tan(theta_rad / 2.0)
    r_at_current_h = torch.maximum(
        r_at_current_h_val,
        torch.tensor(1e-6, dtype=torch.float32) # Min radius to avoid division by zero
    )
    if eff_cone_height < 0 : # If below the cone base, effectively infinite radius for penalty calc, or treat as outside
        r_at_current_h = torch.tensor(float('inf'), dtype=torch.float32)


    # 4. Define multipliers
    # delta_r scales penalty by how close the effector is to the center
    current_delta_r = delta_r_param / r_at_current_h
    if torch.isinf(current_delta_r): # Handle case where r_at_current_h was inf
        current_delta_r = torch.tensor(0.0, dtype=torch.float32)


    # delta_h scales penalty by how close we are from the target, in the Z direction
    current_delta_h = delta_h_param * torch.clamp((cone_h - eff_cone_height) / cone_h, min=0.0)
    # Clamp to ensure non-negative if eff_cone_height > cone_h

    # 5. Check if effector is inside penalty cone region for error calculation
    # Mask is true if outside the cone's slope OR above the cone's max height
    # or below the cone's base (eff_cone_height < 0)
    is_above_cone = eff_cone_height > cone_h
    is_below_cone_base = eff_cone_height < 0
    
    mask = (dist_to_cone_axis > r_at_current_h) | is_above_cone | is_below_cone_base
    
    r_error = torch.where(
        mask,
        torch.tensor(0.0, dtype=torch.float32),
        r_at_current_h - dist_to_cone_axis # Penalty increases as dist_to_cone_axis gets smaller than r_at_current_h
    )
    
    # 6. Define final reward component (before weight)
    rew = -1.0 * current_delta_r * current_delta_h * r_error
    
    return rew.item()


# --- Main plotting function ---
def plot_cone_penalty_3d(
    target_pos: list[float],
    target_quat_wxyz: list[float], # (w,x,y,z)
    cone_radius: float,
    cone_height_val: float,
    delta_r_val: float,
    delta_h_val: float,
    grid_range: float = 1.0, # Max extent from target_pos for plotting grid
    grid_points_per_axis: int = 20
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_p = torch.tensor(target_pos, dtype=torch.float32)
    target_q = torch.tensor(target_quat_wxyz, dtype=torch.float32)

    x_coords = np.linspace(target_pos[0] - grid_range, target_pos[0] + grid_range, grid_points_per_axis)
    y_coords = np.linspace(target_pos[1] - grid_range, target_pos[1] + grid_range, grid_points_per_axis)
    z_coords = np.linspace(target_pos[2] - grid_range, target_pos[2] + grid_range, grid_points_per_axis)

    plot_points_x = []
    plot_points_y = []
    plot_points_z = []
    penalty_values = []

    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                eff_pos = torch.tensor([x, y, z], dtype=torch.float32)
                penalty = calculate_single_point_penalty(
                    eff_pos, target_p, target_q,
                    cone_radius, cone_height_val, delta_r_val, delta_h_val
                )
                if penalty > 1e-5: # Only plot if penalty is significantly non-zero
                    plot_points_x.append(x)
                    plot_points_y.append(y)
                    plot_points_z.append(z)
                    penalty_values.append(penalty)

    if not penalty_values:
        print("No points with significant penalty found to plot.")
        ax.set_title("Cone Penalty Visualization (No penalties > 0)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        return

    penalties_np = np.array(penalty_values)
    
    # Create a colormap: green (low penalty) -> yellow -> red (high penalty)
    cmap = LinearSegmentedColormap.from_list("penalty_cmap", ["green", "yellow", "red"])
    
    # Normalize penalty values for coloring
    norm = Normalize(vmin=penalties_np.min(), vmax=penalties_np.max())
    colors = cmap(norm(penalties_np))

    scatter = ax.scatter(plot_points_x, plot_points_y, plot_points_z, c=colors, marker='o', s=15)

    # Add a color bar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
    cbar.set_label('Penalty Magnitude (rew component)')

    ax.set_xlabel("X world")
    ax.set_ylabel("Y world")
    ax.set_zlabel("Z world")
    ax.set_title(f"3D Cone Penalty Visualization\ncone_r={cone_radius}, cone_h={cone_height_val}")
    
    # Plot target position
    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='blue', marker='x', s=100, label='Target Position')
    
    # Plot cone Z-axis
    rot_target = Rotation.from_quat(target_q.numpy()[1:].tolist() + [target_q.numpy()[0]])
    cone_axis_vec = rot_target.apply([0,0,cone_height_val])
    ax.quiver(target_pos[0], target_pos[1], target_pos[2], 
              cone_axis_vec[0], cone_axis_vec[1], cone_axis_vec[2], 
              length=cone_height_val, color='blue', arrow_length_ratio=0.1, label="Cone Z-axis")

    ax.legend()
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters for the cone penalty (these are from your Gen3SkimmerEnvCfg)
    # Example: self.rewards.cone_penalty = RewTerm(
    #     func=mdp_skimmer.cone_penalty,
    #     weight=-1.0, # The plot shows the 'rew' component, before this weight
    #     params={
    #         "asset_cfg": scene_entity_cfg, # Not directly used in this plot func
    #         "command_name": ee_pose_cmd,   # Not directly used
    #         "cone_h": cone_h, # e.g., 0.30
    #         "cone_r": cone_r, # e.g., 0.10
    #         "delta_r": 1.0,   # This is delta_r_param
    #         "delta_h": 1.0,   # This is delta_h_param
    #     }
    # )
    
    # --- Parameters to configure for plotting ---
    # Target pose in world coordinates
    target_position = [0.0, 0.0, 0.0]  # Example: x, y, z
    target_orientation_wxyz = [1.0, 0.0, 0.0, 0.0] # Example: (w,x,y,z) - identity rotation (cone axis along world Z)
    # To make the cone point along world X axis:
    # target_orientation_wxyz = [0.7071, 0.0, 0.7071, 0.0] # Rotate +90 deg around Y

    # Cone geometric parameters
    cone_base_radius = 0.1  # meters
    cone_actual_height = 0.3 # meters

    # Penalty scaling parameters (initial scalar values from your config)
    delta_r_config_val = 1.0
    delta_h_config_val = 1.0
    
    # Plotting grid parameters
    plot_grid_half_extent = 10 # Plot from target_pos - extent to target_pos + extent
    plot_grid_num_points = 25   # Number of points along each axis of the grid

    plot_cone_penalty_3d(
        target_pos=target_position,
        target_quat_wxyz=target_orientation_wxyz,
        cone_radius=cone_base_radius,
        cone_height_val=cone_actual_height,
        delta_r_val=delta_r_config_val,
        delta_h_val=delta_h_config_val,
        grid_range=plot_grid_half_extent,
        grid_points_per_axis=plot_grid_num_points
    )