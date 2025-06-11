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

# Needed to import the utility functions
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
from gen3.tasks.manager_based.gen3_skimmer.mdp.rewards import (
    dist_eff_to_target_z_axis,
    dist_eff_to_target_xy_plane
)

# Adapted from gen3.tasks.manager_based.gen3_skimmer.mdp.rewards so it doesn't
# require the env and asset_cfg, but takes the target pose and effector pose directly.
def cone_penalty(
        eff_pos_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        cone_r: float = None,
        cone_h: float = None,
        delta_r: float = None,
        delta_h: float = None,
        weight: float = -1.0  # Default weight for the penalty
    ) -> torch.Tensor:
    """Penalize the end-effector for being outside the target area."""
    # Check that cone_r and cone_h are provided
    if cone_r is None or cone_h is None or cone_r <= 0 or cone_h <= 0:
        raise ValueError("cone_r and cone_h must be provided.")

    # Check that delta_r and delta_h are positive
    if delta_r is None or delta_h is None or delta_r <= 0 or delta_h <= 0:
        raise ValueError("delta_r and delta_h must be provided, and positive.")

    print("cone_r =", cone_r)
    print("cone_h =", cone_h)
    print("delta_r =", delta_r)
    print("delta_h =", delta_h)
    ### 1. Get target and effector poses
    
    ## 1.1. Get robot articulation and command
    # Removed asset_cfg and command_name as parameters
    # asset: Articulation = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    
    ## 1.2. Get target pose
    # target_pos_b = command[:, :3] # (num_envs, 3)
    # target_pos_w, target_quat_w = combine_frame_transforms(
    #     asset.data.root_state_w[:, :3],
    #     asset.data.root_state_w[:, 3:7],
    #     target_pos_b
    # ) # (num_envs, 3)
    
    ## 1.3. Get effector pose
    # asset_cfg.body_ids[0] should be the index of the end-effector link
    #eff_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3] # (num_envs, 3) # type: ignore

    ### 2. Calculate cone's parameters

    ## 2.1. Calculate theta = 2 * atan(r/h)
    theta = 2 * math.degrees(math.atan2(cone_r, cone_h))
    
    ## 2.2. Compute the effector height with respect to the target XY plane
    eff_cone_height = dist_eff_to_target_xy_plane(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    print("eff_cone_height = ", eff_cone_height)
    ## 2.3. Compute the effector distance to the cone Z axis
    dist_to_cone_axis = dist_eff_to_target_z_axis(eff_pos_w, (target_pos_w, target_quat_w)) # (num_envs,)
    print("dist_to_cone_axis = ", dist_to_cone_axis)

    ## 2.4. Calculate the radius at the current height as r = h * tan(theta/2)
    r_at_current_h = torch.maximum(
        eff_cone_height * math.tan(math.radians(theta / 2.0)),
        torch.tensor(1e-3, device=eff_cone_height.device, dtype=eff_cone_height.dtype)
    ) # (num_envs,)
    print("r_at_current_h = ", r_at_current_h)

    ### 3. Define multipliers

    ## 3.1. delta_r scales penalty by how close the effector is to the center (r = 0), not normalized
    # Normalizes delta_r based on curent radius, so that this penalty doesn't diminish as we
    # approach the target.
    delta_r = delta_r / r_at_current_h # (num_envs,)
    print("delta_r = ", delta_r)

    ## 3.2. delta_h scales penalty by how close we are from the target, in the Z direction of the cone (its height)
    delta_h = delta_h * (cone_h - eff_cone_height) / cone_h # (num_envs,)
    print("delta_h = ", delta_h)
    # TODO: check if needs to be normalized

    ### 4. Check if effector is inside penalty cone
    # Don't penalize if outside cone max radius or max height
    # eff_cone_height: (num_envs,), cone_h: scalar
    mask = (dist_to_cone_axis > r_at_current_h) | (eff_cone_height > cone_h) | (eff_cone_height <= 0)
    r_error = torch.where(mask, torch.zeros_like(dist_to_cone_axis), r_at_current_h - dist_to_cone_axis)
    num_nonzero = torch.count_nonzero(mask)
    percentage_nonzero = 100.0 * num_nonzero.item() / mask.numel()
    print(f"Mask: {mask}")
    print(f"Mask: {num_nonzero.item()} out of {mask.numel()} ({percentage_nonzero:.2f}%) are non-zero (True)")
    print("r_error = ", r_error)
    # if dist_to_cone_axis > r_at_current_h or eff_cone_height > cone_h:
    #     r_error = 0
    # else:
    #     r_error = r_at_current_h - dist_to_cone_axis # (num_envs,)
    
    ### 4. Define final reward
    # This reward will be multiplied by the `weight` (some negative number)
    rew = weight * delta_r * delta_h * r_error

    print("rew = ", rew)

    # compute the reward
    # result = torch.sum(rew, dim=1)
    result = rew.item()

    print("cone_penalty result = ", result)

    # Check for NaNs
    # if torch.isnan(result).any():
    #     raise ValueError("Reward contains NaN values.")
    # # Check for unwanted shape (should be 1D: (num_envs,))
    # if result.ndim != 1:
    #     raise ValueError(f"Reward has unexpected shape: {result.shape}")

    print("Returning result = ", result)

    return result

## --- Helper function to calculate penalty for a single point ---
# def calculate_single_point_penalty(
#     eff_pos_w: torch.Tensor,       # Current point in world (1D tensor with 3 elements)
#     target_pos_w: torch.Tensor,    # Target position in world (1D tensor with 3 elements)
#     target_quat_w: torch.Tensor,   # Target orientation in world (w,x,y,z) (1D tensor with 4 elements)
#     cone_r: float,                 # Cone radius at base
#     cone_h: float,                 # Cone height
#     delta_r_param: float,          # Scalar parameter for radial penalty scaling
#     delta_h_param: float           # Scalar parameter for height penalty scaling
# ) -> float:
#     """
#     Calculates the penalty magnitude for a single 3D point based on the cone_penalty logic.
#     Returns a scalar penalty value (rew component before weight multiplication).
#     """
#     if cone_r <= 0 or cone_h <= 0:
#         raise ValueError("cone_r and cone_h must be positive.")
#     if delta_r_param <= 0 or delta_h_param <= 0:
#         raise ValueError("delta_r_param and delta_h_param must be positive.")

#     # Ensure inputs are torch tensors
#     eff_pos_w = torch.as_tensor(eff_pos_w, dtype=torch.float32)
#     target_pos_w = torch.as_tensor(target_pos_w, dtype=torch.float32)
#     target_quat_w = torch.as_tensor(target_quat_w, dtype=torch.float32)

#     # 1. Calculate target frame axes in world
#     # Scipy Rotation uses (x,y,z,w) for quaternion
#     rot = Rotation.from_quat(target_quat_w.numpy()[1:] # x,y,z
#                              .tolist() + [target_quat_w.numpy()[0]]) # w
    
#     target_z_axis_w = torch.tensor(rot.apply([0., 0., 1.]), dtype=torch.float32)

#     # 2. Calculate geometric quantities relative to the target frame
#     vec_eff_to_target_origin = eff_pos_w - target_pos_w
    
#     # Effector height with respect to the target XY plane (along target_z_axis_w)
#     eff_cone_height = torch.dot(vec_eff_to_target_origin, target_z_axis_w)
    
#     # Effector distance to the cone's Z-axis
#     projection_on_z_axis = eff_cone_height * target_z_axis_w
#     vec_in_xy_plane = vec_eff_to_target_origin - projection_on_z_axis
#     dist_to_cone_axis = torch.norm(vec_in_xy_plane)

#     # 3. Calculate cone's parameters at current height
#     # theta = 2 * atan(r/h)
#     theta_rad = 2 * math.atan2(cone_r, cone_h) # math.atan2 takes (y,x)
    
#     # Radius of the cone at the current effector height: r = h_eff * tan(theta/2)
#     # Ensure r_at_current_h is positive and non-zero
#     # eff_cone_height can be negative if below the cone base plane
#     r_at_current_h_val = eff_cone_height * math.tan(theta_rad / 2.0)
#     r_at_current_h = torch.maximum(
#         r_at_current_h_val,
#         torch.tensor(1e-6, dtype=torch.float32) # Min radius to avoid division by zero
#     )
#     if eff_cone_height < 0 : # If below the cone base, effectively infinite radius for penalty calc, or treat as outside
#         r_at_current_h = torch.tensor(float('inf'), dtype=torch.float32)


#     # 4. Define multipliers
#     # delta_r scales penalty by how close the effector is to the center
#     current_delta_r = delta_r_param / r_at_current_h
#     if torch.isinf(current_delta_r): # Handle case where r_at_current_h was inf
#         current_delta_r = torch.tensor(0.0, dtype=torch.float32)


#     # delta_h scales penalty by how close we are from the target, in the Z direction
#     current_delta_h = delta_h_param * torch.clamp((cone_h - eff_cone_height) / cone_h, min=0.0)
#     # Clamp to ensure non-negative if eff_cone_height > cone_h

#     # 5. Check if effector is inside penalty cone region for error calculation
#     # Mask is true if outside the cone's slope OR above the cone's max height
#     # or below the cone's base (eff_cone_height < 0)
#     is_above_cone = eff_cone_height > cone_h
#     is_below_cone_base = eff_cone_height < 0
    
#     mask = (dist_to_cone_axis > r_at_current_h) | is_above_cone | is_below_cone_base
    
#     r_error = torch.where(
#         mask,
#         torch.tensor(0.0, dtype=torch.float32),
#         r_at_current_h - dist_to_cone_axis # Penalty increases as dist_to_cone_axis gets smaller than r_at_current_h
#     )
    
#     # 6. Define final reward component (before weight)
#     rew = -1.0 * current_delta_r * current_delta_h * r_error
    
#     return rew.item()


# --- Main plotting function ---
def plot_cone_penalty_3d(
    target_pos: list[float],
    target_quat_wxyz: list[float], # (w,x,y,z)
    cone_radius: float,
    cone_height_val: float,
    delta_r_val: float,
    delta_h_val: float,
    grid_range: float = 1.0, # Max extent from target_pos for plotting grid
    grid_points_per_axis: int = 10
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    target_pos_w = torch.tensor([target_pos], dtype=torch.float32)
    target_quat_w = torch.tensor([target_quat_wxyz], dtype=torch.float32)

    t = np.linspace(0, 2 * np.pi, grid_points_per_axis)  # Parameter for circle
    x_center = float(target_pos_w[0, 0])
    y_center = float(target_pos_w[0, 1])
    z_center = float(target_pos_w[0, 2])
    x_coords = cone_radius * np.cos(t) + x_center
    y_coords = cone_radius * np.sin(t) + y_center  # Circle in XY plane
    z_coords = np.linspace(z_center - 0.3, z_center + 0.30, grid_points_per_axis)

    plot_points_x = []
    plot_points_y = []
    plot_points_z = []
    penalty_values = []

    for x in x_coords:
        for y in y_coords:
            for z in z_coords:
                eff_pos_w = torch.tensor([[x, y, z]], dtype=torch.float32) # (1, 3) tensor to match expected input shape
                print("#######################################################################")
                print(f"Calculating penalty for point: ({x}, {y}, {z})")
                penalty = cone_penalty(
                    eff_pos_w, target_pos_w, target_quat_w,
                    cone_radius, cone_height_val, delta_r_val, delta_h_val
                )
                print(f"Penalty: {penalty}")
                # print(f"Calculated penalty: {penalty.item()} for point ({x}, {y}, {z})")
                if abs(penalty) > 1e-5: # Only plot if penalty is significantly non-zero
                    plot_points_x.append(x)
                    plot_points_y.append(y)
                    plot_points_z.append(z)
                    penalty_values.append(penalty)

    print("penalty_values = ", penalty_values)
    if not penalty_values:
        print("No points with significant penalty found to plot.")
        ax.set_title("Cone Penalty Visualization (No penalties > 0)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
        return

    # Print top 10 highest and lowest penalty values
    sorted_penalties = sorted(penalty_values)
    print("Top 10 lowest penalty values:", sorted_penalties[:10])
    print("Top 10 highest penalty values:", sorted_penalties[-10:])

    penalties_np = np.array(penalty_values)
    
    # Create a colormap: green (low penalty) -> yellow -> red (high penalty)
    cmap = LinearSegmentedColormap.from_list("penalty_cmap", ["green", "yellow", "red"])
    
    # Normalize penalty values for coloring
    # If penalties are negative, min will be most negative, max closest to 0
    norm = Normalize(vmin=penalties_np.min(), vmax=penalties_np.max())
    # cmap green -> red. If min is very negative, green is high penalty.
    # To make red = high penalty (most negative), you can invert the normalized values
    # or flip the colormap, or map -penalties_np.

    # Option 1: Map -penalties to make positive values for standard cmap
    # positive_penalties = -penalties_np 
    # norm = Normalize(vmin=positive_penalties.min(), vmax=positive_penalties.max())
    # colors = cmap(norm(positive_penalties))
    # cbar.set_label('Penalty Magnitude (-rew)')

    # Option 2: Use original penalties and ensure cmap reflects desired colors
    # If penalties are e.g. -50 (high) to -1 (low)
    # Green (low value in cmap) will be -50. Red (high value in cmap) will be -1.
    # This is likely what you want: more negative = greener. Less negative = redder.
    # If you want the opposite (more negative = redder), then:
    cmap = LinearSegmentedColormap.from_list("penalty_cmap", ["red", "yellow", "green"]) # Red for min (most negative)
    norm = Normalize(vmin=penalties_np.min(), vmax=penalties_np.max())
    colors = cmap(norm(penalties_np))

    scatter = ax.scatter(plot_points_x, plot_points_y, plot_points_z, c=colors, marker='o', s=15)

    # Add a color bar
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.1)
    cbar.set_label('Penalty Value (rew)')

    ax.set_xlabel("X world")
    ax.set_ylabel("Y world")
    ax.set_zlabel("Z world")
    ax.set_title(f"3D Cone Penalty Visualization\ncone_r={cone_radius}, cone_h={cone_height_val}")
    
    # Plot target position
    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='blue', marker='x', s=100, label='Target Position')
    
    # Plot cone Z-axis
    # target_quat_w is shape (1, 4) representing [[w, x, y, z]]
    # Scipy's Rotation.from_quat expects [x, y, z, w]
    quat_for_scipy = target_quat_w.numpy()[0, [1, 2, 3, 0]] # Extracts [x, y, z, w] from the first (and only) row
    rot_target = Rotation.from_quat(quat_for_scipy)
    
    # The cone's axis in its local frame is (0,0,1) if its USD 'axis' prop is 'Z'
    # We want to visualize its length as cone_height_val
    local_cone_axis = np.array([0.0, 0.0, 1.0]) 
    cone_axis_world_direction = rot_target.apply(local_cone_axis) # Direction vector in world

    # Quiver plots from a point (target_pos) along a direction vector
    # The actual length of the plotted arrow is controlled by the 'length' param of quiver,
    # but it's good practice for cone_axis_vec to represent the actual displacement if used elsewhere.
    # For visualization, we just need the direction.
    ax.quiver(target_pos[0], target_pos[1], target_pos[2], 
              cone_axis_world_direction[0], cone_axis_world_direction[1], cone_axis_world_direction[2], 
              length=cone_height_val, color='blue', arrow_length_ratio=0.1, label="Cone Z-axis (Tip to Base)")

    ax.legend()
    plt.tight_layout()
    plt.savefig("cone_penalty_plot.png")
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
    plot_grid_half_extent = 0.3 # Plot from target_pos - extent to target_pos + extent
    plot_grid_num_points = 30   # Number of points along each axis of the grid

    plot_cone_penalty_3d(
        target_pos=target_position,
        target_quat_wxyz=target_orientation_wxyz,
        cone_radius=cone_base_radius,
        cone_height_val=cone_actual_height,
        delta_r_val=delta_r_config_val,
        delta_h_val=delta_h_config_val,
        # grid_range=plot_grid_half_extent,
        grid_points_per_axis=plot_grid_num_points
    )