# Skimmer Task Visualization Features

This document describes the new visualization features added to the Gen3 Skimmer task for analyzing robot motion and cone penalty avoidance.

## Features

### 1. Transparent Cone Visualization

A transparent green cone is now visualized in the simulation environment to show the penalty region that the robot should avoid. The cone represents the area where the robot gets penalized for approaching the target from above.

**Configuration:**
- **Cone Height:** 1.0m (0.20m * 5)
- **Cone Radius:** 0.5m (0.10m * 5)
- **Color:** Green with 40% opacity
- **Position:** Fixed at (0.5, 0.0, 0.0) in world coordinates

The cone is automatically added to the scene during both training and testing phases.

### 2. Trajectory Plotting

Two types of trajectory plots are available:

#### 2D Projections (YZ and XZ)
- **YZ Projection:** Shows end-effector motion in the Y-Z plane relative to the target
- **XZ Projection:** Shows end-effector motion in the X-Z plane relative to the target
- **Cone Boundaries:** Red dashed lines show the cone penalty region
- **Color-coded:** Trajectory points are colored by time step

#### 3D Trajectory
- **Full 3D view:** Shows complete end-effector trajectory in 3D space
- **Cone surface:** Semi-transparent red cone surface shows the penalty region
- **Interactive:** Can be rotated and zoomed for detailed analysis

### 3. Trajectory Analysis

The system provides quantitative analysis of trajectory performance:

- **Total steps:** Number of trajectory points recorded
- **Steps inside cone:** How many points were inside the penalty region
- **Percentage inside cone:** Percentage of trajectory inside the penalty region
- **Average distance to boundary:** Average distance to cone boundary when inside

## Usage

### Training with Visualization

```bash
# Basic training
python train.py --task Gen3-Skimmer-v0

# Training with trajectory plotting
python train.py --task Gen3-Skimmer-v0 --plot_trajectories

# Training with video recording
python train.py --task Gen3-Skimmer-v0 --video --video_length 200
```

### Testing with Visualization

```bash
# Basic testing
python play_skimmer.py --task Gen3-Skimmer-v0

# Testing with trajectory plotting
python play_skimmer.py --task Gen3-Skimmer-v0 --plot_trajectories

# Testing with trajectory saving
python play_skimmer.py --task Gen3-Skimmer-v0 --save_trajectory

# Testing with both plotting and saving
python play_skimmer.py --task Gen3-Skimmer-v0 --plot_trajectories --save_trajectory

# Real-time testing
python play_skimmer.py --task Gen3-Skimmer-v0 --real-time --plot_trajectories
```

### Using the Convenience Script

```bash
# Training with visualization
python run_with_visualization.py --mode train --plot_trajectories --task Gen3-Skimmer-v0

# Testing with visualization
python run_with_visualization.py --mode test --plot_trajectories --save_trajectory --task Gen3-Skimmer-v0
```

## Output Files

When trajectory plotting is enabled, the following files are generated:

### Training Mode
- `logs/rsl_rl/skimmer_gen3/{timestamp}/trajectories/` - Trajectory plots and data

### Testing Mode
- `logs/rsl_rl/skimmer_gen3/{timestamp}/trajectory_plots/trajectory_2d.png` - 2D projections
- `logs/rsl_rl/skimmer_gen3/{timestamp}/trajectory_plots/trajectory_3d.png` - 3D trajectory
- `logs/rsl_rl/skimmer_gen3/{timestamp}/trajectory_plots/trajectory_analysis.txt` - Analysis results
- `logs/rsl_rl/skimmer_gen3/{timestamp}/trajectory_data.npz` - Raw trajectory data (if --save_trajectory)

## Interpreting Results

### Good Performance Indicators
- **Low percentage inside cone:** < 20% indicates good cone avoidance
- **Smooth trajectory:** Continuous motion without sudden jumps
- **Approach from below:** Trajectory approaches target from lower Z positions

### Poor Performance Indicators
- **High percentage inside cone:** > 50% indicates poor cone avoidance
- **Discontinuous motion:** Sudden jumps or jerky movements
- **Approach from above:** Trajectory approaches target from higher Z positions

### Analysis Metrics
- **Percentage inside cone:** Primary metric for cone avoidance performance
- **Average distance to boundary:** How close the robot gets to the cone boundary
- **Trajectory smoothness:** Visual inspection of trajectory plots

## Technical Details

### Cone Parameters
The cone penalty region is defined by:
- **Height:** 1.0m (scaled from 0.20m)
- **Base Radius:** 0.5m (scaled from 0.10m)
- **Shape:** Linear interpolation from base to tip

### Coordinate System
- **Origin:** Target position (goal pose)
- **X-axis:** Forward direction
- **Y-axis:** Left direction  
- **Z-axis:** Up direction

### Plotting Libraries
- **Matplotlib:** 2D and 3D plotting
- **NumPy:** Numerical computations
- **Torch:** Tensor operations

## Troubleshooting

### Common Issues

1. **Plots not showing:**
   - Ensure matplotlib backend is properly configured
   - Check if display is available (for headless systems)

2. **Missing trajectory data:**
   - Verify end-effector link name is correct ("end_effector_link")
   - Check if robot data is accessible

3. **Cone visualization not appearing:**
   - Ensure scene configuration is loaded correctly
   - Check USD material properties

### Debugging

Enable verbose output to debug issues:
```bash
python play_skimmer.py --task Gen3-Skimmer-v0 --plot_trajectories --verbose
```

## Future Enhancements

Potential improvements to consider:
- **Dynamic cone positioning:** Cone follows target position
- **Multiple cone visualization:** Show different penalty regions
- **Real-time plotting:** Live trajectory updates during simulation
- **Statistical analysis:** More detailed performance metrics
- **Export functionality:** Export plots in different formats 