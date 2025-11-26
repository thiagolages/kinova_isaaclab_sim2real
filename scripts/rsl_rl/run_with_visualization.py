#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run training and testing with cone visualization and trajectory plotting.

This script demonstrates how to use the new features:
1. Transparent cone visualization during training and testing
2. YZ and XZ trajectory plotting with cone boundaries
3. Trajectory analysis and statistics

Usage:
    # Training with trajectory plotting
    python run_with_visualization.py --mode train --plot_trajectories --task Gen3-Skimmer-v0
    
    # Testing with trajectory plotting and saving
    python run_with_visualization.py --mode test --plot_trajectories --save_trajectory --task Gen3-Skimmer-v0
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Run training or testing with visualization features.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, 
                       help="Mode to run: train or test")
    parser.add_argument("--task", type=str, default="Gen3-Skimmer-v0", 
                       help="Task name to run")
    parser.add_argument("--plot_trajectories", action="store_true", 
                       help="Enable trajectory plotting")
    parser.add_argument("--save_trajectory", action="store_true", 
                       help="Save trajectory data to file (test mode only)")
    parser.add_argument("--video", action="store_true", 
                       help="Record videos during execution")
    parser.add_argument("--real_time", action="store_true", 
                       help="Run in real-time (test mode only)")
    parser.add_argument("--checkpoint", type=str, 
                       help="Checkpoint path for testing")
    parser.add_argument("--num_envs", type=int, 
                       help="Number of environments")
    parser.add_argument("--max_iterations", type=int, 
                       help="Maximum training iterations")
    
    args = parser.parse_args()
    
    # Build command based on mode
    if args.mode == "train":
        cmd = [
            "python", "train.py",
            "--task", args.task
        ]
        
        if args.plot_trajectories:
            cmd.extend(["--plot_trajectories"])
        
        if args.video:
            cmd.extend(["--video"])
        
        if args.num_envs:
            cmd.extend(["--num_envs", str(args.num_envs)])
        
        if args.max_iterations:
            cmd.extend(["--max_iterations", str(args.max_iterations)])
            
    elif args.mode == "test":
        cmd = [
            "python", "play_skimmer.py",
            "--task", args.task
        ]
        
        if args.plot_trajectories:
            cmd.extend(["--plot_trajectories"])
        
        if args.save_trajectory:
            cmd.extend(["--save_trajectory"])
        
        if args.video:
            cmd.extend(["--video"])
        
        if args.real_time:
            cmd.extend(["--real-time"])
        
        if args.checkpoint:
            cmd.extend(["--checkpoint", args.checkpoint])
        
        if args.num_envs:
            cmd.extend(["--num_envs", str(args.num_envs)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"Command completed successfully with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Could not find the script to run. Make sure you're in the correct directory.")
        sys.exit(1)

if __name__ == "__main__":
    main() 