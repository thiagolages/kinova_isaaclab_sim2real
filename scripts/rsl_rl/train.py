# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

import platform
from importlib.metadata import version

# if version("rsl-rl-lib") != "2.3.0":
#     if platform.system() == "Windows":
#         cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#     else:
#         cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#     print(
#         f"Please install the correct version of RSL-RL.\nExisting version is: '{version('rsl-rl-lib')}'"
#         " and required version is: '2.3.0'.\nTo install the correct version, run:"
#         f"\n\n\t{' '.join(cmd)}\n"
#     )
#     exit(1)

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--plot_trajectories", action="store_true", default=False, help="Plot end-effector trajectories.")
parser.add_argument("--trajectory_interval", type=int, default=1000, help="Interval between trajectory plots (in iterations).")
parser.add_argument("--enable_cone_visualization", action="store_true", default=False, help="Enable dynamic cone visualization.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import gen3.tasks  # noqa: F401

# Import trajectory plotting functions
from gen3.tasks.manager_based.gen3_skimmer.mdp.plot_trajectory import (
    plot_trajectory_with_cone,
    plot_3d_trajectory_with_cone,
    analyze_trajectory_cone_penalty
)

# Import cone visualizer
from gen3.tasks.manager_based.gen3_skimmer.mdp.cone_visualizer import create_cone_visualizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    
    # This way, the Ray Tune workflow can extract experiment name.
    if agent_cfg.run_name:
        log_dir = f"{agent_cfg.run_name}_"
    else:
        log_dir = ""
    
    # Avoids running different experiments with the same name
    if os.path.exists(os.path.join(log_root_path, log_dir)):
        print(f"[ERROR] Log directory already exists: {os.path.join(log_root_path, log_dir)}")
        exit(1)
    
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join(log_root_path, log_dir)
    print(f"Exact experiment name requested from command line: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Initialize cone visualizer if enabled
    cone_visualizer = None
    if args_cli.enable_cone_visualization:
        print("[INFO] Initializing cone visualizer...")
        # Get cone parameters from environment config
        cone_h = 0.20 * 5  # 20cm * 5 = 1.0m
        cone_r = 0.10 * 5  # 10cm * 5 = 0.5m
        cone_visualizer = create_cone_visualizer(env.unwrapped, cone_r, cone_h)
        print(f"[INFO] Created cone visualizer for {env.unwrapped.num_envs} environments")

    # Initialize trajectory recording if enabled
    trajectory_data = []
    if args_cli.plot_trajectories:
        print("[INFO] Trajectory plotting enabled.")
        # Create trajectory plots directory
        trajectory_dir = os.path.join(log_dir, "trajectories")
        os.makedirs(trajectory_dir, exist_ok=True)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # Custom training loop with trajectory recording and cone visualization
    def custom_learn(num_learning_iterations: int, init_at_random_ep_len: bool = True):
        """Custom learning function with trajectory recording and cone visualization."""
        
        # Initialize trajectory recording variables
        current_trajectory = {
            'end_effector_positions': [],
            'target_positions': [],
            'target_orientations': [],
            'iteration': 0
        }
        
        # Get cone parameters
        cone_h = 0.20 * 5  # 20cm * 5 = 1.0m
        cone_r = 0.10 * 5  # 10cm * 5 = 0.5m
        
        for iteration in range(num_learning_iterations):
            # Run one iteration of training
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=init_at_random_ep_len)
            
            # Update cone visualization if enabled
            if cone_visualizer is not None:
                try:
                    # Get current target positions from the environment
                    command_manager = env.unwrapped.command_manager
                    command_name = "ee_pose"
                    command = command_manager.get_command(command_name)
                    
                    # Extract target positions and orientations
                    target_positions = command[:, :3]  # (num_envs, 3)
                    target_orientations = command[:, 3:7]  # (num_envs, 4)
                    
                    # Update cone positions
                    cone_visualizer.update_cones(target_positions, target_orientations)
                    
                except Exception as e:
                    print(f"[WARNING] Failed to update cone visualization: {e}")
            
            # Record trajectory data if enabled
            if args_cli.plot_trajectories and iteration % args_cli.trajectory_interval == 0:
                try:
                    # Get end-effector positions for all environments
                    robot = env.unwrapped.scene["robot"]
                    eff_link_id = robot.body_names.index("end_effector_link")
                    ee_positions = robot.data.body_state_w[:, eff_link_id, :3]  # (num_envs, 3)
                    
                    # Get target positions
                    command_manager = env.unwrapped.command_manager
                    command_name = "ee_pose"
                    command = command_manager.get_command(command_name)
                    target_positions = command[:, :3]  # (num_envs, 3)
                    target_orientations = command[:, 3:7]  # (num_envs, 4)
                    
                    # Record trajectory for first environment (for visualization)
                    current_trajectory['end_effector_positions'].append(ee_positions[0].cpu().numpy())
                    current_trajectory['target_positions'].append(target_positions[0].cpu().numpy())
                    current_trajectory['target_orientations'].append(target_orientations[0].cpu().numpy())
                    current_trajectory['iteration'] = iteration
                    
                    # Create plots if we have enough data
                    if len(current_trajectory['end_effector_positions']) >= 50:  # Minimum trajectory length
                        print(f"[INFO] Creating trajectory plots for iteration {iteration}...")
                        
                        # Convert to tensors
                        ee_positions_tensor = torch.tensor(np.array(current_trajectory['end_effector_positions']))
                        target_pos_tensor = torch.tensor(np.array(current_trajectory['target_positions'][-1]))  # Use last target
                        target_quat_tensor = torch.tensor(np.array(current_trajectory['target_orientations'][-1]))  # Use last target
                        
                        # Create plots
                        plot_trajectory_with_cone(
                            ee_positions_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h, 
                            save_path=os.path.join(trajectory_dir, f"trajectory_2d_iter_{iteration}.png"),
                            show_plot=False
                        )
                        
                        plot_3d_trajectory_with_cone(
                            ee_positions_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h,
                            save_path=os.path.join(trajectory_dir, f"trajectory_3d_iter_{iteration}.png"),
                            show_plot=False
                        )
                        
                        # Analyze trajectory
                        analysis = analyze_trajectory_cone_penalty(
                            ee_positions_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h
                        )
                        
                        # Save analysis
                        analysis_file = os.path.join(trajectory_dir, f"trajectory_analysis_iter_{iteration}.txt")
                        with open(analysis_file, 'w') as f:
                            f.write(f"Trajectory Analysis for Iteration {iteration}\n")
                            f.write("=" * 50 + "\n")
                            f.write(f"Total steps: {analysis['total_steps']}\n")
                            f.write(f"Steps inside cone: {analysis['steps_inside_cone']}\n")
                            f.write(f"Percentage inside cone: {analysis['percentage_inside_cone']:.2f}%\n")
                            f.write(f"Average distance to boundary: {analysis['avg_distance_to_boundary']:.4f}\n")
                        
                        print(f"[INFO] Trajectory analysis saved to {analysis_file}")
                        
                        # Reset trajectory data for next recording
                        current_trajectory = {
                            'end_effector_positions': [],
                            'target_positions': [],
                            'target_orientations': [],
                            'iteration': iteration
                        }
                        
                except Exception as e:
                    print(f"[WARNING] Failed to record trajectory data: {e}")
        
        # Cleanup cone visualizer
        if cone_visualizer is not None:
            cone_visualizer.cleanup()

    # run training with custom learning function
    if args_cli.plot_trajectories or args_cli.enable_cone_visualization:
        print("[INFO] Using custom training loop with visualization features.")
        custom_learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    else:
        # Use standard training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
