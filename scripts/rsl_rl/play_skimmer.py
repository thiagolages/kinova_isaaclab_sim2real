# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import platform
from importlib.metadata import version

#if version("rsl-rl-lib") != "2.3.0":
#    if platform.system() == "Windows":
#        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#    else:
#        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", "rsl-rl-lib==2.3.0"]
#    print(
#        f"Please install the correct version of RSL-RL.\nExisting version is: '{version('rsl-rl-lib')}'"
#        " and required version is: '2.3.0'.\nTo install the correct version, run:"
#        f"\n\n\t{' '.join(cmd)}\n"
#    )
#    exit(1)

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Gen3-Skimmer-v0", help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--plot-trajectories", action="store_true", default=False, help="Plot end-effector trajectories.")
parser.add_argument("--save-trajectory", action="store_true", default=False, help="Save trajectory data to file.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import gen3.tasks  # noqa: F401

# Import trajectory plotting functions
from gen3.tasks.manager_based.gen3_skimmer.mdp.plot_trajectory import (
    plot_trajectory_with_cone,
    plot_3d_trajectory_with_cone,
    analyze_trajectory_cone_penalty
)
from isaaclab.utils.math import (
    combine_frame_transforms,
)

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    print(f"[INFO] Log directory: {log_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(ppo_runner.alg.policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # TEMPORARY TO SLOW DOWN THE SIMULATION
    dt = env.unwrapped.physics_dt * 5

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # Define your home pose (adjust as needed)
    # home_pos = torch.tensor([0.5, 0.2, 0.50], device=env.unwrapped.device)  # Example position
    # home_quat = torch.tensor([ 0.0, 0.0, 0.0,  1.0], device=env.unwrapped.device)  # Z pointing to the side (rotation of 90deg around X) (xyzw)
    # home_pose = torch.cat([home_pos, home_quat])  # Shape (7,)

    # sub = torch.tensor([0.08, 0.51, 0.35], device=env.unwrapped.device)  # Example position
    # skimmer_pos = home_pos - sub
    # skimmer_quat = torch.tensor([ 0.0, 0.0, 0.0,  1.0], device=env.unwrapped.device)  # Z pointing to the side (rotation of 90deg around X) (xyzw)
    # skimmer_pose = torch.cat([skimmer_pos, skimmer_quat])  # Shape (7,)

    # 0.4, -0.1, 0.2
    #####
    skimmer_pos = torch.tensor([0.5 - 0.08, 0.2 - 0.51, 0.50 - 0.35], device=env.unwrapped.device)  # Example position
    skimmer_quat = torch.tensor([ 0.0, 0.0, 0.0,  1.0], device=env.unwrapped.device)  # Z pointing to the side (rotation of 90deg around X) (xyzw)
    skimmer_pose = torch.cat([skimmer_pos, skimmer_quat])  # Shape (7,)
    
    # home_pos = skimmer_pos + torch.tensor([-0.1, -0.32, 0.20], device=env.unwrapped.device)  # Example position
    #home_pos = torch.tensor([0.3, 0.2, 0.35], device=env.unwrapped.device)  # Good example
    # home_pos = torch.tensor([0.5, 0.2-0.8, 0.35], device=env.unwrapped.device)  # Good example
    # home_pos = torch.tensor([0.5274, 0.1875-0.4+0.4, 0.625], device=env.unwrapped.device)  # Good example (new)
    home_pos = torch.tensor([0.4274-0.05, 0.1875-0.7, 0.325], device=env.unwrapped.device)  # Good example (new)
    # home_quat = torch.tensor([ 0.7071068, -0.7071068, 0.0,  0.0], device=env.unwrapped.device) # -90deg X (wxyz)
    home_quat = torch.tensor([ 0.7071068, 0.0, 0.7071068,  0.0], device=env.unwrapped.device) # 90deg Y (wxyz)
    # home_quat = torch.tensor([ 1.0, 0.0, 0.0,  0.0], device=env.unwrapped.device) # identity
    home_pose = torch.cat([home_pos, home_quat])  # Shape (7,)
    #####


    reach_threshold = 0.03  # meters

    # Usually 'ee_pose', adjust if your command name is different
    command_name = "ee_pose"
    cmd_manager = env.unwrapped.command_manager

    home_timeout = 4 # seconds to reach home
    set_time_left = False  # Flag to check if time_left is set
    
    # Initialize trajectory recording
    trajectory_data = []
    end_effector_positions = []
    end_effector_orientations = []
    actions_list = []
    target_positions = []
    target_orientations = []
    actions = None

    # Get cone parameters from environment config
    cone_h = 0.20 #* 5  # 20cm * 5 = 1.0m
    cone_r = 0.10 # 10cm * 5 = 0.5m
    sphere_r = None #0.05  # 15cm sphere radius (same as sphere_penalty_r in env config)
    plot_traj = False
    target_changed = False
    old_cmd_buf = None
    cmd_buf = None
    target_pos_w = None
    count = 0
    idx = 0
    target_str = "NONE"
    aux_cmd_buf = home_pose

    # simulate environment
    while simulation_app.is_running():
        
        start_time = time.time()
        
        term = cmd_manager.get_term(command_name)
        # Get end-effector position and orientation
        robot = env.unwrapped.scene["robot"]

# If we want to extract joint positions from a saved file with
# positions and quaternions
###############################################################################
        # import numpy as np

        # # Load trajectory data from npz file
        # traj = np.load("rl_saved_data_nov_25/10_trajectory_data.npz")
        # ee_positions = traj["target_positions"] #traj["end_effector_positions"]  # shape (N, 3)
        # ee_orientations = traj["target_orientations"] #traj["end_effector_orientations"]  # shape (N, 4)

        # # Prepare storage for IK results
        # num_points = ee_positions.shape[0]
        # rl_q = []

        # prev_q = None
        # for i in range(num_points):
        #     pos = ee_positions[i]
        #     quat = ee_orientations[i]
        #     # Use previous q as initial guess when possible
        #     initial_guess = prev_q
        #     q_sol = robot.inverse_kinematics(
        #         position=pos, 
        #         orientation=quat,
        #         initial_guess=initial_guess
        #     )
        #     q_np = q_sol.cpu().numpy() if hasattr(q_sol, "cpu") else q_sol
        #     rl_q.append(q_np)
        #     prev_q = q_np  # Update for next iteration

        # rl_q = np.array(rl_q)  # shape (N, num_joints)
        # np.savez("rl_q.npz", rl_q=rl_q)

        # exit()
###############################################################################


        eff_link_id = robot.body_names.index("end_effector_link")

        if cmd_buf is None:
            cmd_buf = cmd_manager.get_command(command_name)
            aux_cmd_buf = home_pose
            plot_traj = False
            # Store new target
            current_target_pos_b = cmd_buf[0, :3]
            current_target_quat_b = cmd_buf[0, 3:7]

            # Get robot base pose
            root_pos_w = robot.data.root_state_w[0, :3]
            root_quat_w = robot.data.root_state_w[0, 3:7]

            # Convert target pose to world frame
            target_pos_w, target_quat_w = combine_frame_transforms(
                root_pos_w,
                root_quat_w,
                current_target_pos_b,
                current_target_quat_b
            )  # (num_envs, 3)

        print(f"time_left = {term.time_left[0]}")

        if term.time_left[0] <= 0.1:
            # set_time_left = False  # Reset flag if time is left
        
            idx += 1
            print("BEFORE")
            print(f"time_left = {term.time_left[0]}")
            print(f"idx = {idx}")
            term.time_left[:] = home_timeout
            if idx % 2 == 0:
                aux_cmd_buf = home_pose
                plot_traj = False
                target_str = "HOME"
            else:
                aux_cmd_buf = skimmer_pose
                plot_traj = True
                target_str = "SKIMMER"
            
            print("AFTER")
            print(f"time_left = {term.time_left[0]}")
            print(f"cmd_buf = {cmd_buf[0]}")
            print(f"plot_traj = {plot_traj}")

            # set_time_left = True  # Set time_left only once

        # Always replace cmd_buf with aux_cmd_buf
        cmd_buf[:] = aux_cmd_buf

        # go_home = term.command_counter[0] % 2 == 0
        
        # if go_home:
        #     # plot_traj = False
        #     cmd_buf[:] = home_pose          # broadcast to every parallel env
        #     # print("cmd_buf[0] =", cmd_buf[0])
            
        #     # if set_time_left == False:
        #     #     term.time_left[:] = home_timeout
        #     #     set_time_left = True  # Set time_left only once

        # # DO NOT ENFORCE SPECIFIC POSE
        # else:
        #     #TEMPORARY:go to desired pose
        #     cmd_buf[:,:] = skimmer_pose          # broadcast to every parallel env
        #     # if set_time_left == False:
        #     #     term.time_left[:] = home_timeout
        #     #     set_time_left = True  # Set time_left only once


        #     # # always keep the skimmer at a height of 0.30m or lower
        #     # for i in range(args_cli.num_envs):
        #     #     if cmd_buf[i, 2] > 0.30:
        #     #         cmd_buf[i, 2] = float(torch.empty(1).uniform_(0.1, 0.3).item())

        # else:
        #     plot_traj = True
        #     cmd_buf = cmd_manager.get_command(command_name)
        #     cmd_buf[:] = skimmer_pose          # broadcast to every parallel env
        #     # print("cmd_buf[0] =", cmd_buf[0])
        #     if set_time_left == False:
        #         term.time_left[:] = home_timeout
        #         set_time_left = True  # Set time_left only once
        
        
        # else:
        #     set_time_left = False  # Reset flag for next command

        # print(f"old_cmd_buf = {old_cmd_buf}")
        # print(f"cmd_buf[0] = {cmd_buf[0]}")

        # if old_cmd_buf is not None:
        #     print(f"torch.allclose(old_cmd_buf, cmd_buf[0]) = {torch.allclose(old_cmd_buf, cmd_buf[0])}")
        # else:
        #     print(f"old_cmd_buf is None")

        if old_cmd_buf is not None and not torch.allclose(old_cmd_buf, cmd_buf[0]):
            target_changed = True
            print(f"[INFO] Target changed at timestepXXXXXXXXXXXXX {idx}!")
            print(f"old_cmd_buf = {old_cmd_buf}")
            print(f"target is now at {cmd_buf[0]}")
        
        old_cmd_buf = cmd_buf[0].clone()

        # print("term.time_left[0] =", term.time_left[0])
        # print("cmd_manager.get_command(command_name) = ", cmd_manager.get_command(command_name)[0])

        with torch.inference_mode():
            

            # Record trajectory data if enabled
            if args_cli.plot_trajectories or args_cli.save_trajectory:
                
                # Plot trajectories if enabled
                if target_changed:
                    if len(end_effector_positions) == 0:
                        print("end_effector_positions is empty ! Continuing...")
                        target_changed = False
                        continue
                    
                    count += 1
                    print(f"[INFO] Plotting end-effector trajectories #{count}...")
                    
                    # Convert to tensors
                    ee_positions_tensor = torch.tensor(np.array(end_effector_positions))
                    ee_orientations_tensor = torch.tensor(np.array(end_effector_orientations))
                    target_pos_tensor = torch.tensor(np.array(target_positions[-2]))  # Use last target
                    target_quat_tensor = torch.tensor(np.array(target_orientations[-2]))  # Use last target

                    # INSERT_YOUR_CODE
                    print(f"ee_positions_tensor shape: {ee_positions_tensor.shape}")
                    print(f"ee_orientations_tensor shape: {ee_orientations_tensor.shape}")
                    print(f"target_pos_tensor shape: {target_pos_tensor.shape}")
                    print(f"target_quat_tensor shape: {target_quat_tensor.shape}")
                    
                    if args_cli.plot_trajectories and plot_traj:
                        # Create plots directory
                        plots_dir = os.path.join(log_dir, "trajectory_plots")
                        os.makedirs(plots_dir, exist_ok=True)
                        
                        # Plot 2D projections
                        plot_trajectory_with_cone(
                            ee_positions_tensor, ee_orientations_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h, sphere_radius=sphere_r,
                            save_path=os.path.join(plots_dir, f"{count}_trajectory_2d.png"),
                            show_plot=False
                        )
                        
                        # Plot 3D trajectory
                        plot_3d_trajectory_with_cone(
                            ee_positions_tensor, ee_orientations_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h, sphere_radius=sphere_r,
                            save_path=os.path.join(plots_dir, f"{count}_trajectory_3d.png"),
                            show_plot=False
                        )
                    
                        # Analyze trajectory
                        analysis = analyze_trajectory_cone_penalty(
                            ee_positions_tensor, target_pos_tensor, target_quat_tensor,
                            cone_r, cone_h
                        )
                        
                        print("\nTrajectory Analysis:")
                        print(f"Total steps: {analysis['total_steps']}")
                        print(f"Steps inside cone: {analysis['steps_inside_cone']}")
                        print(f"Percentage inside cone: {analysis['percentage_inside_cone']:.2f}%")
                        print(f"Average distance to boundary: {analysis['avg_distance_to_boundary']:.4f}")
                    
                        # Save analysis to file
                        analysis_file = os.path.join(plots_dir, f"{count}_trajectory_analysis.txt")
                        with open(analysis_file, 'w') as f:
                            f.write("Trajectory Analysis\n")
                            f.write("==================\n")
                            f.write(f"Total steps: {analysis['total_steps']}\n")
                            f.write(f"Steps inside cone: {analysis['steps_inside_cone']}\n")
                            f.write(f"Percentage inside cone: {analysis['percentage_inside_cone']:.2f}%\n")
                            f.write(f"Average distance to boundary: {analysis['avg_distance_to_boundary']:.4f}\n")
                        
                        print(f"Analysis saved to {analysis_file}")
                
                    # Save trajectory data if enabled
                    if args_cli.save_trajectory:
                        
                        print("[INFO] Saving trajectory data...")
                        
                        trajectory_file = os.path.join(log_dir, f"{count}_trajectory_data.npz")
                        np.savez(
                            trajectory_file,
                            actions=np.array(actions_list),
                            end_effector_positions=np.array(end_effector_positions),
                            end_effector_orientations=np.array(end_effector_orientations),
                            target_positions=np.array(target_positions),
                            target_orientations=np.array(target_orientations),
                            cone_radius=cone_r,
                            cone_height=cone_h,
                            sphere_radius=sphere_r
                        )
                        print(f"Trajectory data saved to {trajectory_file}")

                
                    # Reset trajectory recording
                    target_changed = False
                    actions_list.clear()
                    end_effector_positions.clear()
                    end_effector_orientations.clear()
                    target_positions.clear()
                    target_orientations.clear()

                    # Store new target
                    current_target_pos_b = cmd_buf[0, :3]
                    current_target_quat_b = cmd_buf[0, 3:7]

                    # Get robot base pose
                    root_pos_w = robot.data.root_state_w[0, :3]
                    root_quat_w = robot.data.root_state_w[0, 3:7]

                    # Convert target pose to world frame
                    target_pos_w, target_quat_w = combine_frame_transforms(
                        root_pos_w,
                        root_quat_w,
                        current_target_pos_b,
                        current_target_quat_b
                    )  # (num_envs, 3)

                    # end if target_changed

                else: # if target not changed
                    
                    # Always get end-effector position and orientation
                    ee_pos_w = robot.data.body_state_w[0, eff_link_id, :3]
                    ee_quat_w = robot.data.body_state_w[0, eff_link_id, 3:7]  # Get orientation
                    
                    # # Get end-effector position with respect to the base
                    # base_link_id = robot.body_names.index("base_link")
                    # base_link_pos = robot.data.body_state_w[0, base_link_id, :3]
                    # ee_pos_w = ee_pos_w - base_link_pos                    
                    
                    if actions is not None:
                        actions_list.append(actions.cpu().numpy())
                    end_effector_positions.append(ee_pos_w.cpu().numpy())
                    end_effector_orientations.append(ee_quat_w.cpu().numpy())
                    target_positions.append(target_pos_w.cpu().numpy())
                    target_orientations.append(target_quat_w.cpu().numpy())

                    print("##################################################")
                    # print(f"actions_list                     = {actions_list}")
                    print(f"len(actions_list)                = {len(actions_list)}")
                    print(f"ee_pos_w                    = {ee_pos_w}")
                    print(f"ee_quat_w                   = {ee_quat_w}")
                    print(f"target_pos_w                = {target_pos_w}")
                    print(f"target_quat_w               = {target_quat_w}")
                    print(f"current_target_pos_b        = {current_target_pos_b}")
                    print(f"current_target_quat_b       = {current_target_quat_b}")
                    print(f"root_pos_w                  = {root_pos_w}")
                    print(f"root_quat_w                 = {root_quat_w}")
                    print(f"cmd_buf[0, :3]              = {cmd_buf[0, :3]}")                
                    print(f"relative_ee_pos             = {ee_pos_w - target_pos_w}")
                    print(f"len(end_effector_positions) = {len(end_effector_positions)}")
                    print(f"len(target_positions)       = {len(target_positions)}")
                    print(f"len(target_orientations)    = {len(target_orientations)}")
                    print(f"target_str                  = {target_str}")

                    # end target_changed is False
            
                
                
            
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            print("obs = ", obs[0])
            print("actions[0] = ", actions[0])

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
