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
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import gen3.tasks  # noqa: F401


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

    dt = env.unwrapped.physics_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # Define your home pose (adjust as needed)
    home_pos = torch.tensor([0.5, 0.0, 0.80], device=env.unwrapped.device)  # Example position
    home_quat = torch.tensor([ 0.0, 0.0, 0.0,  1.0], device=env.unwrapped.device)  # Z pointing to the side (rotation of 90deg around X) (xyzw)
    home_pose = torch.cat([home_pos, home_quat])  # Shape (7,)

    skimmer_pos = torch.tensor([0.4, 0.4, 0.30], device=env.unwrapped.device)  # Example position
    skimmer_quat = torch.tensor([ 0.0, 0.0, 0.0,  1.0], device=env.unwrapped.device)  # Z pointing to the side (rotation of 90deg around X) (xyzw)
    skimmer_pose = torch.cat([skimmer_pos, skimmer_quat])  # Shape (7,)

    reach_threshold = 0.03  # meters

    # Usually 'ee_pose', adjust if your command name is different
    command_name = "ee_pose"
    cmd_manager = env.unwrapped.command_manager

    home_timeout = 5 # seconds to reach home
    set_time_left = False  # Flag to check if time_left is set
    # simulate environment
    while simulation_app.is_running():
        
        start_time = time.time()
        
        term = cmd_manager.get_term(command_name)

        if term.time_left[0] <= 0.0:
            set_time_left = False  # Reset flag if time is left

        if term.command_counter[0] % 2 == 0:
            cmd_buf = cmd_manager.get_command(command_name)
            cmd_buf[:] = home_pose          # broadcast to every parallel env
            print("cmd_buf[0] =", cmd_buf[0])
            
            if set_time_left == False:
                term.time_left[:] = home_timeout
                set_time_left = True  # Set time_left only once

        else:
            cmd_buf = cmd_manager.get_command(command_name)
            cmd_buf[:] = skimmer_pose          # broadcast to every parallel env
            print("cmd_buf[0] =", cmd_buf[0])
            if set_time_left == False:
                term.time_left[:] = home_timeout
                set_time_left = True  # Set time_left only once
        # else:
        #     set_time_left = False  # Reset flag for next command

        print("term.time_left[0] =", term.time_left[0])
        # print("cmd_manager.get_command(command_name) = ", cmd_manager.get_command(command_name)[0])

        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # print("actions[0] = ", actions[0])
            # print("obs = ", obs[0])


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
