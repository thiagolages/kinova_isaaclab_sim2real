from plot_trajectory import *
import numpy as np
import os
# folder = "/workspace/kinova_isaaclab_sim2real/logs/rsl_rl/skimmer_gen3/002_pos_-0.5w_ori_-0.75w/"
# folder = "/workspace/kinova_isaaclab_sim2real/logs/rsl_rl/skimmer_gen3/002_2000iter_pos_-0.5w_ori_-0.75w_2025-08-07_22-31-41"


folder = "/workspace/research/assets/new_results/"
files = sorted(os.listdir(folder))
import matplotlib.pyplot as plt

for file in files:
    if not file.endswith(".npz"):
        continue
    # Only process files that have either 'w' or '7' in their names
    if not (('2' in file and '1' not in file) or '6' in file or '1_' in file):
        continue
        
    print(f"Processing file: {file}")
    data = np.load(os.path.join(folder, file))
    print(data.files)
    
    # data["target_positions"] = data["target_positions"][:-1]
    # Get the figures from the plotting functions (assume they return fig, ax)
    fig1 = plot_trajectory_with_cone(
        end_effector_positions=data["end_effector_positions"],
        end_effector_orientations=data["end_effector_orientations"],
        target_position=data["target_positions"][0], # remove last target position
        target_orientation=data["target_orientations"],
        cone_radius=data["cone_radius"],
        cone_height=data["cone_height"],
        show_plot=False
    )

    fig2 = plot_3d_trajectory_with_cone(
        end_effector_positions=data["end_effector_positions"],
        end_effector_orientations=data["end_effector_orientations"],
        target_position=data["target_positions"][0], # remove last target position
        target_orientation=data["target_orientations"],
        cone_radius=data["cone_radius"],
        cone_height=data["cone_height"],
        show_plot=False
    )

    # Now show the figures here
    fig1.show()
    fig2.show()
    plt.show()  # This will block until all open figures are closed
    # input("Press Enter to continue to the next file...")


    
    print(f"Just processed file: {file}")

