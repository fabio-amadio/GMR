import argparse
import os

import numpy as np

from general_motion_retargeting import load_robot_motion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GMR robot-motion files to CSV (for beyondmimic)")
    parser.add_argument(
        "--folder", type=str, help="Path to the folder containing robot-motion files from GMR",
    )
    args = parser.parse_args()

    out_folder = os.path.join(args.folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    motion_files = sorted(
        file for file in os.listdir(args.folder)
        if file.endswith((".pkl", ".npz"))
    )

    for i, file in enumerate(motion_files, start=1):
        motion_path = os.path.join(args.folder, file)
        motion_data, *_ = load_robot_motion(motion_path)

        dof_pos = motion_data["dof_pos"]
        frame_rate = motion_data["fps"]
        motion = np.zeros((dof_pos.shape[0], dof_pos.shape[1] + 7), dtype=np.float32)
        motion[:, :3] = motion_data["root_pos"]
        motion[:, 3:7] = motion_data["root_rot"]
        motion[:, 7:] = dof_pos

        if frame_rate > 30:
            # downsample to 30 fps
            downsample_factor = frame_rate / 30.0
            indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
            old_length = motion.shape[0]
            motion = motion[indices]
            print(f"Downsampled from {old_length} to {motion.shape[0]} frames")

        output_name = f"{os.path.splitext(file)[0]}.csv"
        output_path = os.path.join(out_folder, output_name)
        if os.path.exists(output_path):
            output_path = os.path.join(out_folder, f"{file}.csv")

        np.savetxt(output_path, motion, delimiter=",")
        print(f"({i}/{len(motion_files)}) Saved to {output_path}")
