#!/usr/bin/env python3
import argparse
import os
import pathlib
import time

import numpy as np
import torch
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import KinematicsModel
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting import save_robot_motion


REQUIRED_KEYS = ("body_names", "body_pos", "body_quat", "hz")
ROBOT_CHOICES = [
    "unitree_g1",
    "unitree_g1_with_hands",
    "unitree_h1",
    "unitree_h1_2",
    "booster_t1",
    "booster_t1_29dof",
    "stanford_toddy",
    "fourier_n1",
    "engineai_pm01",
    "kuavo_s45",
    "hightorque_hi",
    "galaxea_r1pro",
    "berkeley_humanoid_lite",
    "booster_k1",
    "pnd_adam_lite",
    "openloong",
    "tienkung",
    "fourier_gr3",
]



def estimate_actual_human_height(betas):
    if betas is None:
        return None

    betas = np.asarray(betas)
    if betas.size == 0:
        return None

    return float(1.66 + 0.1 * betas.reshape(-1)[0])



def load_bodypose_motion(motion_file):
    motion_path = pathlib.Path(motion_file)
    with np.load(motion_path, allow_pickle=True) as motion_npz:
        missing_keys = [key for key in REQUIRED_KEYS if key not in motion_npz]
        if missing_keys:
            raise KeyError(
                f"Missing required keys in {motion_path}: {missing_keys}. "
                f"Expected keys: {REQUIRED_KEYS}."
            )

        body_names = np.asarray(motion_npz["body_names"])
        body_pos = np.asarray(motion_npz["body_pos"], dtype=np.float32)
        body_quat = np.asarray(motion_npz["body_quat"], dtype=np.float32)

        hz_value = np.asarray(motion_npz["hz"])
        if hz_value.shape != ():
            raise ValueError(f"Expected scalar hz in {motion_path}, got shape {hz_value.shape}.")
        motion_fps = float(hz_value.item())

        betas = np.asarray(motion_npz["betas"], dtype=np.float32) if "betas" in motion_npz else None

    if body_names.ndim != 1:
        raise ValueError(f"Expected body_names to have shape (N,), got {body_names.shape}.")
    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError(f"Expected body_pos to have shape (T, N, 3), got {body_pos.shape}.")
    if body_quat.ndim != 3 or body_quat.shape[-1] != 4:
        raise ValueError(f"Expected body_quat to have shape (T, N, 4), got {body_quat.shape}.")
    if body_pos.shape[:2] != body_quat.shape[:2]:
        raise ValueError(
            "body_pos and body_quat must have the same first two dimensions, "
            f"got {body_pos.shape[:2]} and {body_quat.shape[:2]}."
        )
    if body_pos.shape[1] != len(body_names):
        raise ValueError(
            "The number of joints in body_pos/body_quat must match body_names, "
            f"got {body_pos.shape[1]} joints and {len(body_names)} names."
        )
    if motion_fps <= 0:
        raise ValueError(f"Expected hz to be positive, got {motion_fps}.")

    joint_names = [str(name) for name in body_names.tolist()]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"body_names must be unique, got duplicates in {motion_path}.")

    frame_data = []
    for frame_pos, frame_quat in zip(body_pos, body_quat):
        frame_data.append(
            {
                joint_name: (joint_pos.copy(), joint_quat.copy())
                for joint_name, joint_pos, joint_quat in zip(joint_names, frame_pos, frame_quat)
            }
        )

    return frame_data, motion_fps, estimate_actual_human_height(betas)



def default_save_path(motion_file):
    motion_path = pathlib.Path(motion_file)
    return str(motion_path.with_name("motion_shape_g1.npz"))



def build_robot_motion(qpos_list, motion_fps, xml_file):
    qpos_array = np.asarray(qpos_list, dtype=np.float32)
    root_pos = qpos_array[:, :3]
    root_rot_wxyz = qpos_array[:, 3:7]
    dof_pos = qpos_array[:, 7:]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    kinematics_model = KinematicsModel(xml_file, device=device)

    num_frames = qpos_array.shape[0]
    identity_root_pos = torch.zeros((num_frames, 3), device=device)
    identity_root_rot = torch.zeros((num_frames, 4), device=device)
    identity_root_rot[:, -1] = 1.0

    local_body_pos, _ = kinematics_model.forward_kinematics(
        identity_root_pos,
        identity_root_rot,
        torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
    )

    return {
        "fps": motion_fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": local_body_pos.detach().cpu().numpy(),
        "link_body_list": np.asarray(kinematics_model.body_names),
    }



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion_file",
        required=True,
        type=str,
        help="Path to a motion_shape.npz-style SMPL-X body-pose file.",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOT_CHOICES,
        default="unitree_g1",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the retargeted robot motion. Defaults to a sibling motion_shape_g1.npz.",
    )
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion in the MuJoCo viewer.",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record the video.",
    )
    parser.add_argument(
        "--video_path",
        default=None,
        help="Path to save the recorded video.",
    )
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit playback to the source motion rate.",
    )
    args = parser.parse_args()

    args.save_path = args.save_path or default_save_path(args.motion_file)
    if args.video_path is None:
        motion_stem = pathlib.Path(args.motion_file).stem
        args.video_path = f"videos/{args.robot}_{motion_stem}.mp4"

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    frame_data_list, motion_fps, actual_human_height = load_bodypose_motion(args.motion_file)
    print(
        f"Loaded {len(frame_data_list)} frames from {args.motion_file} at {motion_fps:.2f} Hz."
    )
    if actual_human_height is not None:
        print(f"Estimated actual human height: {actual_human_height:.3f} m")

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    qpos_list = []
    num_frames = len(frame_data_list)
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    frame_index = 0

    try:
        while True:
            if not args.loop and frame_index >= num_frames:
                break

            current_frame = frame_index % num_frames
            if args.loop and current_frame == 0 and frame_index > 0:
                retargeter.setup_retarget_configuration()

            fps_counter += 1
            current_time = time.time()
            if current_time - fps_start_time >= fps_display_interval:
                actual_fps = fps_counter / (current_time - fps_start_time)
                print(f"Actual rendering FPS: {actual_fps:.2f}")
                fps_counter = 0
                fps_start_time = current_time

            qpos = retargeter.retarget(frame_data_list[current_frame])

            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retargeter.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
                follow_camera=False,
            )

            if len(qpos_list) < num_frames:
                qpos_list.append(qpos.copy())

            frame_index += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot_motion_viewer.close()

    if len(qpos_list) != num_frames:
        print(
            f"[WARNING] Only collected {len(qpos_list)}/{num_frames} frames before exit; "
            f"not saving incomplete motion to {args.save_path}."
        )
        return

    motion_data = build_robot_motion(qpos_list, motion_fps, retargeter.xml_file)
    save_robot_motion(args.save_path, motion_data)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
