#!/usr/bin/env python3
import argparse
import gc
import multiprocessing as mp
import os
import pathlib

import torch
from natsort import natsorted
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import save_robot_motion
from smplx_bodypos_to_robot import ROBOT_CHOICES
from smplx_bodypos_to_robot import build_robot_motion
from smplx_bodypos_to_robot import load_bodypose_motion


DEFAULT_OUTPUT_NAME = "motion_shape_g1.npz"



def resolve_target_path(motion_file, src_folder, tgt_folder, output_name):
    motion_path = pathlib.Path(motion_file)
    if tgt_folder is None:
        return str(motion_path.with_name(output_name))

    src_root = pathlib.Path(src_folder)
    relative_parent = motion_path.parent.relative_to(src_root)
    return str(pathlib.Path(tgt_folder) / relative_parent / output_name)



def collect_motion_files(src_folder):
    motion_files = []
    for dirpath, _, filenames in os.walk(src_folder):
        if "motion_shape.npz" in filenames:
            motion_files.append(os.path.join(dirpath, "motion_shape.npz"))
    return natsorted(motion_files)



def process_file(task):
    motion_file, tgt_file, robot = task

    try:
        frame_data_list, motion_fps, actual_human_height = load_bodypose_motion(motion_file)
        retargeter = GMR(
            actual_human_height=actual_human_height,
            src_human="smplx",
            tgt_robot=robot,
            verbose=False,
        )
        qpos_list = [retargeter.retarget(frame_data).copy() for frame_data in frame_data_list]
        motion_data = build_robot_motion(qpos_list, motion_fps, retargeter.xml_file)
        save_robot_motion(tgt_file, motion_data)
        torch.cuda.empty_cache()
        gc.collect()
        return True, motion_file, tgt_file
    except Exception as exc:
        torch.cuda.empty_cache()
        gc.collect()
        return False, motion_file, str(exc)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_folder",
        required=True,
        type=str,
        help="Folder to scan recursively for motion_shape.npz files.",
    )
    parser.add_argument(
        "--tgt_folder",
        default=None,
        type=str,
        help="Optional folder to mirror outputs into. If omitted, writes next to each source file.",
    )
    parser.add_argument(
        "--robot",
        choices=ROBOT_CHOICES,
        default="unitree_g1",
    )
    parser.add_argument(
        "--output_name",
        default=DEFAULT_OUTPUT_NAME,
        type=str,
        help="Filename to write for each retargeted motion.",
    )
    parser.add_argument(
        "--override",
        default=False,
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Number of worker processes to use.",
    )
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError(f"Expected --num_workers >= 1, got {args.num_workers}.")
    if not args.output_name.endswith(".npz"):
        raise ValueError(f"Expected --output_name to end with .npz, got {args.output_name}.")

    src_folder = os.path.abspath(args.src_folder)
    tgt_folder = os.path.abspath(args.tgt_folder) if args.tgt_folder is not None else None

    motion_files = collect_motion_files(src_folder)
    print(f"Found {len(motion_files)} motion_shape.npz files in {src_folder}.")

    tasks = []
    skipped = 0
    for motion_file in motion_files:
        tgt_file = resolve_target_path(motion_file, src_folder, tgt_folder, args.output_name)
        if os.path.exists(tgt_file) and not args.override:
            skipped += 1
            continue
        tasks.append((motion_file, tgt_file, args.robot))

    print(f"Queued {len(tasks)} files for retargeting. Skipped {skipped} existing outputs.")
    if not tasks:
        print("Nothing to do.")
        return

    results = []
    if args.num_workers == 1:
        for task in tqdm(tasks, desc="Retargeting motions"):
            results.append(process_file(task))
    else:
        with mp.Pool(args.num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks), desc="Retargeting motions"):
                results.append(result)

    failures = [(motion_file, message) for success, motion_file, message in results if not success]
    successes = [(motion_file, tgt_file) for success, motion_file, tgt_file in results if success]

    print(f"Saved {len(successes)} retargeted motions.")
    if failures:
        print(f"Failed on {len(failures)} motions:")
        for motion_file, message in failures:
            print(f"  - {motion_file}: {message}")
    else:
        print("All motions processed successfully.")


if __name__ == "__main__":
    main()
