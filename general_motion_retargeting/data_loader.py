import pathlib
import pickle

import numpy as np


ROBOT_MOTION_FIELDS = (
    "fps",
    "root_pos",
    "root_rot",
    "dof_pos",
    "local_body_pos",
    "link_body_list",
)



def _normalize_npz_value(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value



def _prepare_npz_value(value):
    if value is None:
        return np.array(None, dtype=object)
    return np.asarray(value)



def save_robot_motion(motion_file, motion_data):
    """
    Save robot motion data to a `.pkl` or `.npz` file.
    """
    motion_path = pathlib.Path(motion_file)
    motion_path.parent.mkdir(parents=True, exist_ok=True)

    missing_fields = [field for field in ROBOT_MOTION_FIELDS if field not in motion_data]
    if missing_fields:
        raise KeyError(f"Missing robot motion fields: {missing_fields}")

    suffix = motion_path.suffix.lower()
    if suffix == ".pkl":
        with motion_path.open("wb") as f:
            pickle.dump(motion_data, f)
        return
    if suffix == ".npz":
        np.savez_compressed(
            motion_path,
            **{
                field: _prepare_npz_value(motion_data[field])
                for field in ROBOT_MOTION_FIELDS
            },
        )
        return
    raise ValueError(
        f"Unsupported robot motion extension '{motion_path.suffix}'. "
        "Expected '.pkl' or '.npz'."
    )



def load_robot_motion(motion_file):
    """
    Load robot motion data from a `.pkl` or `.npz` file.
    """
    motion_path = pathlib.Path(motion_file)
    suffix = motion_path.suffix.lower()

    if suffix == ".pkl":
        with motion_path.open("rb") as f:
            motion_data = pickle.load(f)
    elif suffix == ".npz":
        with np.load(motion_path, allow_pickle=True) as data:
            motion_data = {
                field: _normalize_npz_value(data[field])
                for field in ROBOT_MOTION_FIELDS
            }
    else:
        raise ValueError(
            f"Unsupported robot motion extension '{motion_path.suffix}'. "
            "Expected '.pkl' or '.npz'."
        )

    motion_fps = motion_data["fps"]
    motion_root_pos = motion_data["root_pos"]
    motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # from xyzw to wxyz
    motion_dof_pos = motion_data["dof_pos"]
    motion_local_body_pos = motion_data["local_body_pos"]
    motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list
