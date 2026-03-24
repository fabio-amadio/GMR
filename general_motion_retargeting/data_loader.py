import pathlib
import pickle

import numpy as np


WORLD_BODY_MOTION_FIELDS = (
    "fps",
    "body_link_names",
    "body_pos_w",
    "body_quat_w",
    "dof_pos",
)
LEGACY_ROOT_MOTION_FIELDS = (
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



def _resolve_robot_motion_fields(keys):
    key_set = set(keys)

    if all(field in key_set for field in WORLD_BODY_MOTION_FIELDS):
        return WORLD_BODY_MOTION_FIELDS
    if all(field in key_set for field in LEGACY_ROOT_MOTION_FIELDS):
        return LEGACY_ROOT_MOTION_FIELDS

    world_missing = [field for field in WORLD_BODY_MOTION_FIELDS if field not in key_set]
    legacy_missing = [field for field in LEGACY_ROOT_MOTION_FIELDS if field not in key_set]
    raise KeyError(
        "Unsupported robot motion schema. "
        f"Missing world-frame fields: {world_missing}. "
        f"Missing legacy root-frame fields: {legacy_missing}."
    )



def _quat_rotate_xyzw(quat, vec):
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    while quat_xyz.ndim < vec.ndim:
        quat_xyz = np.expand_dims(quat_xyz, axis=-2)
        quat_w = np.expand_dims(quat_w, axis=-2)
    t = 2.0 * np.cross(quat_xyz, vec)
    return vec + quat_w * t + np.cross(quat_xyz, t)



def _derive_legacy_views_from_world_motion(motion_data):
    body_link_names = np.asarray(motion_data["body_link_names"])
    body_pos_w = np.asarray(motion_data["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(motion_data["body_quat_w"], dtype=np.float32)
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)

    if body_link_names.ndim != 1:
        raise ValueError(f"Expected body_link_names to have shape (N,), got {body_link_names.shape}.")
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected body_pos_w to have shape (T, N, 3), got {body_pos_w.shape}.")
    if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"Expected body_quat_w to have shape (T, N, 4), got {body_quat_w.shape}.")
    if body_pos_w.shape[:2] != body_quat_w.shape[:2]:
        raise ValueError(
            "body_pos_w and body_quat_w must have matching frame/body dimensions, "
            f"got {body_pos_w.shape[:2]} and {body_quat_w.shape[:2]}."
        )
    if body_pos_w.shape[1] != len(body_link_names):
        raise ValueError(
            "The number of joints in body_pos_w/body_quat_w must match body_link_names, "
            f"got {body_pos_w.shape[1]} joints and {len(body_link_names)} names."
        )
    if dof_pos.ndim != 2 or dof_pos.shape[0] != body_pos_w.shape[0]:
        raise ValueError(
            "Expected dof_pos to have shape (T, D) with the same number of frames as body_pos_w, "
            f"got {dof_pos.shape} and {body_pos_w.shape[0]} frames."
        )

    root_pos = body_pos_w[:, 0, :]
    root_rot_wxyz = body_quat_w[:, 0, :]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    local_body_pos = body_pos_w - root_pos[:, None, :]
    inv_root_rot_xyzw = root_rot_xyzw.copy()
    inv_root_rot_xyzw[:, :3] *= -1.0
    local_body_pos = _quat_rotate_xyzw(inv_root_rot_xyzw, local_body_pos).astype(np.float32, copy=False)

    derived_motion_data = dict(motion_data)
    derived_motion_data["root_pos"] = root_pos.astype(np.float32, copy=False)
    derived_motion_data["root_rot"] = root_rot_xyzw.astype(np.float32, copy=False)
    derived_motion_data["local_body_pos"] = local_body_pos
    derived_motion_data["link_body_list"] = body_link_names
    return derived_motion_data



def save_robot_motion(motion_file, motion_data):
    """
    Save robot motion data to a `.pkl` or `.npz` file.
    """
    motion_path = pathlib.Path(motion_file)
    motion_path.parent.mkdir(parents=True, exist_ok=True)

    motion_fields = _resolve_robot_motion_fields(motion_data.keys())

    suffix = motion_path.suffix.lower()
    if suffix == ".pkl":
        with motion_path.open("wb") as f:
            pickle.dump({field: motion_data[field] for field in motion_fields}, f)
        return
    if suffix == ".npz":
        np.savez_compressed(
            motion_path,
            **{
                field: _prepare_npz_value(motion_data[field])
                for field in motion_fields
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
            motion_fields = _resolve_robot_motion_fields(data.files)
            motion_data = {
                field: _normalize_npz_value(data[field])
                for field in motion_fields
            }
    else:
        raise ValueError(
            f"Unsupported robot motion extension '{motion_path.suffix}'. "
            "Expected '.pkl' or '.npz'."
        )

    motion_fields = _resolve_robot_motion_fields(motion_data.keys())
    if motion_fields == WORLD_BODY_MOTION_FIELDS:
        motion_data = _derive_legacy_views_from_world_motion(motion_data)

    motion_fps = motion_data["fps"]
    motion_root_pos = motion_data["root_pos"]
    motion_root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # from xyzw to wxyz
    motion_dof_pos = motion_data["dof_pos"]
    motion_local_body_pos = motion_data["local_body_pos"]
    motion_link_body_list = motion_data["link_body_list"]
    return motion_data, motion_fps, motion_root_pos, motion_root_rot, motion_dof_pos, motion_local_body_pos, motion_link_body_list
