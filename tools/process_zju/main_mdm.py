# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import numpy as np
import torch
import tqdm
from av3d.utils.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix

from body_model import SMPLlayer
from bary import query_points

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
# Convert mdm genereated data to the format used by AV3D

def load_rest_pose_info(subject_id: int, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "mdm_params_kick", # "CHANGE: new_params"
    )
    fp = os.path.join(data_dir, "1.npy")
    smpl_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "new_params",
    )
    smpl_fp = os.path.join(smpl_dir, "1.npy")
    smpl_data = np.load(smpl_fp, allow_pickle=True).item()
    Rh = np.zeros((1, 3), dtype=np.float32)
    Th = np.zeros((1, 3), dtype=np.float32)
    vertices, joints, joints_transform, bones_transform = body_model(
        poses=np.zeros((1, 72), dtype=np.float32),
        shapes=smpl_data["shapes"],
        Rh=Rh,
        Th=Th,
        scale=1,
        new_params=False,
    )
    # breakpoint()
    return (
        vertices.squeeze(0), 
        joints.squeeze(0), 
        joints_transform.squeeze(0),
        bones_transform.squeeze(0),
    )


def load_pose_info(subject_id: int, frame_id: int, body_model):
    data_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "mdm_params_kick",
    )
    fp = os.path.join(data_dir, "%d.npy" % (frame_id + 1))
    mdm_data = np.load(fp, allow_pickle=True).item()
    smpl_dir = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "new_params",
    )
    smpl_fp = os.path.join(smpl_dir, "1.npy")
    smpl_data = np.load(smpl_fp, allow_pickle=True).item()
    Th = np.zeros((1, 3), dtype=np.float32)

    poses = mdm_data['thetas']

    poses = torch.from_numpy(poses) # tensor of B x 6
    poses = rotation_6d_to_matrix(poses) # tensor of B x 3 x 3
    poses = matrix_to_axis_angle(poses) # tensor of B x 3
    # poses = np.matmul(poses, rot)
    poses = poses.reshape(1, 72)
    Rh = poses[:, :3].view(1,3).numpy()
    poses[:, :3] = 0 # for reason, see: https://github.com/zju3dv/EasyMocap/blob/master/doc/02_output.md#attention-for-smplsmpl-x-users
    vertices, joints, joints_transform, bones_tranform = body_model(
        poses=np.array(poses),
        shapes=np.array(smpl_data['shapes']),
        Rh=Rh,
        Th=Th,
        scale=1,
        new_params=False,
    )
    pose_params = torch.cat(
        [
            torch.tensor(poses),
            torch.tensor(Rh),
            torch.tensor(Th),
        ], dim=-1
    ).float()
    return (
        vertices.squeeze(0),
        joints.squeeze(0),
        joints_transform.squeeze(0),
        pose_params.squeeze(0),
        bones_tranform.squeeze(0),
    )


def cli(subject_id: int):
    print ("processing subject %d" % subject_id)
    # smpl body model
    body_model = SMPLlayer(
        model_path=os.path.join(PROJECT_DIR, "data"), gender="neutral", 
    )
    frame_ids = list(range(120))
    # rest state info
    rest_verts, rest_joints, rest_tfs, rest_tfs_bone = (
        load_rest_pose_info(subject_id, body_model)
    )
    lbs_weights = body_model.weights.float()
    print('rest pose info loaded correctly')
    # pose state info
    verts, joints, tfs, params, tf_bones, inds = [], [], [], [], [], []
    for frame_id in tqdm.tqdm(frame_ids):
        _verts, _joints, _tfs, _params, _tfs_bone = (
            load_pose_info(subject_id, frame_id, body_model)
        )
        verts.append(_verts)
        joints.append(_joints)
        tfs.append(_tfs)
        params.append(_params)
        tf_bones.append(_tfs_bone)
    verts = torch.stack(verts)
    joints = torch.stack(joints)
    tfs = torch.stack(tfs)
    params = torch.stack(params)
    tf_bones = torch.stack(tf_bones)

    data = {
        "lbs_weights": lbs_weights,  # [6890, 24]
        "rest_verts": rest_verts,  # [6890, 3]
        "rest_joints": rest_joints,  # [24, 3]
        "rest_tfs": rest_tfs,  # [24, 4, 4]
        "rest_tfs_bone": rest_tfs_bone, # [24, 4, 4]
        "verts": verts,  # [1470, 6890, 3]
        "joints": joints,  # [1470, 24, 3]
        "tfs": tfs,  # [1470, 24, 4, 4]
        "tf_bones": tf_bones,  # [1470, 24, 4, 4]
        "params": params,  # [1470, 72 + 3 + 3]
    }
    save_path = os.path.join(
        PROJECT_DIR, "data", "zju", "CoreView_%d" % subject_id, "pose_data_mdm.pt"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)


if __name__ == "__main__":
    for subject_id in [313]:
        cli(subject_id)
        

'''
For new_params (zju original data)
(Pdb) data.item().keys()
dict_keys(['poses', 'Rh', 'Th', 'shapes'])
(Pdb) data.item()['poses'].shape
(1, 72)
(Pdb) data.item()['Rh'].shape
(1, 3)
(Pdb) data.item()['Th'].shape
(1, 3)
(Pdb) data.item()['shapes'].shape
(1, 10)
'''

'''
For stuff generated by mdm
sample00_rep00_smpl_params.npy
(Pdb) array.item().keys() -> dict_keys(['motion', 'thetas', 'root_translation', 'faces', 'vertices', 'text', 'length'])
(Pdb) array.item()['motion'].shape -> (25, 6, 120)
(Pdb) array.item()['thetas'].shape -> (24, 6, 120)
(Pdb) array.item()['root_translation'].shape -> (3, 120)
(Pdb) array.item()['faces'].shape -> (13776, 3)
(Pdb) array.item()['vertices'].shape -> torch.Size([6890, 3, 120])
(Pdb) array.item()['length'] -> 120
'''
