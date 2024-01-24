# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import cv2
import numpy as np
import torch
from av3d.datasets.abstract import CachedIterDataset
from av3d.datasets.zju_parser import SubjectParser
from av3d.utils.camera import generate_rays, transform_cameras
from av3d.utils.structures import Bones, Cameras, namedtuple_map
from av3d.utils.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
from av3d.utils_hn.camera_util import rotate_camera_by_frame_idx
from av3d.utils.io import resample_dp, mat2K

def _dataset_view_split(parser, split):
    _train_camera_ids = [0]
    if split == "all":
        camera_ids = parser.camera_ids
    elif split == "train": #"text-prompt"
        camera_ids = _train_camera_ids
    elif split in ["val_ind", "val_ood", "val_view"]:
        camera_ids = list(set(parser.camera_ids) - set(_train_camera_ids))
    elif split == "test":
        camera_ids = [0]
    return camera_ids


def _dataset_frame_split(parser, split):
    if split in ["train", "val_view"]:
        splits_fp = os.path.join(parser.root_dir, "splits/train.txt")
    else:
        splits_fp = os.path.join(parser.root_dir, f"splits/{split}.txt")
    with open(splits_fp, mode="r") as fp:
        frame_list = np.loadtxt(fp, dtype=int).tolist()
    return frame_list


def _dataset_index_list(parser, split):
    camera_ids = _dataset_view_split(parser, split)
    frame_list = _dataset_frame_split(parser, split)
    index_list = []
    frame_list = [0] * 196
    camera_ids = [0]
    # breakpoint()
    for frame_id in frame_list:
        index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
    return index_list


class SubjectLoader(CachedIterDataset):
    """Single subject data loader for training and evaluation."""

    SPLIT = ["all", "train", "val_ind", "val_ood", "val_view", "test"]

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        resize_factor: float = 1.0,
        color_bkgd_aug: str = None,
        num_rays: int = None,
        cache_n_repeat: int = 0,
        near: float = None,
        far: float = None,
        legacy: bool = False,
        bones_23: bool = False,
        **kwargs,
    ):
        assert split in self.SPLIT, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        self.resize_factor = resize_factor
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.legacy = legacy
        self.bones_23 = bones_23
        self.training = (num_rays is not None) and (split in ["train", "all"])
        self.color_bkgd_aug = color_bkgd_aug
        self.parser = SubjectParser(subject_id=subject_id, root_fp=root_fp)
        self.index_list = _dataset_index_list(self.parser, split)
        self.dtype = torch.get_default_dtype()
        super().__init__(self.training, cache_n_repeat)

    def __len__(self):
        return len(self.index_list)

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        #feat = data["feat_1024"]
        image, alpha = torch.split(rgba, [3, 1], dim=-1)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, dtype=rgba.dtype)
        else:
            # just use black during inference
            color_bkgd = torch.zeros(3, dtype=rgba.dtype)

        # only replace regions with `alpha == 0` to `color_bkgd`
        image = image * (alpha != 0) + color_bkgd * (alpha == 0)

        if self.num_rays is not None:  # usually this is in the training phase
            resolution = image.shape[0] * image.shape[1]
            # only sample rays in regions with `alpha == 0 or 1`
            indices = torch.where(
                ((alpha == 0) | (alpha == 1)).reshape(resolution)
            )[0]
            density = alpha.reshape(resolution)
            ray_indices = indices[torch.randperm(len(indices))][: self.num_rays]
            density = density[ray_indices] 
            pixels = image.reshape(resolution, 3)[ray_indices]
            # f#eat= feat.reshape(resolution, 16)[ray_indices]            
            rays = namedtuple_map(
                lambda r: r.reshape([resolution] + list(r.shape[2:])), rays
            )
            rays = namedtuple_map(lambda x: x[ray_indices], rays)
        else:
            pixels = image
            density = alpha
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "mask": density,
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            #"feat": feat,
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        frame_id, camera_id = self.index_list[index]
        K = self.parser.cameras[camera_id]["K"].copy()
        w2c = self.parser.cameras[camera_id]["w2c"].copy()
        D = self.parser.cameras[camera_id]["D"].copy()
        # breakpoint()
        # create pixels
        rgba = np.concatenate(
            [
                self.parser.load_image(frame_id, camera_id),
                self.parser.load_mask(frame_id, camera_id, trimap=True)[
                    ..., None
                ],
            ],
            axis=-1,
        )
        rgba = (
            torch.from_numpy(
                cv2.resize(
                    cv2.undistort(rgba, K, D),
                    (0, 0),
                    fx=self.resize_factor,
                    fy=self.resize_factor,
                    interpolation=cv2.INTER_AREA,
                )
            ).to(self.dtype)
            / 255.0
        )
        # kaug = torch.from_numpy(K.reshape((3,3)))[None]
        # feat = self.parser.load_feat(index, camera_id)
        # feat = feat.reshape((16,112,112)).copy()
        # feat = torch.from_numpy(feat)[None]
        # dp_bbox = self.parser.load_bbox(index, camera_id)
        # dp_bbox = torch.from_numpy(dp_bbox)[None]
        # feat_rz = resample_dp(feat, dp_bbox, mat2K(torch.eye(3)), 1024) 
        # feat_rz = feat_rz.squeeze(0)
        # feat_rz = feat_rz.permute(2,1,0)
        
        # rotate_camera_by_frame_idx
        ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
        }
        # ray_list = []
        # breakpoint()
        # for frame_id in np.arange(0,196,1):
        E = rotate_camera_by_frame_idx(
                extrinsics=w2c,#self.train_camera['extrinsics'], 
                frame_idx=index,
                # rotate_axis = 'z',
                # period=20,
                trans=None, #np.array([0.0, 0.0, 0.75]),
                inv_angle = True)
        
        # breakpoint()
        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(E).to(self.dtype),
            # extrins=torch.from_numpy(w2c).to(self.dtype),
            distorts=None,
            width=self.parser.WIDTH,
            height=self.parser.HEIGHT,
        )
        cameras = transform_cameras(cameras, self.resize_factor)
        rays = generate_rays(
            cameras, opencv_format=True, near=self.near, far=self.far
        )
        # ray_list.append(rays)
            
        # breakpoint()
        return {
            "subject_id": self.parser.subject_id,
            "camera_id": camera_id,
            # `meta_id` is used to query pose info from `pose_meta_info`
            "meta_id": frame_id,
            "rgba": rgba,  # [h, w, 4]
            "rays": rays,  # [h, w]
            "rigid_clusters": torch.tensor(
                self.parser.RIGID_BONE_IDS
            ).long(),  # bone cluster ids
            "timestamps": torch.tensor(frame_id).float(),
            # "feat_1024": feat_rz #[h, w, 16]
        }

    def build_pose_meta_info(self):
        meta_ids = [frame_id for frame_id, _ in self.index_list]
        meta_data = self.parser.load_meta_data(frame_ids=meta_ids)
        # create indexing for this split
        # load canonical meta info.
        rest_matrixs = meta_data["rest_tfs_bone"]#[1:]  # [23, 4, 4]
        rest_tails = meta_data["rest_joints"]#[
        #     [
        #         self.parser.JOINT_NAMES.index(tail_name)
        #         for _, tail_name in self.parser.BONE_NAMES
        #     ]
        # ]  # [23, 3]
        
        if self.bones_23:
            rest_matrixs = meta_data["rest_tfs_bone"][1:]  # [23, 4, 4]
            rest_tails = meta_data["rest_joints"][
                [
                    self.parser.JOINT_NAMES.index(tail_name)
                    for _, tail_name in self.parser.BONE_NAMES
                ]
            ]  # [23, 3]
        
        bones_rest = Bones(
            heads=None,
            tails=torch.from_numpy(rest_tails).to(self.dtype),
            transforms=torch.from_numpy(rest_matrixs).to(self.dtype),
        )  # real bones [23,]
        # load view space meta info.
        pose_matrixs = meta_data["tf_bones"]#[:, 1:]  # [N, 23, 4, 4]
        pose_tails = meta_data["joints"]#[
        #     :,
        #     [
        #         self.parser.JOINT_NAMES.index(tail_name)
        #         for _, tail_name in self.parser.BONE_NAMES
        #     ],
        # ]  # [N, 23, 3]

        if self.bones_23:
            pose_matrixs = meta_data["tf_bones"][:, 1:]  # [N, 23, 4, 4]
            pose_tails = meta_data["joints"][
                :,
                [
                    self.parser.JOINT_NAMES.index(tail_name)
                    for _, tail_name in self.parser.BONE_NAMES
                ],
            ]  # [N, 23, 3]
        
        bones_posed = [
            Bones(
                heads=None,
                tails=torch.from_numpy(pose_tails[i]).to(self.dtype),
                transforms=torch.from_numpy(pose_matrixs[i]).to(self.dtype),
            )
            for i in range(len(meta_ids))
        ]  # [23,] * N
        
        if self.legacy:
            # The paper uses axis-angles as pose latent. However it can
            # be discontinuous.
            pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
        else:
            # An improved version is to use the 6D rotation? (not verified)
            pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
            _aa, _g_aa, _g_transl = torch.split(pose_latent, [72, 3, 3], dim=-1)
            assert (
                _aa[:, 0:3] == 0
            ).all()  # ZJU diable root rotation by default.
            _aa[
                :, 0:3
            ] = _g_aa  # write the global rotation into the root rotation.
            pose_latent = torch.cat(
                [
                    matrix_to_rotation_6d(
                        axis_angle_to_matrix(_aa.reshape(-1, 24, 3))
                    ).reshape(-1, 24 * 6),
                    _g_transl,
                ],
                dim=-1,
            )
        vertices = torch.from_numpy(meta_data["rest_verts"]).to(self.dtype)
        return {
            "meta_ids": meta_ids,
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
            "pose_latent": pose_latent,
            "vertices": vertices
        }