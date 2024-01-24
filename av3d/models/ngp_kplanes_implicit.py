# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import torch.nn as nn
from av3d.models.basic.mlp import MLP
from av3d.models.basic.refine import Binarize, LaplaceDensity, binary_MLP
from av3d.models.basic.nerf import (
    NerfModel,
    sample_along_rays,
    sample_pdf,
    volumetric_rendering,
)
from av3d.models.deform_posi_enc.rigid import DisentangledDPEncoder
from av3d.models.deform_posi_enc.snarf import SNARFDPEncoder
from av3d.lib.model.fast_avatar_mip_kplane_origin import ForwardDeformer

import torch.nn.functional as F
from nerfacc.estimators.occ_grid import OccGridEstimator

from av3d.utils.bone import (
    closest_distance_to_points,
    closest_distance_to_rays,
    get_end_points,
)
from av3d.utils.structures import Bones, Rays, namedtuple_map

from av3d.plenoxels.models.kplane_field import KPlaneField
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable
from nerfacc.grid import ray_aabb_intersect, traverse_grids
from nerfacc.volrend import (
    accumulate_along_rays,
    render_weight_from_alpha,
    rendering,
)
from scipy.spatial.transform import Rotation as R
import math
from av3d.lib.model.helpers import masked_softmax
import trimesh
import pytorch3d.ops as ops
import sys
from av3d.lib.model.helpers import skinning
threebase='av3d_release/av3d/third_party/threestudio/'
sys.path.insert(0,threebase)

import threestudio

def _restore_and_fill(values, masks, fill_in=0.0):
    assert masks.dim() == 1
    restored_values = torch.zeros(
        [masks.shape[0]] + list(values.shape[1:]),
        dtype=values.dtype,
        device=values.device,
    )
    restored_values[masks] = values
    restored_values[~masks] += fill_in
    return restored_values


def _select_rays_near_to_bones(rays: Rays, bones: Bones, threshold: float):
    """Select rays near to the bones and calculate per-ray near far plane."""
    dists, t_vals = closest_distance_to_rays(bones, rays)  # [n_rays, n_bones]

    heads, tails = get_end_points(bones)
    margin = torch.linalg.norm(heads - tails, dim=-1) / 2.0 + threshold

    # get near far but relax with margin
    t_margin = margin / torch.linalg.norm(rays.directions, dim=-1, keepdim=True)
    t_vals[dists >= threshold] = -1e10
    far = (t_vals + t_margin).max(dim=-1).values
    t_vals[dists >= threshold] = 1e10
    near = (t_vals - t_margin).min(dim=-1).values.clamp(min=0)
    selector = near < far

    rays = namedtuple_map(lambda x: x[selector], rays)
    near = near[selector]
    far = far[selector]
    return rays, near, far, selector


def _interp_along_rays(masks, z_vals, values_list, dim=-1):
    assert masks.dim() == 2 and dim == -1
    t = torch.arange(masks.shape[dim], device=masks.device) + 1
    indices_next = (masks.shape[dim] - 1) - torch.cummax(
        masks.flip([dim]) * t, dim
    ).indices.flip([dim])
    indices_before = torch.cummax(masks * t, dim).indices

    z_vals_next = torch.gather(z_vals, dim, indices_next)
    z_vals_before = torch.gather(z_vals, dim, indices_before)
    z_weight = (z_vals - z_vals_before) / (z_vals_next - z_vals_before + 1e-10)

    masks_next = torch.gather(masks, 1, indices_next)
    masks_before = torch.gather(masks, 1, indices_before)
    masks_new = masks_next & masks_before

    outs = [masks_new]
    for values in values_list:
        values_next = torch.gather(
            values, 1, indices_next.unsqueeze(-1).expand_as(values)
        )
        values_before = torch.gather(
            values, 1, indices_before.unsqueeze(-1).expand_as(values)
        )
        values_interp = (values_next - values_before) * z_weight.unsqueeze(
            -1
        ) + values_before
        outs.append(
            values_interp * masks_new.to(values_interp.dtype).unsqueeze(-1)
        )
    return outs

def validate_empty_rays(ray_indices, t_start, t_end):
    if ray_indices.nelement() == 0:
        threestudio.warn("Empty rays_indices!")
        ray_indices = torch.LongTensor([0]).to(ray_indices)
        t_start = torch.Tensor([0]).to(ray_indices)
        t_end = torch.Tensor([0]).to(ray_indices)
    return ray_indices, t_start, t_end


class DynNerfModel(NerfModel):
    def __init__(
        self,
        pos_enc: nn.Module,
        geometry_type: str = None,
        geometry: dict = None,
        variance:dict = None,
        shading_mode: int = None,  # shading mode
        # The dim of the pose-dependent condition for shading
        # None or zero means no shading condition.
        shading_pose_dim: int = None,
        # Sample-bone distance threshold in the world space.
        world_dist: float = None,
        mlp_coarse: nn.Module = None,
        # mlp_fine: nn.Module = None,
        grid_config = None,
        concat_features_across_scales = None,
        multiscale_res = None,
        use_appearance_embedding = None,
        appearance_embedding_dim = None,
        spatial_distortion= None,
        density_activation: Callable = None,
        linear_decoder: bool = None,
        linear_decoder_layers: Optional[int] = None,
        num_images: Optional[int] = None,
        use_ao: Optional[bool] = None,
        use_direction: Optional[bool] = None,
        **kwargs,
    ):
        # `pos_enc` is a deformable positional encoding, that maps a world
        # coordinate `x_w` to its representation `x_c` conditioned on the
        # pose latent `p`. pos_enc(x, p) -> x_c
        # breakpoint()
        assert shading_mode in [None, "implicit", "implicit_AO"]
        self.shading_mode = shading_mode
        self.shading_pose_dim = shading_pose_dim
        self.world_dist = world_dist
        self.geometry_type = geometry_type
        self.geometry = geometry
        self.variance_conf = variance
        ao_activation = None
        
        if mlp_coarse is None:
            # Define the MLP that query the color & density etc from the
            # representation `x_c`. We treat object as lambertian so we don't
            # model view-dependent color in av3d.
            if shading_mode is None:
                # the color is not shaded (not pose-dependent).
                mlp_coarse = KPlaneField(
                    # aabb = torch.tensor([[-2.0, -1.75, -0.5], [2.0, 1.75, 3.5]]),
                    aabb = torch.tensor([[-5.0,-5.0,-5.0], [5.0,5.0,5.0]]),
                    grid_config = grid_config,
                    concat_features_across_scales = concat_features_across_scales,
                    multiscale_res = multiscale_res,
                    use_appearance_embedding = use_appearance_embedding,
                    appearance_embedding_dim = appearance_embedding_dim,
                    spatial_distortion = spatial_distortion,
                    density_activation = density_activation,
                    linear_decoder = linear_decoder,
                    linear_decoder_layers = linear_decoder_layers,
                    num_images = num_images,
                    use_ao = use_ao,
                    use_direction = use_direction,
                )
                mlp_fine = None           
                ao_activation = None
            elif shading_mode == "implicit":
                # the color is implicitly conditioned on the shading condition
                assert shading_pose_dim is not None
                mlp_coarse = MLP(
                    input_dim=pos_enc.out_dim,
                    # implicitly shading-conditioned color
                    condition_dim=shading_pose_dim,
                    # diable AO output
                    num_ao_channels=0,
                    condition_ao_dim=0,
                )
                ao_activation = None
            elif shading_mode == "implicit_AO":
                assert shading_pose_dim is not None
                # the color is scaled by ambiant occlution
                # which is learnt implicitly
                mlp_coarse = MLP(
                    input_dim=pos_enc.out_dim,
                    # disable implicitly conditioned color
                    condition_dim=0,
                    # enable AO output
                    num_ao_channels=1,
                    condition_ao_dim=shading_pose_dim,
                )
                ao_activation = torch.nn.Sigmoid()
            else:
                raise ValueError(shading_mode)

        super().__init__(mlp_coarse,mlp_fine, pos_enc, use_viewdirs=False, **kwargs)
        self.ao_activation = ao_activation
        self.scene_aabb = torch.tensor([-10.0, -10.0, -10.0, 10.0, 10.0, 10.0], device="cuda:0")
        self.cone_angle = torch.tensor(0.0, dtype=torch.float32, device="cuda:0")
        self.render_step_size = (
            (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
            * math.sqrt(3)
            / 1024 #render_n_samples
        ).item()
        ind = torch.load('av3d/index_samples.pt')
        self.ind = ind.to("cuda:0")
        self.occupancy_grid = OccGridEstimator(
        roi_aabb=self.scene_aabb/2,
        resolution=[112,112,112]).to("cuda:0")
        self.geometry = threestudio.find(self.geometry_type)(self.geometry)
        self.conv_size = 112
        
        self.variance = LaplaceDensity(params_init=self.variance_conf['params_init'], beta_min=self.variance_conf['beta_min']) 

        self.cos_anneal_ratio = 1.0 #if cos_anneal_end == 0 else min(1.0, step / cos_anneal_end)
        
    def _query_mlp(
        self,
        samples,
        bones_posed,
        bones_rest,
        rigid_clusters,
        pose_latent,
        randomized=True,
        **kwargs,
    ):
        x = samples

        # Deform and encode the world coordinates conditioned on Bone movements.
        # `x_enc` is ready to enter the MLP for querying color and density etc,
        # `x_warp` is basically `x_enc` but strips out the [sin, cos] frequency encoding.
        mask, valid = None, None
        if isinstance(self.pos_enc, DisentangledDPEncoder):
            x_enc, x_warp = self.pos_enc(
                x,
                None,
                bones_posed,
                rigid_clusters=rigid_clusters,
            )
        elif isinstance(self.pos_enc, SNARFDPEncoder):
            x_enc, x_warp, mask, valid = self.pos_enc(
                x,
                None,
                bones_posed,
                bones_rest,
                rigid_clusters=rigid_clusters,
                pose_latent=pose_latent,
            )
        
        elif isinstance(self.pos_enc, ForwardDeformer):
            smpl_params = pose_latent #kwargs.get("pose_latent", None)
            smpl_params = smpl_params.view(-1, 78)
            cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
            
            n_rays = x.shape[0]
            x_enc, x_warp, mask, valid = self.pos_enc(
                    xd = x, # 211, 64, 3
                    x_cov = None, # 211, 64, 3
                    bones_world = bones_posed,
                    bones_cano = bones_rest,
                    rigid_clusters=rigid_clusters, # 23
                    cond=cond, # 69
                    eval_mode = not(randomized),
                    **kwargs
                )
        else:
            raise ValueError(type(self.pos_enc))
        
        return x_enc, x_warp, mask, valid
    
    def render_image(
        # scene
        self,
        radiance_field: torch.nn.Module,
        occupancy_grid: OccGridEstimator,
        rays: Rays,
        near_plane: Optional[float] = None,
        far_plane: Optional[float] = None,
        render_step_size: float = 1e-3,
        render_bkgd: Optional[torch.Tensor] = None,
        cone_angle: float = 0.0,
        alpha_thre: float = 0.0,
        # test options
        test_chunk_size: int = 8192,
        # only useful for dnerf
        timestamps= None,
        bones_posed = None,
        bones_rest = None,
        rigid_clusters = None,
        pose_latent = None,
        verts = None,
        randomized=None,
        **kwargs
    ):
        """Render the pixels of an image."""
        timestamps = None
        rays_shape = rays.origins.shape
        if len(rays_shape) == 3:
            height, width, _ = rays_shape
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        else:
            num_rays, _ = rays_shape
        
        results = []
        chunk = (
            torch.iinfo(torch.int32).max
            if radiance_field.training
            else test_chunk_size
        )
        n_rays = num_rays
        for i in range(0, num_rays, chunk):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk], rays)
            occupancy_grid.binaries[:] = True
            ray_indices, t_starts, t_ends = occupancy_grid.sampling(
                chunk_rays.origins,
                chunk_rays.viewdirs,
                alpha_fn=None,
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                stratified=True,#radiance_field.training,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )

            ray_indices = ray_indices.long()
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            midpoints = (t_starts + t_ends)[:, None] / 2.0
            positions = t_origins + t_dirs * midpoints
            dists = t_ends - t_starts

            n, p = positions.shape
            positions = positions.view(1, n, p)
            x_enc, x_warp, others, mask = self._query_mlp(
                    # rays,
                    positions,
                    bones_posed,
                    bones_rest,
                    randomized=randomized,
                    rigid_clusters= rigid_clusters,
                    pose_latent = pose_latent,
                    **kwargs,
                )
            num_batch, num_point, num_init, num_dim = x_warp.shape
            x_warp = x_warp.reshape(num_batch, num_point * num_init, num_dim)
            mask = mask.reshape(num_batch, num_point * num_init)
            # breakpoint()
            sdf = self.geometry.forward_sdf(x_warp)
            mode = 'min' #'softmax' if not (not randomized) else 'max'
            x_warp = x_warp.view(num_batch, num_point , num_init, num_dim)
            sdf = sdf.view(num_batch, num_point , num_init, -1)
            mask = mask.view(num_batch, num_point , num_init)
            sdf, x_warp = masked_softmax(sdf, mask, x_warp, dim=-1, mode=mode, soft_blend=20)
            features_implicit = self.geometry(x_warp.squeeze(0), output_normal = True)
            
            sdf, sdf_grad, feature, normal  = features_implicit['sdf'], features_implicit['sdf_grad'], features_implicit['features'], features_implicit['normal']
            sdf = sdf.squeeze(-1)
            x_warp = x_warp.squeeze(0)
    
            pnts_c = x_warp
            pnts_c = pnts_c.requires_grad_(True)
            smpl_tfs = bones_posed.transforms @ bones_rest.transforms.inverse()
            transforms = torch.cat(
                    [torch.eye(4)[None, :, :].to(smpl_tfs), smpl_tfs], dim=-3
                )
            if isinstance (self.pos_enc, ForwardDeformer):
                if self.pos_enc.opt.bones_23:
                    smpl_tfs = transforms.expand([1, 24, 4 ,4])
                else:
                    smpl_tfs = transforms.expand([1, 25, 4 ,4])
                smpl_params = pose_latent.view(-1,78)
                cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
            
            if isinstance(self.pos_enc, SNARFDPEncoder):
                normal_d = self.pos_enc.forward_skinning(x_cano = normal.unsqueeze(0), 
                                 bones_world = bones_posed,
                                 bones_cano = bones_rest,
                                 rigid_clusters = rigid_clusters,
                                 pose_latent = pose_latent,
                                 mask= None,).squeeze(0)
            elif isinstance(self.pos_enc, ForwardDeformer):
                weights = self.pos_enc.query_weights(xc = pnts_c.unsqueeze(0), cond = cond, mask = None)
                normal_d = skinning(normal.unsqueeze(0), weights, smpl_tfs).squeeze(0)
           
            #################################################################
            normal_d = F.normalize(normal_d, dim=-1, eps=1e-6)
            alpha = self.get_alpha(sdf, normal_d, t_dirs, dists)#[...,None]
            alpha = alpha.squeeze(-1)
            outs = radiance_field(pts = feature, normals = normal_d, cond_extra = pose_latent, directions = t_dirs)
            rgb = outs['rgb']
            sdf = sdf.unsqueeze(-1)
            weights, trans = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
            opacity = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
            depth = accumulate_along_rays(weights, values=midpoints, ray_indices=ray_indices, n_rays=n_rays)
            rgb = accumulate_along_rays(weights, values=rgb, ray_indices=ray_indices, n_rays=n_rays)

            normal_d = accumulate_along_rays(weights, values=normal_d, ray_indices=ray_indices, n_rays=n_rays)
            normal_d = F.normalize(normal_d, dim=-1, eps=1e-6)

            chunk_results = [rgb, opacity, depth, normal_d, sdf_grad, sdf, len(t_starts)]
            results.append(chunk_results)
        colors, opacities, depths, normal_d, sdf_grad, sdf, n_rendering_samples = [
            torch.cat(r, dim=0) if isinstance(r[0], torch.Tensor) else r
            for r in zip(*results)
        ]
        
        return (
            colors.view((*rays_shape[:-1], -1)),
            opacities.squeeze(),
            depths.squeeze(),
            x_warp,
            normal_d,
            sdf_grad,
            sdf,
            sum(n_rendering_samples),
        )
    def get_alpha(self, sdf, normal, dirs, dists):
        # breakpoint()
        density = self.variance(sdf)
        free_energy = dists * density
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        return alpha


    def forward(
        self,
        rays: Rays,
        color_bkgd: torch.Tensor,
        bones_posed: Bones,
        bones_rest: Bones,
        randomized: bool = True,
        step = None,
        timestamp = None,
        feat = None,
        vert = None,
        **kwargs,
    ):
        ret = []
        extra_info = {}

        # Calculate per-ray near far distance based on the bones
        # and select rays near to the bones to process
        rays, near, far, selector = _select_rays_near_to_bones(
            rays, bones_posed, threshold=self.world_dist
        )
        
        if ~selector.any():
            for _ in range(self.num_levels):
                ret.append(
                    (
                        torch.ones(
                            (selector.shape[0], 3), 
                            dtype=near.dtype,
                            device=selector.device
                        )
                        * color_bkgd,  # rgb
                        torch.zeros(
                            (selector.shape[0]), 
                            dtype=near.dtype,
                            device=selector.device
                        ),  # depth
                        torch.zeros(
                            (selector.shape[0]), 
                            dtype=near.dtype,
                            device=selector.device
                        ),  # acc
                        torch.zeros(
                            (selector.shape[0], self.pos_enc.warp_dim),
                            dtype=near.dtype,
                            device=selector.device
                        ),  # warp
                        torch.zeros(
                            (selector.shape[0], 16),
                            dtype=near.dtype,
                            device=selector.device
                        ),  # warp
                    )
                )
            return ret, extra_info

        # start rendering
        weights = None
        mlp = self.mlp_coarse #if i_level == 0 else self.mlp_fine
        def occ_eval_fn(x):
            step_size = self.render_step_size
            a, b = x.shape
            x = x.view(1, a, b)
            samples_selector = (
                closest_distance_to_points(bones_rest, x)
                .min(dim=-1)
                .values
                < self.world_dist
            )
            
            x = x * samples_selector[..., None]
            raw_density,_ = mlp.query_sigma(pts = x.unsqueeze(-2))#.reshape(num_batch, num_point, num_init)
            raw_density = raw_density.view(1, -1)
            raw_density = raw_density.squeeze(0)   
            return raw_density * step_size
        
        # update occupancy grid
        if randomized:
            self.occupancy_grid.train()
        
        if randomized:
            mlp.train()
            comp_rgb, acc, depth, can_points, can_normals, can_sdf_grads, can_sdfs, num_s = self.render_image(
                        radiance_field = mlp,
                        occupancy_grid = self.occupancy_grid,
                        rays = rays,
                        # scene_aabb = self.scene_aabb,
                        near_plane = near.min(),
                        far_plane = far.max(),
                        render_step_size = self.render_step_size,
                        render_bkgd = color_bkgd,
                        bones_posed = bones_posed,
                        bones_rest = bones_rest,
                        # rigid_clusters= kwargs.get("rigid_clusters", None),
                        # pose_latent = kwargs.get("pose_latent", None),
                        randomized = randomized,
                        verts = vert,
                        # timestamps=timestamp,
                        **kwargs
                        )
        
        else:
            with torch.no_grad():
                # print("Inference starts!")
                # breakpoint()
                mlp.eval()
                self.occupancy_grid.eval()
                comp_rgb, acc, depth, can_points, can_normals, _, _, num_s = self.render_image(
                        radiance_field = mlp,
                        occupancy_grid = self.occupancy_grid,
                        rays = rays,
                        # scene_aabb = self.scene_aabb,
                        near_plane = near.min(),
                        far_plane = far.max(),
                        render_step_size = self.render_step_size,
                        render_bkgd = color_bkgd,
                        bones_posed = bones_posed,
                        bones_rest = bones_rest,
                        test_chunk_size=8192,
                        randomized = randomized,
                        verts = vert,
                        **kwargs
                        )
        can_normals = can_normals * acc[...,None]
        comp_rgb = comp_rgb + color_bkgd * (1.0 - acc[..., None])
        ret.append(
            (
                _restore_and_fill(comp_rgb, selector, fill_in=color_bkgd),
                _restore_and_fill(depth, selector, fill_in=0.0),
                _restore_and_fill(acc, selector, fill_in=0.0),
                _restore_and_fill(can_normals, selector, fill_in=0.0),
                _restore_and_fill(comp_rgb, selector, fill_in=0.0),
            )
        )
        if randomized:
            if isinstance(self.pos_enc, ForwardDeformer):
                # {"loss_bone_w", "loss_bone_offset"}
                smpl_params = kwargs.get("pose_latent", None)
                smpl_params = smpl_params.view(-1, 78)
                
                cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
                extra_info.update(
                    self.pos_enc.get_extra_losses(
                        bones_rest,
                        rigid_clusters=kwargs.get("rigid_clusters", None),
                        cond = cond,
                        can_pc = can_points,
                        geometry = self.geometry,
                        normals = can_normals,
                        sdf_grads = can_sdf_grads,
                        sdf = can_sdfs,
                        # pose_latent=kwargs.get("pose_latent", None),
                    )
                )
            
            if isinstance(self.pos_enc, SNARFDPEncoder):
                smpl_params = kwargs.get("pose_latent", None)
                smpl_params = smpl_params.view(-1, 78)
                
                cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
                # {"loss_bone_w", "loss_bone_offset"}
                extra_info.update(
                    self.pos_enc.get_extra_losses(
                        bones_rest,
                        rigid_clusters=kwargs.get("rigid_clusters", None),
                        cond = cond,
                        can_pc = can_points,
                        geometry = self.geometry,
                        normals = can_normals,
                        sdf_grads = can_sdf_grads,
                        sdf = can_sdfs,
                        # pose_latent=kwargs.get("pose_latent", None)
                    )
                )
            
        return ret, extra_info


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s
    
    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s
    
    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min((global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s, self.max_inv_s)

def volsdf_density(sdf, inv_std):
    beta = (1 / inv_std) + torch.tensor(0.0001).cuda()
    alpha = 1/ beta
    return F.relu(alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta)))