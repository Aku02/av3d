import torch
import torch.nn.functional as F


from av3d.lib.model.network import ImplicitNetwork
from av3d.lib.model.helpers import hierarchical_softmax, skinning, bmv, create_voxel_grid, query_weights_smpl
from torch.utils.cpp_extension import load

from av3d.utils.structures import Bones
from av3d.models.basic.posi_enc import PositionalEncoder
from av3d.models.deform_posi_enc.naive import PoseConditionDPEncoder
from torch import Tensor
import numpy as np
from av3d.utils.knn import knn_gather
from av3d.lib.model.smpl import SMPLServer
from av3d.utils.bone import closest_distance_to_points, sample_on_bones, get_end_points
from typing import Callable
from av3d.lib.model.sample import PointOnBones
from av3d.models.basic.posi_enc import PositionalEncoder
from av3d.models.deform_posi_enc.snarf import _SkinWeightsNet
import trimesh


import os
cuda_dir = os.path.join(os.path.dirname(__file__), "../cuda")
fuse_kernel = load(name='fuse_cuda',
                   extra_cuda_cflags=[],
                   sources=[f'{cuda_dir}/fuse_kernel/fuse_cuda.cpp',
                            f'{cuda_dir}/fuse_kernel/fuse_cuda_kernel.cu'])
filter_cuda = load(name='filter',
                   sources=[f'{cuda_dir}/filter/filter.cpp',
                            f'{cuda_dir}/filter/filter.cu'])
precompute_cuda = load(name='precompute',
                   sources=[f'{cuda_dir}/precompute/precompute.cpp',
                            f'{cuda_dir}/precompute/precompute.cu'])
print("ok loaded")

class ForwardDeformer(torch.nn.Module):
    """
    Tensor shape abbreviation:
        B: batch size
        N: number of points
        J: number of bones
        I: number of init
        D: space dimension
    """

    def __init__(self, 
                 opt, 
                 posi_enc: torch.nn.Module,
                 n_transforms: int,  # n transforms for LBS.
                #  with_bkgd: bool = True,  # with the background virtual bone.
                #  search_n_init: int = 5,  # K initialization for root finding.
                #  soft_blend: int = 5,  # soften the skinning weights softmax.
                 # disable the non-linear offset during inference
                 offset_net_enabled: bool = True,  # enable non-linear offset network.
                 offset_pose_dim: int = None,  # pose-dependent latent dim.
                 # zero the offset during inference.
                 offset_constant_zero: bool = False,
                 # sample-bone distance threshold in the canonical space.
                 cano_dist: float = None,
                 ):
        super().__init__()

        self.opt = opt
        self.posi_enc = posi_enc
        self.n_transforms = n_transforms
        # self.with_bkgd = with_bkgd
        # self.search_n_init = search_n_init
        # self.soft_blend = soft_blend
        self.offset_net_enabled = offset_net_enabled
        # self.offset_constant_zero = offset_constant_zero
        self.cano_dist = cano_dist
        # self.n_transforms = n_transforms
        # self.with_bkgd = with_bkgd
        # self.inp_posi_enc = PositionalEncoder(
        #     in_dim=3, min_deg=0, max_deg=4, append_identity=True
        # )

        self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19] 

        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # convert to voxel grid
        meta_info = np.load('av3d/meta.npy', allow_pickle=True)
        meta_info = meta_info.item()
        gender = str(meta_info['gender'])
        betas  = meta_info['betas'] if 'betas' in meta_info else None
        v_template  = meta_info['v_template'] if 'v_template' in meta_info else None
        self.smpl_server = SMPLServer(gender='male', betas=betas, v_template=v_template)
        smpl_verts = self.smpl_server.verts_c
        device = self.smpl_server.verts_c.device

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res
        grid = create_voxel_grid(d, h, w, device=device)
        
        gt_bbox = torch.cat([smpl_verts.min(dim=1).values, 
                             smpl_verts.max(dim=1).values], dim=0)
        self.offset = -(gt_bbox[0] + gt_bbox[1])[None,None,:] / 2

        self.scale = torch.zeros_like(self.offset)
        self.scale[...] = 1./((gt_bbox[1] - gt_bbox[0]).max()/2 * self.opt.global_scale)
        self.scale[:,:,-1] = self.scale[:,:,-1] * self.opt.z_ratio
        
        # self.scale = self.scale*

        self.grid_denorm = grid/self.scale - self.offset
        # self.grid_denorm = self.grid_denorm *5
        # self.offset_net = _OffsetsNet(offset_pose_dim)
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        if self.opt.skinning_mode == 'preset':
            self.lbs_voxel_final = query_weights_smpl(self.grid_denorm, smpl_verts, self.smpl_server.weights_c)
            self.lbs_voxel_final = self.lbs_voxel_final.permute(0,2,1).reshape(1,-1,d,h,w)

        elif self.opt.skinning_mode == 'voxel':
            lbs_voxel = 0.001 * torch.ones((1, 24, d, h, w), dtype=self.grid_denorm.dtype, device=self.grid_denorm.device)
            self.register_parameter('lbs_voxel', torch.nn.Parameter(lbs_voxel,requires_grad=True))

        elif self.opt.skinning_mode == 'mlp':
            self.lbs_network = ImplicitNetwork(**self.opt.network)#, d_in = self.inp_posi_enc.out_dim)
            # self.lbs_network = _SkinWeightsNet(24)#, d_in = self.inp_posi_enc.out_dim)

        else:
            raise NotImplementedError('Unsupported Deformer.')
    
    @property
    def warp_dim(self):
        return 3

    @property
    def out_dim(self):
        return self.posi_enc.out_dim

    @property
    def diag(self):
        return self.posi_enc.diag
    
    def agg(self, density, rgb):
        """The aggregation function for multiple candidates.

        :params density: [..., I, 1]
        :params color: [..., I, 3]
        :params x_cano: [..., I, 3]
        :params mask: [..., I]
        :return
            aggregated density: [..., 1]
            aggregated color: [..., 3]
            aggregated x_cano: [..., 3]
            aggregated valid: which values are all valid.
        """
        density, indices = torch.max(density, dim=-2)
        rgb = torch.gather(
            rgb,
            -2,
            indices.unsqueeze(-2).expand(list(rgb.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        return density, rgb
    
    def aggregate(self, density, rgb, x_cano, mask):
        """The aggregation function for multiple candidates.

        :params density: [..., I, 1]
        :params color: [..., I, 3]
        :params x_cano: [..., I, 3]
        :params mask: [..., I]
        :return
            aggregated density: [..., 1]
            aggregated color: [..., 3]
            aggregated x_cano: [..., 3]
            aggregated valid: which values are all valid.
        """
        density, indices = torch.max(density, dim=-2)
        rgb = torch.gather(
            rgb,
            -2,
            indices.unsqueeze(-2).expand(list(rgb.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        x_cano = torch.gather(
            x_cano,
            -2,
            indices.unsqueeze(-2).expand(list(x_cano.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        mask = mask.any(dim=-1)
        return density, rgb, x_cano, mask
    
    # Built using the fast_snarf codes
    # def get_extra_losses(
    #     self,
    #     bones_rest,
    #     cond,
    #     radiance_field,
    # ):
    #     losses = {}
    #     # breakpoint()
        
    #     num_batch = 1 #bone_samples.shape[0]
    #     heads, tails = get_end_points(bones_rest)
    #     # breakpoint()
    #     bone_samples = sample_on_bones(
    #         bones_rest, n_per_bone=5, range=(0.1, 0.9)
    #     )  # [n_per_bone, n_bones, 3]
    #     bone_samples = bone_samples.reshape(num_batch, -1, 3)
    #     # pts_c, d_gt = self.sampler_bone.get_points(heads.expand(num_batch,  -1, -1))
    #     # breakpoint()
    #     # x_enc = self.posi_enc((pts_c, pts_c))
    #     pred_density = radiance_field.query_density(bone_samples)
    #     d_gt = torch.ones((num_batch, bone_samples.shape[1]), device=bone_samples.device)
    #     pred_density = pred_density.squeeze(-1)
    #     loss_bone_w = F.mse_loss(pred_density, d_gt)
    #     losses["loss_bone_w"] = loss_bone_w
    #     ##################################
    #     # heads, tails = get_end_points(bones_rest)
    #     bone_samples = sample_on_bones(
    #         bones_rest, n_per_bone=5, range=(0.1, 0.9)
    #     )  # [n_per_bone, n_bones, 3]
    #     # pts_c, w_gt = self.sampler_bone.get_joints(bone_samples)
    #     w_pd = self.query_weights(bone_samples) #[5,24,24]
    #     rigid_clusters = torch.arange(24)
    #     w_gt = (
    #         F.one_hot(rigid_clusters, num_classes=w_pd.shape[-1])
    #         .expand_as(w_pd)
    #         .to(w_pd)
    #     )
    #     # breakpoint()
    #     loss_bone_occ = F.mse_loss(w_pd, w_gt)
    #     losses["loss_bone_occ"] = loss_bone_occ
    #     return losses

    def get_extra_losses(
        self,
        bones_rest,
        cond,
        radiance_field,
    ):
        losses = {}
        # breakpoint()
        
        num_batch = 1 #bone_samples.shape[0]
        heads, tails = get_end_points(bones_rest)
        # breakpoint()
        # bone_samples = sample_on_bones(
        #     bones_rest, n_per_bone=5, range=(0.1, 0.9)
        # )  # [n_per_bone, n_bones, 3]
        # bone_samples = bone_samples.reshape(num_batch, -1, 3)
        bone_samples, d_gt = self.sampler_bone.get_points(heads.expand(num_batch,  -1, -1))
        # breakpoint()
        # x_enc = self.posi_enc((pts_c, pts_c))
        pred_density = radiance_field.query_density(bone_samples)
        # d_gt = torch.ones((num_batch, bone_samples.shape[1]), device=bone_samples.device)
        pred_density = pred_density.squeeze(-1)
        loss_bone_w = F.binary_cross_entropy_with_logits(pred_density, d_gt)
        losses["loss_bone_w"] = loss_bone_w
        ##################################
        # heads, tails = get_end_points(bones_rest)
        # bone_samples = sample_on_bones(
        #     bones_rest, n_per_bone=5, range=(0.1, 0.9)
        # )  # [n_per_bone, n_bones, 3]
        pts_c, w_gt = self.sampler_bone.get_joints(heads.expand(num_batch,  -1, -1))
        w_pd = self.query_weights(pts_c) #[5,24,24]
        # rigid_clusters = torch.arange(24)
        # w_gt = (
        #     F.one_hot(rigid_clusters, num_classes=w_pd.shape[-1])
        #     .expand_as(w_pd)
        #     .to(w_pd)
        # )
        # breakpoint()
        loss_bone_occ = F.mse_loss(w_pd, w_gt)
        losses["loss_bone_occ"] = loss_bone_occ
        return losses


    def forward(self, xd,
                x_cov: Tensor,
                bones_world: Bones,
                bones_cano: Bones, rigid_clusters, cond, eval_mode=False):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D] 1024, 64, 3 # 1, 64, 3
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        B = 1
        N, _3 = xd.shape
        xd = xd.view(B, N, _3)
        # breakpoint()
        # xd = xd.reshape(1, -1, _3)
        # x_cov = x_cov.view(1, -1, _3)
        # print(bones_world.transforms.shape)
        tfs = bones_cano.transforms @ bones_world.transforms.inverse()
        # tfs = torch.einsum('nij,njk->nik', bones_world.transforms, bones_cano.transforms.inverse())
        tfs = tfs.expand([1, 24 ,4 ,4])
        # query_smpl = True
        # if query_smpl:
        #     dummy_v = self.smpl_server.verts_c
        #     w = query_weights_smpl(xd, dummy_v, self.smpl_server.weights_c)
        #     xc = skinning(xd, w, tfs, inverse = True)
        #     xc = xc.reshape(B, N, _3) # extract mesh 
        #     return xc, xc, xc
            # breakpoint()
        # tfs = tfs.view(-1, 24, 4, 4)
        # # distance between bones and samples in the world space.
        # dists = closest_distance_to_points(bones_world, xd)  # [B, N, n_bones]

        # # get data for transforms
        # n_init = 9
        # dists = _try_collapse_rigid_bones(
        #     dists.permute(2, 0, 1),
        #     rigid_clusters=rigid_clusters,
        #     collapse_func=lambda x: x.min(dim=0).values,
        # ).permute(
        #     1, 2, 0
        # )  # [B, N, n_transforms]
        # transforms = _try_collapse_rigid_bones(
        #     # (world -> bone -> cano)
        #     bones_cano.transforms @ bones_world.transforms.inverse(),
        #     rigid_clusters=rigid_clusters,
        #     collapse_func=lambda x: x[0],
        # )  # [n_transforms, 4, 4]

        # # select the k nearest
        # breakpoint()
        # knn = torch.topk(dists, n_init, dim=-1, largest=False)
        # knn_idxs = knn.indices  # [B, N, n_init]

        # # Gather the knn transforms
        # transforms = knn_gather(
        #     transforms.reshape(1, -1, 4 * 4), knn_idxs.reshape(1, -1, n_init)
        # ).view(
        #     [B, N, n_init, 4, 4]
        # )  # [B, N, n_init, 4, 4]
        
        # tfs = tfs.view(B, -1, 4, 4)
        # breakpoint()
        xc_opt, others = self.search(xd, cond, tfs, eval_mode=eval_mode)
        # 
        if eval_mode or self.opt.skinning_mode == 'preset':
            # breakpoint()
            xc_opt = xc_opt.view(B, N, -1, 3)
            if x_cov is not None:
                x_cov = (
                    x_cov.unsqueeze(-2).expand(list(xc_opt.shape[:-1]) + [3])
                    if self.posi_enc.diag
                    else x_cov.unsqueeze(-3).expand(list(xc_opt.shape[:-1]) + [3, 3])
                )
                x_enc = self.posi_enc((xc_opt, x_cov))
            else:
                x_enc = self.posi_enc(xc_opt)
            mask = others['valid_ids'].view(x_enc.shape[:-1]) 
            # breakpoint()
            # xc_opt = xc_opt.squeeze(0)
            return x_enc, xc_opt, mask

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()

        n_batch, n_point, n_init, n_dim = xc_opt.shape

        xd_opt = self.forward_skinning(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs, mask=others['valid_ids'].flatten(1,2))

        if not self.opt.use_slow_grad:
            grad_inv = others['J_inv'].reshape(n_batch, n_point * n_init, 3,3)
        else:
            grad_inv = self.gradient(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs).inverse().detach()

        correction = xd_opt - xd_opt.detach()
        correction = bmv(-grad_inv.flatten(0,1), correction.unsqueeze(-1).flatten(0,1)).squeeze(-1).reshape(xc_opt.shape)

        xc = xc_opt + correction
        xc = xc.reshape(n_batch, n_point, n_init, n_dim)
        xc = xc.reshape(B, N, -1, 3)
        if x_cov is not None:
            x_cov = (
                x_cov.unsqueeze(-2).expand(list(xc.shape[:-1]) + [3])
                if self.posi_enc.diag
                else x_cov.unsqueeze(-3).expand(list(xc.shape[:-1]) + [3, 3])
            )
            x_enc = self.posi_enc((xc, x_cov))
        else:
            x_enc = self.posi_enc(xc)
        mask = others['valid_ids'].view(x_enc.shape[:-1]) 
        # xc = xc.squeeze(0)
        return x_enc, xc, mask

    def precompute(self, tfs=None, recompute_skinning=True):

        if recompute_skinning or not hasattr(self,"lbs_voxel_final"):

            if self.opt.skinning_mode == 'mlp':
                self.mlp_to_voxel()
            
            elif self.opt.skinning_mode == 'voxel':
                self.voxel_to_voxel()

        b, c, d, h, w = tfs.shape[0], 3, self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res

        voxel_d = torch.zeros( (b,3,d,h,w), device=tfs.device)
        voxel_J = torch.zeros( (b,12,d,h,w), device=tfs.device)
        
        precompute_cuda.precompute(self.lbs_voxel_final, tfs, voxel_d, voxel_J, self.offset, self.scale)

        return voxel_d, voxel_J

    def search(self, xd, cond, tfs, eval_mode=False):
        """Search correspondences.

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D]
            xc_init (tensor): deformed points in batch. shape: [B, N, I, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc_opt (tensor): canonoical correspondences of xd. shape: [B, N, I, D]
            valid_ids (tensor): identifiers of converged points. [B, N, I]
        """

        voxel_d, voxel_J = self.precompute(tfs, recompute_skinning=not eval_mode)

        # run broyden without grad
        with torch.no_grad():
            result = self.broyden_cuda(xd, voxel_d, voxel_J, tfs, 
                                    cvg_thresh=self.opt.cvg,
                                    dvg_thresh=self.opt.dvg)

        return result['result'], result


    def broyden_cuda(self,
                    xd_tgt,
                    voxel,
                    voxel_J_inv,
                    tfs,
                    cvg_thresh=1e-4,
                    dvg_thresh=1):
        """
        Args:
            g:     f: (N, 3, 1) -> (N, 3, 1)
            x:     (N, 3, 1)
            J_inv: (N, 3, 3)
        """
        b, n, _ = xd_tgt.shape

        n_init = self.init_bones_cuda.shape[0]

        xc = torch.zeros((b,n,n_init,3),device=xd_tgt.device,dtype=torch.float)

        J_inv = torch.zeros((b,n,n_init,3,3),device=xd_tgt.device,dtype=torch.float)

        is_valid = torch.zeros((b,n,n_init),device=xd_tgt.device,dtype=torch.bool)

        fuse_kernel.fuse_broyden(xc, xd_tgt, voxel, voxel_J_inv, tfs, self.init_bones_cuda, self.opt.align_corners, J_inv, is_valid, self.offset, self.scale, cvg_thresh, dvg_thresh)

        mask = filter_cuda.filter(xc, is_valid)

        return {"result": xc, 'valid_ids': mask, 'J_inv': J_inv}


    def forward_skinning(self, xc, cond, tfs, mask=None):
        """Canonical point -> deformed point

        Args:
            xc (tensor): canonoical points in batch. shape: [B, N, D]
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xd (tensor): deformed point. shape: [B, N, D]
        """
        if mask is None:
            w = self.query_weights(xc, cond)
            xd = skinning(xc, w, tfs, inverse=False)
        else:
            w = self.query_weights(xc, cond, mask=mask.flatten(0,1))
            xd = skinning(xc,w, tfs, inverse=False)

        return xd

    def mlp_to_voxel(self):

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res
        # enc_inp = self.inp_posi_enc(self.grid_denorm)
        # lbs_voxel_final = self.lbs_network(self.grid_denorm, cond, None)
        lbs_voxel_final = self.lbs_network(self.grid_denorm, {}, None)
        lbs_voxel_final = self.opt.soft_blend * lbs_voxel_final

        if self.opt.softmax_mode == "hierarchical":
            lbs_voxel_final = hierarchical_softmax(lbs_voxel_final)
        else:
            lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)

        self.lbs_voxel_final = lbs_voxel_final.permute(0,2,1).reshape(1,24,d,h,w)

    def voxel_to_voxel(self):

        lbs_voxel_final = self.lbs_voxel*self.opt.soft_blend

        self.lbs_voxel_final = F.softmax(lbs_voxel_final, dim=1)

    def query_weights(self, xc, cond=None, mask=None, mode='bilinear'):

        # if not hasattr(self,"lbs_voxel_final"):
        if self.opt.skinning_mode == 'mlp':
            self.mlp_to_voxel()
        elif self.opt.skinning_mode == 'voxel':
            self.voxel_to_voxel()

        xc_norm = (xc + self.offset) * self.scale
        
        w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), xc_norm.unsqueeze(2).unsqueeze(2), align_corners=self.opt.align_corners, mode=mode, padding_mode='zeros')
        
        w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
        
        return w


def _try_collapse_rigid_bones(
        data: Tensor,
        rigid_clusters: Tensor = None,
        collapse_func: Callable = None,
        ) -> Tensor:
        """Try collapse somes bone attributes.

        The reason we are doing this is because some bones may be rigidly
        attached to each other so there any some redundancies in the data and
        sometimes it can cause ambiguity. For example, in the task of
        skinning, a surface point can listen to either bone if there are
        multiple bones moving with the same transformation.

        Warning: This function always assume you are trying to collapse the
        **first** dimension of the data.

        :params data: Bone attribute to be collapsed. [B, ...]
        :params rigid_clusters: The cluster id for each bone in torch.int32.
            [B,] The bones with the same cluster id are supposed to be moved
            rigidly together.
        :params collapse_func: Callable function to decide how to collapse.
            For example, `lambda x: x[0]`; `lambda x: x.min(dim=0)` etc.
        :returns
            the collapsed data. The shape should be [n_clusters, ...], where
            n_clusters = len(unique(rigid_clusters)). It may also return
            the original data if you pass None to rigid_clusters.
        """
        if rigid_clusters is None:
            # don't do collapse
            return data
        assert collapse_func is not None
        assert len(rigid_clusters) == len(data)
        data_collapsed = []
        for cluster_id in torch.unique(rigid_clusters):
            selector = rigid_clusters == cluster_id
            # the bones map to the same transform_id should have the
            # same transformation because they are supposed to be
            # rigidly attached to each other.
            # TODO(ruilong) We skip the check here for speed but maybe
            # better to have a check?
            data_collapsed.append(collapse_func(data[selector]))
        data_collapsed = torch.stack(data_collapsed, dim=0)
        return data_collapsed