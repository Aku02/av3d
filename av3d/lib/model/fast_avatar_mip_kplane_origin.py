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
from av3d.models.deform_posi_enc.snarf import _SkinWeightsNet, _OffsetsNet, _initialize_canonical
import trimesh
from av3d.lib.model.helpers import masked_softmax, tv_loss
from scipy.spatial.transform import Rotation as R
from torch import einsum
from av3d.models.projection.map import SurfaceAlignedConverter
from av3d.models.projection.map_utils import point_mesh_face_distance

from pytorch3d.structures import Meshes, Pointclouds
import pytorch3d.ops as ops


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
        self.offset_net_enabled = offset_net_enabled
        self.cano_dist = cano_dist
        self.inp_posi_enc = PositionalEncoder(
            in_dim=3, min_deg=0, max_deg=4, append_identity=True
        )
        if self.opt.bones_23:
            self.ret = 24
        else:
            self.ret = 25
        # self.init_bones = [0, 1, 2, 4, 5, 16, 17, 18, 19] 
        self.init_bones = [0, 1, 2, 4, 5,6, 11, 12, 13, 16, 17, 18, 19, 20]
        # self.init_bones = [i for i in range(self.ret)]

        self.init_bones_cuda = torch.tensor(self.init_bones).cuda().int()

        # convert to voxel grid
        meta_info = np.load('av3d/meta.npy', allow_pickle=True)
        meta_info_1 = np.load('av3d/data/zju/CoreView_313/lbs/smpl_params.npy', allow_pickle=True).item()
        meta_info = meta_info.item()
        gender = str(meta_info['gender'])
        betas  = meta_info['betas'] if 'betas' in meta_info else None
        v_template  = meta_info['v_template'] if 'v_template' in meta_info else None
        self.smpl_server = SMPLServer(gender='male', betas=meta_info_1['shapes'][0], v_template=v_template)
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
        
        self.grid_denorm = grid/self.scale - self.offset
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)
        
        if self.opt.skinning_mode == 'preset':
            self.lbs_voxel_final = query_weights_smpl(self.grid_denorm, smpl_verts, self.smpl_server.weights_c)
            self.lbs_voxel_final = self.lbs_voxel_final.permute(0,2,1).reshape(1,-1,d,h,w) 
            self.sa = SurfaceAlignedConverter()
           
        elif self.opt.skinning_mode == 'voxel':
            lbs_voxel = 0.001 * torch.ones((1, self.ret, d, h, w), dtype=self.grid_denorm.dtype, device=self.grid_denorm.device)
            self.register_parameter('lbs_voxel', torch.nn.Parameter(lbs_voxel,requires_grad=True))
            self.sa = SurfaceAlignedConverter()  
            scaling_factor = 1
            # faces = self.sa.faces_idx[None, ...].repeat(len(self.sa.verts), 1, 1)
            meshes = Meshes(verts=self.sa.verts.float().unsqueeze(0)*scaling_factor, faces=self.sa.faces_idx.unsqueeze(0))
            points = ops.sample_points_from_meshes(meshes, num_samples = 10000)
            self.points = points
            pcls = Pointclouds(points=points)
            drop, idx = point_mesh_face_distance(meshes, pcls)
            triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]
            triangles = triangles_meshes[idx]  
            nearest, stats = self.sa._parse_nearest_projection(triangles, pcls.points_packed())
            self.normals = self.sa._calculate_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])  

        elif self.opt.skinning_mode == 'mlp':
            self.lbs_network = ImplicitNetwork(**self.opt.network, d_out = self.ret)#, d_in = self.inp_posi_enc.out_dim)
            
            if self.offset_net_enabled:
                self.offset_net = _OffsetsNet(offset_pose_dim)
            self.sa = SurfaceAlignedConverter()       
            
        else:
            raise NotImplementedError('Unsupported Deformer.')
    
    @property
    def warp_dim(self):
        return 3

    @property
    def out_dim(self):
        return 84#self.posi_enc.out_dim

    @property
    def diag(self):
        return self.posi_enc.diag
    
    
    def agg(self, density, features, mask, warp):
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
        features = torch.gather(
            features,
            -2,
            indices.unsqueeze(-2).expand(list(features.shape[:-2]) + [1, 15]),
        ).squeeze(-2)
        
        warp = torch.gather(
            warp,
            -2,
            indices.unsqueeze(-2).expand(list(warp.shape[:-2]) + [1, 3]),
        ).squeeze(-2)
        mask = torch.gather(
            mask,
            -1,
            indices,
        ).squeeze(-1)
        return density, features, mask, warp

    def aggregate(self, density, rgb, x_cano, mask, cano_feat):
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

        cano_feat = torch.gather(
            cano_feat,
            -2,
            indices.unsqueeze(-2).expand(list(cano_feat.shape[:-2]) + [1, 16]),
        ).squeeze(-2)
        return density, rgb, x_cano, mask, cano_feat
    
    def get_extra_losses(
        self,
        bones_rest,
        rigid_clusters,
        cond,
        can_pc,
        geometry,
        normals,
        sdf_grads,
        sdf = None,
        pre= False,
        ):
        losses = {}        
        num_batch = 1 #bone_samples.shape[0]
        ##################################
        
        if self.opt.skinning_mode == 'mlp' or self.opt.skinning_mode == 'voxel':
            bone_samples = sample_on_bones(
                bones_rest, n_per_bone=5, range=(0.1, 0.9)
            )  # [n_per_bone, n_bones, 3]
            bone_samples = torch.cat((torch.zeros((bone_samples.shape[0], 1, bone_samples.shape[-1])).to(bone_samples), bone_samples), dim =-2)
            if pre:
                w_pd = self.query_weights_grad(bone_samples, cond) #[5,24,24]
            else:    
                w_pd = self.query_weights(bone_samples, cond) #[5,24,24]
            rigid_clusters = torch.arange(self.ret)
            w_gt = (
                F.one_hot(rigid_clusters, num_classes=w_pd.shape[-1])
                .expand_as(w_pd)
                .to(w_pd)
            )
            # w_gt[:,0, 0] = w_gt[:,0, 0] * 0.0
            loss_bone_occ = F.mse_loss(w_pd, w_gt)
            losses["loss_bone_occ"] = loss_bone_occ
        
        #########################################
        if self.opt.skinning_mode == 'voxel':
            loss_tv = tv_loss(self.lbs_voxel,l2=True)*((self.opt.res//32)**3)
            losses["loss_tv"] = loss_tv
            
        ########################################
        
        if pre:
            ##################################
            scaling_factor = 1.2
            # faces = self.sa.faces_idx[None, ...].repeat(len(self.sa.verts), 1, 1)
            meshes = Meshes(verts=self.sa.verts.float().unsqueeze(0)*scaling_factor, faces=self.sa.faces_idx.unsqueeze(0))
            points = ops.sample_points_from_meshes(meshes, num_samples = 200000)

            pcls = Pointclouds(points=points)
            drop, idx = point_mesh_face_distance(meshes, pcls)
            triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]
            triangles = triangles_meshes[idx]  
            nearest, stats = self.sa._parse_nearest_projection(triangles, pcls.points_packed())
            normals = self.sa._calculate_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])
            ##################################
            outs = geometry(points.squeeze(0), output_normal = True)
            points_normal, points_sdf = outs['normal'], outs['sdf_grad']
            loss_normals = F.mse_loss(points_normal, normals)
            # print("sdf loss", loss_normals)
            sdf_consistency = torch.mean(outs["sdf"] * outs["sdf"])
            canonical_normals = torch.nn.functional.normalize(points_sdf, dim=1)
            assert canonical_normals.shape[1] == 3
            assert len(canonical_normals.shape) == 2
            eikonal_loss = ((canonical_normals.norm(2, dim=1) - 1) ** 2).mean()
            losses['loss_normals'] = loss_normals
            losses['loss_sdf_consistency'] = sdf_consistency
            losses['eikonal_loss'] = eikonal_loss*100
            # print("eikonal loss", eikonal_loss)
            
            
            
        else:
            ##################################
            scaling_factor = 1.2
            meshes = Meshes(verts=self.sa.verts.float().unsqueeze(0)*scaling_factor, faces=self.sa.faces_idx.unsqueeze(0))
            points = ops.sample_points_from_meshes(meshes, num_samples = 10000)
            self.points = points
            pcls = Pointclouds(points=points)
            drop, idx = point_mesh_face_distance(meshes, pcls)
            triangles_meshes = meshes.verts_packed()[meshes.faces_packed()]
            triangles = triangles_meshes[idx]  
            nearest, stats = self.sa._parse_nearest_projection(triangles, pcls.points_packed())
            self.normals = self.sa._calculate_normals(pcls.points_packed(), nearest, triangles, meshes.verts_normals_packed()[meshes.faces_packed()][idx])  
            ##################################
            outs = geometry(self.points.squeeze(0), output_normal = True)
            
            points_normal, points_sdf, sdf_const = outs['normal'], outs['sdf_grad'], outs['sdf']
        
            loss_normals = 0.0#F.mse_loss(points_normal, self.normals)
            canonical_normals = torch.nn.functional.normalize(points_sdf, dim=1)
            assert canonical_normals.shape[1] == 3
            assert len(canonical_normals.shape) == 2
            eikonal_loss = ((canonical_normals.norm(2, dim=1) - 1) ** 2).mean() #* 100
            #####################
            sdf_consistency = torch.mean(sdf_const * sdf_const) #* 100
            losses['loss_normals']  = loss_normals * 0.1
            losses['loss_sdf_consistency'] = sdf_consistency
            losses['eikonal_loss'] = eikonal_loss #* 100
            
        #######################################
        if self.offset_net_enabled:
            bone_samples = sample_on_bones(
                bones_rest, n_per_bone=5, range=(0.0, 1.0)
            )  # [n_per_bone, n_bones, 3]
            offsets = self.offset_net(bone_samples, cond["smpl_params"].view(-1))
            offsets[offsets >0.01] = 0.0
            losses["loss_bone_offset"] = (offsets**2).mean() * 100
        # breakpoint()
        return losses

    def forward(self, xd,
                x_cov: Tensor,
                bones_world: Bones,
                bones_cano: Bones, rigid_clusters, cond, eval_mode=False, **kwargs):
        """Given deformed point return its caonical correspondence

        Args:
            xd (tensor): deformed points in batch. shape: [B, N, D] 1023, 64, 3 # 1, 64, 3
            cond (dict): conditional input.
            tfs (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]

        Returns:
            xc (tensor): canonical correspondences. shape: [B, N, I, D]
            others (dict): other useful outputs.
        """
        B, N, _3 = xd.shape
        xd = xd.reshape(1, -1, _3)
    
        
        # # (cano -> bone -> world)
        tfs_fw = bones_world.transforms @ bones_cano.transforms.inverse() # forward skinning
        # (world -> bone -> cano)
        tfs_bw = bones_cano.transforms @ bones_world.transforms.inverse() # inverse skinning
        # tfs_fw = tfs_fw.expand([B, 24, 4 ,4])
        transforms = torch.cat(
                [torch.eye(4)[None, :, :].to(tfs_fw), tfs_fw], dim=-3
            )
        if self.opt.bones_23:
            tfs_fw = transforms.expand([1, 24, 4 ,4])
        else:
            tfs_fw = transforms.expand([1, 25, 4 ,4])
        tfs = tfs_fw
        xc_opt, others = self.search(xd, cond, tfs, eval_mode=eval_mode)
        n_batch, n_point, n_init, n_dim = xc_opt.shape
        
        xc_opt[~others['valid_ids']] = 0

        # do not back-prop through broyden
        xc_opt = xc_opt.detach()        
        if self.offset_net_enabled:
            offsets = self.offset_net(xc_opt, cond['smpl_params'].view(-1), mask=others['valid_ids'])
            offsets[offsets >0.01] = 0.0
            xc_opt = xc_opt + offsets 
        
        if eval_mode or self.opt.skinning_mode == 'preset':
            xc_opt = xc_opt.view(B, N, -1, 3)
            if x_cov is not None:
                x_cov = (
                    x_cov.unsqueeze(-2).expand(list(xc_opt.shape[:-1]) + [3])
                    if self.posi_enc.diag
                    else x_cov.unsqueeze(-3).expand(list(xc_opt.shape[:-1]) + [3, 3])
                )
                x_enc = self.posi_enc((xc_opt, x_cov))
            else:
                # x_enc = self.posi_enc(xc_opt)
                mask = others['valid_ids']
            
            mask = others['valid_ids'].view(xc_opt.shape[:-1]) 
            others['J_inv'] = others['J_inv'].view(list(xc_opt.shape[:-1]) + [3, 3])
            return mask, xc_opt, others, mask
        
        # offsets = self.offset_net(xc_opt, cond['smpl_params'].view(-1), mask=others['valid_ids'])
        xd_opt, weights = self.forward_skinning(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs, mask=others['valid_ids'].flatten(1,2))
        
        # xd_opt = xd_opt + offsets.view( n_batch, n_point *n_init, n_dim)
        
        if not self.opt.use_slow_grad:
            grad_inv = others['J_inv'].reshape(n_batch, n_point * n_init, 3,3)
        else:
            grad_inv = self.gradient(xc_opt.reshape((n_batch, n_point * n_init, n_dim)), cond, tfs).inverse().detach()

        correction = xd_opt - xd_opt.detach()
        correction[correction > 0.05] = 0.0
        correction = bmv(-grad_inv.flatten(0,1), correction.unsqueeze(-1).flatten(0,1)).squeeze(-1).reshape(xc_opt.shape)
        
        xc = xc_opt + correction
        xc[~others['valid_ids']] = 0
        xc = xc.reshape(n_batch, n_point, n_init, n_dim)
        if x_cov is not None:
            x_cov = (
                x_cov.unsqueeze(-2).expand(list(xc.shape[:-1]) + [3])
                if self.posi_enc.diag
                else x_cov.unsqueeze(-3).expand(list(xc.shape[:-1]) + [3, 3])
            )
            x_enc = self.posi_enc((xc, x_cov))
        else:
            mask = others['valid_ids']
            
            # x_enc = self.posi_enc(xc)
            mask = others['valid_ids'].view(xc.shape[:-1]) 
            others['J_inv'] = others['J_inv'].view(list(xc.shape[:-1]) + [3, 3])
            
        return mask, xc, others, mask

    def precompute(self, tfs=None,cond = None, recompute_skinning=True):

        if recompute_skinning or not hasattr(self,"lbs_voxel_final"):

            if self.opt.skinning_mode == 'mlp':
                self.mlp_to_voxel(cond)
            
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

        voxel_d, voxel_J = self.precompute(tfs,cond = cond, recompute_skinning=not eval_mode)

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

        return xd, w

    def mlp_to_voxel(self, cond = None):

        d, h, w = self.opt.res//self.opt.z_ratio, self.opt.res, self.opt.res
        # enc_inp = self.inp_posi_enc(self.grid_denorm)
        lbs_voxel_final = self.lbs_network(self.grid_denorm, cond, None)
        # lbs_voxel_final = self.lbs_network(self.grid_denorm, {}, None)
        lbs_voxel_final = self.opt.soft_blend * lbs_voxel_final

        if self.opt.softmax_mode == "hierarchical":
            lbs_voxel_final = hierarchical_softmax(lbs_voxel_final)
        else:
            lbs_voxel_final = F.softmax(lbs_voxel_final, dim=-1)

        self.lbs_voxel_final = lbs_voxel_final.permute(0,2,1).reshape(1,self.ret,d,h,w)

    def voxel_to_voxel(self):

        lbs_voxel_final = self.lbs_voxel*self.opt.soft_blend

        self.lbs_voxel_final = F.softmax(lbs_voxel_final, dim=1)

    def query_weights_grad(self, xc, cond=None, mask=None, mode='bilinear'):

        # if not hasattr(self,"lbs_voxel_final"):
        if self.opt.skinning_mode == 'mlp':
            self.mlp_to_voxel(cond)
        elif self.opt.skinning_mode == 'voxel':
            self.voxel_to_voxel()

        xc_norm = (xc + self.offset) * self.scale
        
        w = F.grid_sample(self.lbs_voxel_final.expand(xc.shape[0],-1,-1,-1,-1), xc_norm.unsqueeze(2).unsqueeze(2), align_corners=self.opt.align_corners, mode=mode, padding_mode='zeros')
        
        w = w.squeeze(-1).squeeze(-1).permute(0,2,1)
        
        return w
    
    def query_weights(self, xc, cond=None, mask=None, mode='bilinear'):

        if not hasattr(self,"lbs_voxel_final"):
            if self.opt.skinning_mode == 'mlp':
                self.mlp_to_voxel(cond)
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