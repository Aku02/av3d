import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from av3d.plenoxels.ops.interpolation import grid_sample_wrapper
from av3d.plenoxels.raymarching.spatial_distortions import SpatialDistortion

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import pytorch3d.ops as ops
from pytorch3d.structures import Meshes, Pointclouds

from av3d.models.projection.map_utils import load_model, point_mesh_face_distance, barycentric_to_points, points_to_barycentric

class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply

def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            ) -> torch.Tensor:
    coo_combs = list(itertools.combinations(
        range(pts.shape[-1]), grid_dimensions)
    )
    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class KPlaneField(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        use_appearance_embedding: bool,
        appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        linear_decoder_layers: Optional[int],
        num_images: Optional[int],
        use_ao: Optional[bool],
        use_direction: Optional[bool],
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = trunc_exp #density_activation
        self.linear_decoder = linear_decoder
        # 1. Init planes
        # self.grids = nn.ModuleList()
        self.feature_dim = 0
        self.dp = False
        self.use_ao = use_ao
        self.use_direction = use_direction

        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config #[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            # self.grids.append(gp)
        # log.info(f"Initialized model grids: {self.grids}")

        # 2. Init appearance code-related parameters
        self.use_average_appearance_embedding = True  # for test-time
        self.use_appearance_embedding = use_appearance_embedding
        self.num_images = num_images
        self.appearance_embedding = None
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_embedding_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = nn.Embedding(self.num_images, self.appearance_embedding_dim)
        else:
            self.appearance_embedding_dim = 0

        # 3. Init decoder params
        if self.use_direction:
            self.direction_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        # 3. Init decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB
            # This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,#self.direction_encoder.n_output_dims,
                n_output_dims=(3 * self.feature_dim) + (3*78),
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
            
            self.cano_net = tcnn.Network(
                n_input_dims=self.feature_dim + 78 +15,
                n_output_dims=16,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "ReLU",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
        else:
            self.geo_feat_dim = 16 #15
            ###################################################
            if self.use_ao:
                self.sigma_net = CondMLP(input_dim =  256 + 78, out_dim=1)
            ###################################################
            self.in_dim_color = (
                    # self.direction_encoder.n_output_dims
                    + 3#self.geo_feat_dim
                    + 3
                    # + 78
                    # + 15
                    # + 16
            )
            if self.use_direction:
                self.in_dim_color = (
                    6
                    + self.direction_encoder.n_output_dims
                )
                
            ########################################
            self.color_net = RgbMLP(input_dim = self.in_dim_color, out_dim=3)
            ########################################

    def query_sigma(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts)
            pts = pts / 2  # from [-2, 2] to [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        if timestamps is not None:
            # timestamps = timestamps.expand_as(pts[:, :, :1]) #timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, n_init, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, n_init, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.grids,  # noqa
            grid_dimensions=self.grid_config["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None)

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            if features.shape[0] == 0:
                return torch.zeros((pts.shape[0])).to(features), torch.zeros((pts.shape[0], 15)).to(features)
            else:

                features = self.sigma_net(features)

                return features

    def forward(self,
                pts: torch.Tensor,
                directions: Optional[torch.Tensor] = None,
                # cond_feat:torch.Tensor,
                cond_extra: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None,
                verts:Optional[torch.Tensor] = None,
                normals: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None):
        camera_indices = None
        if self.use_appearance_embedding:
            if timestamps is None:
                raise AttributeError("timestamps (appearance-ids) are not provided.")
            camera_indices = timestamps
            timestamps = None
        normals = normals.detach()
        if self.use_direction:
            encoded_directions = self.direction_encoder(directions)
        if self.linear_decoder:
            color_features = [features]
        else:
            if self.use_direction:
                color_features = [pts.view(-1, 3), normals.view(-1, 3), encoded_directions]
            else:
                color_features = [pts.view(-1, 3), normals.view(-1, 3)]
        if self.use_appearance_embedding:
            if camera_indices.dtype == torch.float32:
                # Interpolate between two embeddings. Currently they are hardcoded below.
                #emb1_idx, emb2_idx = 100, 121  # trevi
                emb1_idx, emb2_idx = 11, 142  # sacre
                emb_fn = self.appearance_embedding
                emb1 = emb_fn(torch.full_like(camera_indices, emb1_idx, dtype=torch.long))
                emb1 = emb1.view(emb1.shape[0], emb1.shape[2])
                emb2 = emb_fn(torch.full_like(camera_indices, emb2_idx, dtype=torch.long))
                emb2 = emb2.view(emb2.shape[0], emb2.shape[2])
                embedded_appearance = torch.lerp(emb1, emb2, camera_indices)
            elif self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                if hasattr(self, "test_appearance_embedding"):
                    embedded_appearance = self.test_appearance_embedding(camera_indices)
                elif self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.appearance_embedding.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = embedded_appearance.view(-1, 1, ea_dim).expand(n_rays, n_samples, n_init, -1).reshape(-1, ea_dim)
            if not self.linear_decoder:
                color_features.append(embedded_appearance)
        color_features = torch.cat(color_features, dim=-1)
        if self.linear_decoder:
            if self.use_appearance_embedding:
                basis_values = self.color_basis(torch.cat([directions, embedded_appearance], dim=-1))
            else:

                basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, n_init, 3)
            cano_feat  = self.cano_net(color_features).to(directions).view(n_rays, n_samples, n_init, 16)
        else:
            rgb, feats = self.color_net(color_features)#.to(pts)#.view(n_rays, n_samples, n_init, 3)
            if self.use_ao:
                ao_features = [feats, cond_extra[None, :].expand(feats.shape[0], -1)]
                ao_features = torch.cat(ao_features, dim=-1)
                ao = self.sigma_net(ao_features).to(pts)#.view(n_rays, n_samples, n_init, 1)
                rgb = rgb * ao
            # rgb = torch.sigmoid(rgb)
            # cano_feat  = self.cano_net(color_features).to(directions)#.view(n_rays, n_samples, n_init, 16)
        return {"rgb": rgb}

    def get_params(self):
        field_params = {k: v for k, v in self.grids.named_parameters(prefix="grids")}
        nn_params = [
            self.sigma_net.named_parameters(prefix="sigma_net"),
            self.direction_encoder.named_parameters(prefix="direction_encoder"),
        ]
        if self.linear_decoder:
            nn_params.append(self.color_basis.named_parameters(prefix="color_basis"))
        else:
            nn_params.append(self.color_net.named_parameters(prefix="color_net"))
        nn_params = {k: v for plist in nn_params for k, v in plist}
        other_params = {k: v for k, v in self.named_parameters() if (
            k not in nn_params.keys() and k not in field_params.keys()
        )}
        return {
            "nn": list(nn_params.values()),
            "field": list(field_params.values()),
            "other": list(other_params.values()),
        }


import torch.nn as nn
import torch.nn.functional as F
class CondMLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()

        self.l0 = nn.Linear(input_dim, 256)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.l1 = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.l0(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.l1(x)
        x = torch.sigmoid(x)
        return x

class RgbMLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()

        self.l0 = nn.Linear(input_dim, 256)
        self.res1 = ResBlock()
        self.res2 = ResBlock()
        self.res3 = ResBlock()
        self.res4 = ResBlock()
        self.l1 = nn.Linear(256, out_dim)
        

    def forward(self, x):

        x = self.l0(x)
        x = self.res1(x)
        x = self.res2(x)
        xz = self.res3(x)
        x = self.res4(xz)
        x = self.l1(x)
        x = torch.sigmoid(x)
        return x, xz

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.lz = nn.Linear(256, 256)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, 256)

    def forward(self, x):
        z = F.relu(self.lz(x))
        res = x + z
        x = F.relu(self.l1(res))
        x = F.relu(self.l2(x)) + res
        return x