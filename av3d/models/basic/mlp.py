# Copyright (c) Meta Platforms, Inc. and affiliates.
""" The MLP for NeRF / Mip-NeRF """
import torch
import torch.nn as nn


def dense_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation=torch.nn.ReLU(),  # The activation function.
        skip_layer: int = 4,  # The layer to add skip layers to.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of sigma channels.
        num_ao_channels: int = 0,  # The number of ambient occlusion channels.
        input_dim: int = 60,  # The number of input tensor channels.
        condition_dim: int = 27,  # The number of conditional tensor channels.
        # The number of conditional tensor channels for ambient.
        condition_ao_dim: int = 0,
    ):
        super().__init__()
        self.net_depth = net_depth # 4
        self.net_width = net_width # 128
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.num_ao_channels = num_ao_channels
        self.input_dim = input_dim # 27
        self.condition_dim = condition_dim
        self.condition_ao_dim = condition_ao_dim

        self.input_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.input_layers.append(dense_layer(in_features, self.net_width))
            if i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        hidden_features = in_features
        self.sigma_layer = dense_layer(hidden_features, self.num_sigma_channels)

        if self.condition_dim > 0:
            self.bottleneck_layer = dense_layer(hidden_features, self.net_width)
            self.condition_layers = nn.ModuleList()
            in_features = self.net_width + self.condition_dim
            for _ in range(self.net_depth_condition):
                self.condition_layers.append(
                    dense_layer(in_features, self.net_width_condition)
                )
                in_features = self.net_width_condition
        if self.num_rgb_channels > 0:
            in_features = in_features #+ in_features
            self.rgb_layer = dense_layer(in_features, self.num_rgb_channels)
            self.cano_layer = dense_layer(in_features, 16)

        if self.condition_ao_dim > 0:
            hidden_features = hidden_features #+ hidden_features
            self.bottleneck_ao_layer = dense_layer(
                hidden_features, self.net_width
            )
            self.condition_ao_layers = nn.ModuleList()
            in_features = self.net_width + self.condition_ao_dim
            for _ in range(self.net_depth_condition):
                self.condition_ao_layers.append(
                    dense_layer(in_features, self.net_width_condition)
                )
                in_features = self.net_width_condition
        if self.num_ao_channels > 0:
            self.ao_layer = dense_layer(in_features, self.num_ao_channels)
    
    def query_normal(self, x, pos_enc: nn.Module, delta=0.02):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            x_enc = pos_enc(x)
            # x_warp.requires_grad_(True)
            sigma = self.query_sigma(x_enc, return_feat=False)
            alpha = 1 - torch.exp(-delta * torch.relu(sigma))
            normal = torch.autograd.grad(
                outputs=alpha,
                inputs=x,
                grad_outputs=torch.ones_like(alpha, requires_grad=False, device=alpha.device),
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        # breakpoint()
        return normal[:,:3], alpha

    def query_sigma(self, x, masks=None, return_feat=True):
        """Query the sigma (and the features) of the points in canonical space.

        :params x: [..., input_dim]
        :params masks: Optional [...,]
        :return
            raw_sigma: [..., 1]
            raw_feat: Optional [..., D]
        """
        if masks is None:
            return self._query_sigma(x, return_feat=return_feat)
        else:
            if return_feat:
                _raw_sigma, _raw_feat = self._query_sigma(
                    x[masks], return_feat=True
                )
                raw_sigma = torch.zeros(
                    (*x.shape[:-1], _raw_sigma.shape[-1]),
                    dtype=_raw_sigma.dtype,
                    device=_raw_sigma.device,
                )
                raw_sigma[masks] = _raw_sigma
                raw_feat = torch.zeros(
                    (*x.shape[:-1], _raw_feat.shape[-1]),
                    dtype=_raw_feat.dtype,
                    device=_raw_feat.device,
                )
                raw_feat[masks] = _raw_feat
                return raw_sigma, raw_feat
            else:
                _raw_sigma = self._query_sigma(x[masks], return_feat=False)
                raw_sigma = torch.zeros(
                    (*x.shape[:-1], _raw_sigma.shape[-1]),
                    dtype=_raw_sigma.dtype,
                    device=_raw_sigma.device,
                )
                raw_sigma[masks] = _raw_sigma
                return raw_sigma

    def _query_sigma(self, x, return_feat=True):
        """Query the sigma (and the features) of the points in canonical space.

        :params x: [..., input_dim]
        :return
            raw_sigma: [..., 1]
            raw_feat: Optional [..., D]
        """
        inputs = x # [1034, 60]
        for i in range(self.net_depth): # 8
            x = self.input_layers[i](x) # 1034, 256
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_feat = x # 1034, 256
        raw_sigma = self.sigma_layer(x) # 1034, 1
        if return_feat:
            return raw_sigma, raw_feat
        else:
            return raw_sigma

    def query_rgb(self, raw_feat, masks=None, condition=None):
        """Query the view-dependent rgb of the samples in world space.

        The `raw_feat` should be already interpolated in the world space.
        The `condition` is in general the view dirs in the world space.

        :params raw_feat: [..., D]
        :params condition: Optional [..., condition_dim]
        :return raw_rgb: [..., 3]
        """
        if masks is None:
            return self._query_rgb(raw_feat, condition)
        else:
            _raw_rgb, _raw_cano = self._query_rgb(
                raw_feat[masks],
                condition[masks] if condition is not None else None,
            )
            raw_rgb = torch.zeros(
                (*raw_feat.shape[:-1], _raw_rgb.shape[-1]),
                dtype=_raw_rgb.dtype,
                device=_raw_rgb.device,
            )
            
            raw_cano = torch.zeros(
                (*raw_feat.shape[:-1], _raw_cano.shape[-1]),
                dtype=_raw_cano.dtype,
                device=_raw_cano.device,
            )
            raw_rgb[masks] = _raw_rgb
            raw_cano[masks] = _raw_cano
            
            return raw_rgb, raw_cano

    def _query_rgb(self, raw_feat, condition=None):
        """Query the view-dependent rgb of the samples in world space.

        The `raw_feat` should be already interpolated in the world space.
        The `condition` is in general the view dirs in the world space.

        :params raw_feat: [..., D]
        :params condition: Optional [..., condition_dim]
        :return raw_rgb: [..., 3]
        """
        x = raw_feat
        if condition is not None:
            # breakpoint()
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
            for i in range(self.net_depth_condition):
                x = self.condition_layers[i](x)
                x = self.net_activation(x)
        raw_rgb = self.rgb_layer(x)
        raw_cano = self.cano_layer(x)
        
        return raw_rgb, raw_cano

    def query_ao(self, raw_feat, masks=None, condition=None):
        """Query the pose-dependent ambient occlusion of the samples in world space.

        The `raw_feat` should be already interpolated in the world space.
        The `condition` is in general the pose in the world space.

        :params raw_feat: [..., D]
        :params condition: Optional [..., condition_dim]
        :return raw_rgb: [..., 3]
        """
        if masks is None:
            return self._query_ao(raw_feat, condition)
        else:
            _raw_ao = self._query_ao(
                raw_feat[masks],
                condition[masks] if condition is not None else None,
            )
            raw_ao = torch.zeros(
                (*raw_feat.shape[:-1], _raw_ao.shape[-1]),
                dtype=_raw_ao.dtype,
                device=_raw_ao.device,
            )
            raw_ao[masks] = _raw_ao
            return raw_ao

    def _query_ao(self, raw_feat, condition=None):
        """Query the pose-dependent ambient occlusion of the samples in world space.

        The `raw_feat` should be already interpolated in the world space.
        The `condition` is in general the pose in the world space.

        :params raw_feat: [..., D]
        :params condition: Optional [..., condition_dim]
        :return raw_rgb: [..., 3]
        """
        x = raw_feat
        if condition is not None:
            bottleneck = self.bottleneck_ao_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
            for i in range(self.net_depth_condition):
                x = self.condition_ao_layers[i](x)
                x = self.net_activation(x)
        raw_ao = self.ao_layer(x)
        return raw_ao

    def forward(self, samples, masks=None, cond_view=None, cond_extra=None):
        """
        :params samples: [B, ..., Dx] encoded.
        :params masks: Optional [B, ...] shows the valid candidates.
        :params cond_view: Optional [B, 3] or [B, ..., Dv]
        :params cond_extra: Optional [B, D] or [B, ..., D]
        :return
            raw_sigma [B, ..., 1], raw_rgb [B, ..., 3]
        """
        # cond_extra = self.gc(cond_extra)
        assert samples.shape[-1] == self.input_dim, (
            "Shape of the input samples should match with the self.input_dim. "
            "Got %s v.s. %d" % (str(samples.shape), self.input_dim)
        )
        if masks is not None:
            assert samples.shape[:-1] == masks.shape, (
                "Shape of the input samples should match with the masks. "
                "Got %s v.s. %s" % (str(samples.shape), str(masks.shape))
            )
            # if neigh is not None:
            #     assert neigh.shape[:-1] == masks.shape, (
            #         "Shape of the input samples should match with the masks. "
            #         "Got %s v.s. %s" % (str(neigh.shape), str(masks.shape))
            #     )
            
        condition = []
        if cond_view is not None:
            assert cond_view.dim() in [1, 2, samples.dim()]
            if cond_view.dim() == 1:
                cond_view = cond_view.expand([samples.shape[0], -1])
            if cond_view.dim() == 2:
                cond_view = cond_view.view(
                    [cond_view.shape[0]]
                    + [1] * (samples.dim() - cond_view.dim())
                    + [cond_view.shape[-1]]
                ).expand(list(samples.shape[:-1]) + [cond_view.shape[-1]])
            condition.append(cond_view)
        if cond_extra is not None: 
            assert cond_extra.dim() in [1, 2, samples.dim()]
            if cond_extra.dim() == 1:
                cond_extra = cond_extra.expand([samples.shape[0], -1])
            if cond_extra.dim() == 2:
                cond_extra = cond_extra.view(
                    [cond_extra.shape[0]]
                    + [1] * (samples.dim() - cond_extra.dim())
                    + [cond_extra.shape[-1]]
                ).expand(list(samples.shape[:-1]) + [cond_extra.shape[-1]])
            condition.append(cond_extra)

        # 211, 64, 6, 60
        raw_sigma, raw_feat = self.query_sigma(samples, masks, return_feat=True)
        # if neigh is not None:
        #     raw_sigma_1, raw_feat_1 = self.query_sigma(neigh, masks, return_feat=True)
        # breakpoint()
        
        # raw_sigma = raw_sigma*0.5 + raw_sigma_1*0.5
        # raw_feat = torch.cat([raw_feat, raw_feat_1], dim = -1)
        if self.condition_ao_dim == 0:
            # pass all conditions to rgb branch
            condition = torch.cat(condition, dim=-1) if len(condition) else None
            if self.condition_dim > 0:
                assert (
                    condition.shape[-1] == self.condition_dim
                ), "Shape of condition (%s) does not seem right!" % str(
                    condition.shape
                )
            raw_rgb, raw_cano = self.query_rgb(raw_feat, masks, condition=condition)
            return raw_rgb, raw_sigma

        else:
            # the cond_extra is for ambient occulsion
            raw_rgb, raw_cano = self.query_rgb(raw_feat, masks, condition=cond_view)
            raw_ao = self.query_ao(raw_feat, masks, condition=cond_extra)
            return raw_rgb, raw_sigma, raw_ao, raw_cano


class StraightMLP(MLP):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_activation: str = torch.nn.ReLU(),  # The activation function.
        skip_layer: int = 4,  # The layer to add skip layers to.
        input_dim: int = 60,  # The number of input tensor channels.
        output_dim: int = 1,  # The number of output tensor channels.
    ):
        super().__init__(
            net_depth=net_depth,
            net_width=net_width,
            net_depth_condition=0,
            net_width_condition=0,
            net_activation=net_activation,
            skip_layer=skip_layer,
            num_rgb_channels=0,
            num_sigma_channels=output_dim,
            input_dim=input_dim,
            condition_dim=0,
        )

    def forward(self, x, mask=None):
        return self.query_sigma(x, mask, return_feat=False)
