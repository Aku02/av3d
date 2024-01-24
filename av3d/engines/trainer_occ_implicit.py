# Copyright (c) Meta Platforms, Inc. and affiliates.
import functools
import logging
import math
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig
from av3d.engines.abstract import AbstractEngine
from av3d.utils.evaluation_backup import eval_epoch
from av3d.utils.structures import namedtuple_map
from av3d.utils.training import (
    clean_up_ckpt,
    compute_psnr_from_mse,
    learning_rate_decay,
    save_ckpt,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from av3d.lib.model.smpl import SMPLServer
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import gc
import torch.nn as nn
from av3d.utils.meshing import generate_mesh
import trimesh
from av3d.lib.model.helpers import skinning
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes, Pointclouds
from av3d.models.projection.map_utils import load_model
from pytorch3d.utils import ico_sphere
import pytorch3d.ops as ops
LOGGER = logging.getLogger(__name__)


def default_collate_fn(data):
    return data[0]


class Trainer(AbstractEngine):
    def __init__(
        self,
        local_rank: int,
        world_size: int,
        cfg: DictConfig,
    ) -> None:
        super().__init__(local_rank, world_size, cfg)
        # setup tensorboard
        # must be after model resuming for `self.init_step`` to be updated.
        self.tb_writer = SummaryWriter(
            log_dir=self.save_dir, purge_step=self.init_step
        )
        self.tb_writer.add_text("cfg", str(self.cfg), 0)

        self.learning_rate_fn = functools.partial(
            learning_rate_decay,
            lr_init=cfg.lr_init,
            lr_final=cfg.lr_final,
            max_steps=cfg.max_steps,
            lr_delay_steps=cfg.lr_delay_steps,
            lr_delay_mult=cfg.lr_delay_mult,
        )

        if self.cfg.distributed:
            self.model = DDP(self.model, device_ids=[local_rank])
            torch.distributed.barrier(device_ids=[self.local_rank])  # sync

    def build_model(self):
        LOGGER.info("* Creating Model.")
        current_seed = torch.initial_seed()
        torch.manual_seed(1234)  # same parameters for multi-gpus
        model = instantiate(self.cfg.model).to(self.device)
        print(model)
        if self.cfg.dataset.version == 1 or self.cfg.dataset.version == 3:
            for p in model.pos_enc.parameters():
                p.requires_grad_(False)
            
            param_ = [ 
                {"params": list(model.mlp_coarse.parameters()), "_name": "mlp_course", "lr": self.cfg.lr_init},
                {"params": list(model.geometry.parameters()), "_name": "geometry", "lr": self.cfg.lr_init},
                {"params": list(model.variance.parameters()), "_name": "variance", "lr": 0.001},]
        else:
            param_ = [ 
                {"params": list(model.mlp_coarse.parameters()), "_name": "mlp_course", "lr": self.cfg.lr_init},
                {"params": list(model.pos_enc.parameters()), "_name": "pos_enc", "lr": self.cfg.lr_init},
                {"params": list(model.geometry.parameters()), "_name": "geometry", "lr": self.cfg.lr_init},
                {"params": list(model.variance.parameters()), "_name": "variance", "lr": 0.001},]
               
        optimizer = torch.optim.Adam(
            param_,
            lr=self.cfg.lr_init,
            weight_decay=self.cfg.weight_decay_mult,
        )
        # breakpoint()
        torch.manual_seed(current_seed)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(self.device)
        for param in self.lpips.parameters(): param.requires_grad=False
        # self.loss_dice = BCEDiceLoss()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.eps = 1e-6
        return model, optimizer

    def build_dataset(self):
        LOGGER.info("* Creating Dataset.")
        dataset = {
            split: instantiate(self.cfg.dataset, split=split)
            for split in [self.cfg.train_split]
        }
        dataset.update(
            {
                split: instantiate(
                    self.cfg.dataset,
                    split=split,
                    num_rays=None,
                    cache_n_repeat=None,
                )
                for split in self.cfg.eval_splits
            }
        )
        meta_data = {
            split: dataset[split].build_pose_meta_info()
            for split in dataset.keys()
        }
        return dataset, meta_data
    
    def pre(self) -> float:  # noqa
        LOGGER.info("Start Running in Rank %d!" % self.local_rank)
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset[self.cfg.train_split],
            num_workers=0,
            batch_size=1,
            collate_fn=default_collate_fn,
        )
        train_dataloader_iter = iter(train_dataloader)
        stats_trace = deque([], maxlen=100)
        reset_timer = True
        is_main_thread = self.local_rank % self.world_size == 0
        LOGGER.info("Pretrain Begins")
        data = next(train_dataloader_iter)
        for step in range(self.init_step, self.cfg.pretrain_steps):
            # try:
            #     data = next(train_dataloader_iter)
            # except StopIteration:
            #     train_dataloader_iter = iter(train_dataloader)
            #     data = next(train_dataloader_iter)
            if reset_timer:
                t_loop_start = time.time()
                reset_timer = False
            lr = self.learning_rate_fn(step)
            stats = self.pretrain_step(data, lr, step)  
            LOGGER.info(f"Pretrain step: %d"%(step))
        LOGGER.info("Pretrain Over")
        del train_dataloader
        return 1.0

    def run(self) -> float:  # noqa
        LOGGER.info("Start Running in Rank %d!" % self.local_rank)
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset[self.cfg.train_split],
            num_workers=0,
            batch_size=1,
            collate_fn=default_collate_fn,
        )
        train_dataloader_iter = iter(train_dataloader)

        stats_trace = deque([], maxlen=100)
        reset_timer = True
        is_main_thread = self.local_rank % self.world_size == 0

        for step in range(self.init_step, self.max_steps + 1):
            try:
                data = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                data = next(train_dataloader_iter)
            if reset_timer:
                t_loop_start = time.time()
                reset_timer = False
            lr = self.learning_rate_fn(step)
            stats = self.train_step(data, lr, step)
            stats_trace.append(stats)
            
            if is_main_thread and step % self.cfg.print_every == 0:
                avg_loss = sum([s["loss"] for s in stats_trace]) / len(
                    stats_trace
                )
                avg_psnr = sum([s["psnr"] for s in stats_trace]) / len(
                    stats_trace
                )
                steps_per_sec = self.cfg.print_every / (
                    time.time() - t_loop_start
                )
                rays_per_sec = (
                    self.cfg.dataset.num_rays * steps_per_sec * self.world_size
                )
                precision = int(np.ceil(np.log10(self.cfg.max_steps))) + 1
                LOGGER.info(
                    ("{:" + "{:d}".format(precision) + "d}").format(step)
                    + f"/{self.cfg.max_steps:d}: "
                    + "".join(
                        [
                            f"{k}={v:0.4f} (Avg {sum([s[k] for s in stats_trace]) / len(stats_trace):0.4f}), "
                            for k, v in stats.items()
                            if "loss" in k and v > 0
                        ]
                    )
                    + f"lr={lr:0.2e}, "
                    + f"{rays_per_sec:0.0f} rays/sec"
                )
                reset_timer = True
                for k, v in stats.items():
                    self.tb_writer.add_scalar(k, v, step)
                self.tb_writer.add_scalar("train_avg_loss", avg_loss, step)
                self.tb_writer.add_scalar("train_avg_psnr", avg_psnr, step)
                self.tb_writer.add_scalar("learning_rate", lr, step)
                self.tb_writer.add_scalar(
                    "train_steps_per_sec", steps_per_sec, step
                )
                self.tb_writer.add_scalar(
                    "train_rays_per_sec", rays_per_sec, step
                )

            if is_main_thread and step % self.cfg.save_every == 0:
                LOGGER.info("* Saving")
                save_ckpt(self.ckpt_dir, step, self.model, self.optimizer)
                clean_up_ckpt(self.ckpt_dir, 5)

            # evaluation on epoch.
            if (
                self.cfg.eval_every > 0
                and step % self.cfg.eval_every == 0
                and step > 0
            ):
                for eval_split in self.cfg.eval_splits:
                    LOGGER.info("* Evaluation on split %s." % eval_split)
                    val_dataset = self.dataset[eval_split]
                    eval_render_every = math.ceil(
                        len(val_dataset)
                        / (self.world_size * self.cfg.eval_per_gpu)
                    )
                    metrics = eval_epoch(
                        self.model,
                        val_dataset,
                        data_preprocess_func=lambda x: self._preprocess(
                            x, eval_split
                        ),
                        render_every=eval_render_every,
                        test_chunk=self.cfg.test_chunk,
                        save_dir=os.path.join(
                            self.save_dir, self.cfg.eval_cache_dir, eval_split
                        ),
                        local_rank=self.local_rank,
                        world_size=self.world_size,
                        step = step,
                    )
                    self.model.train()
                    self.tb_writer.add_scalar(
                        "%s_psnr_eval" % eval_split, metrics["psnr"], step
                    )
                    self.tb_writer.add_scalar(
                        "%s_ssim_eval" % eval_split, metrics["ssim"], step
                    )
                    if is_main_thread:
                        # save the metrics and print
                        with open(
                            os.path.join(
                                self.save_dir, "%s_metrics_otf.txt" % eval_split
                            ),
                            mode="a",
                        ) as fp:
                            fp.write(
                                "step=%d, test_render_every=%d, psnr=%.4f, ssim=%.4f\n"
                                % (
                                    step,
                                    eval_render_every,
                                    metrics["psnr"],
                                    metrics["ssim"],
                                )
                            )
                        LOGGER.info(
                            f"Eval Epoch {step}: "
                            + f"split = {eval_split} "
                            + f"psnr = {metrics['psnr']:.4f} "
                            + f"ssim = {metrics['ssim']:.4f} "
                        )
        LOGGER.info("Finished Training in Rank %d!" % self.local_rank)
        return 1.0

    def _preprocess(self, data, split):
        # to gpu
        for k, v in data.items():
            if k == "rays":
                data[k] = namedtuple_map(lambda x: x.to(self.device), v)
            elif isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
            else:
                pass
        # update pose info for this frame
        meta_data = self.meta_data[split]
        idx = meta_data["meta_ids"].index(data["meta_id"])
        data["bones_rest"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_rest"]
        )
        data["bones_posed"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_posed"][idx]
        )
        
        if "pose_latent" in meta_data:
            data["pose_latent"] = meta_data["pose_latent"][idx].to(self.device)
            
        return data
    
    def get_bce_loss(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    def get_opacity_sparse_loss(self, acc_map, index_off_surface):
        opacity_sparse_loss = F.mse_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = F.mse_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss
    
    def query_oc(self, x):
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.geometry.forward_sdf(x).reshape(-1,1)
        return {'sdf':mnfld_pred}
    
    def get_deformed_mesh_fast_mode(self, verts, cond, smpl_tfs):
        verts = torch.tensor(verts).unsqueeze(0).cuda().float()
        # offset = self.model.pos_enc.offset_net(verts, cond['smpl_params'].view(-1))
        # verts = verts + offset
        weights = self.model.pos_enc.query_weights(xc = verts, cond = cond, mask = None)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def train_step(self, data, lr, step):
        self.model.train()
        data = self._preprocess(data, split=self.cfg.train_split)
        rays = data.pop("rays")
        if self.cfg.dataset.version == 1:
            height, width = rays[0].shape[:2]
            num_rays = height * width
            rays = namedtuple_map(
                lambda r: r.reshape([num_rays] + list(r.shape[2:])), rays
            )
        pixels = data.pop("pixels")
        mask = data.pop("mask")
        # save a mesh at every 10000 steps
        if step % 10001  == 0 and step!= 0:
            mesh_canonical = generate_mesh(lambda x: self.query_oc(x), self.model.pos_enc.sa.verts, point_batch=10000, res_up=2, res_init=64)
            mesh_canonical = trimesh.smoothing.filter_humphrey(mesh_canonical)
            mm = mesh_canonical.export(f'can_high_{step}.obj')
            
            smpl_params = data["pose_latent"].view(-1, 78)
            cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
            smpl_tfs = data['bones_posed'].transforms @ data['bones_rest'].transforms.inverse()
            transforms = torch.cat(
                    [torch.eye(4)[None, :, :].to(smpl_tfs), smpl_tfs], dim=-3
                )
            smpl_tfs = transforms.expand([1, 24, 4 ,4])
            verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, cond, smpl_tfs)
            mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
            mesh_deformed = trimesh.smoothing.filter_humphrey(mesh_deformed)
            mm = mesh_deformed.export(f'deformed_high_{step}.obj')
        
        ret, extra_info = self.model(rays=rays, randomized=True , step = step, **data)
        if len(ret) not in (1, 2):
            raise ValueError(
                "ret should contain either 1 set of output (coarse only), or 2 sets"
                "of output (coarse as ret[0] and fine as ret[1])."
            )

        # The main prediction is always at the end of the ret list.
        rgb, density, acc, _, _ = ret[-1]

        loss = F.mse_loss(rgb.view(pixels.shape), pixels)*10
       
        loss_den = F.mse_loss(acc.view(mask.shape), mask) * 100#0.1

        loss_sparse = self.get_bce_loss(acc.view(-1))
        extra_info["loss_sparse"] = loss_sparse
        zero_indices = torch.nonzero(mask.view(-1) == 0).to(mask.device)
        loss_out_shape = self.get_opacity_sparse_loss(acc.view(-1), zero_indices)
        one_indices = torch.nonzero(mask.view(-1)).to(mask.device)
        loss_in_shape = self.get_in_shape_loss(acc.view(-1), one_indices)
        
        extra_info["loss_out_shape"] = loss_out_shape
        extra_info["loss_in_shape"] = loss_in_shape
        
        loss_lpips = (self.lpips(rgb.view(pixels.shape).permute(0,3,1,2).clip(max=1), pixels.permute(0,3,1,2)) *10) if self.cfg.dataset.version == 1 or self.cfg.dataset.version == 3 or self.cfg.dataset.version == 4 else 0.0
        extra_info["loss_lpips"] = loss_lpips
        
        
        psnr = compute_psnr_from_mse(loss)
        # ignore - no hierarchical sampling
        if len(ret) > 1:
            # If there are both coarse and fine predictions, we compute the loss for
            # the coarse prediction (ret[0]) as well.
            # rgb_c, _, acc_c, _, feat_c = ret[0]
            loss_c = 0.0 #F.mse_loss(rgb_c, pixels)
            
            psnr_c = 0.0 #compute_psnr_from_mse(loss_c)
        else:
            loss_c = 0.0
            psnr_c = 0.0
            # loss_c_feat = 0.0
            loss_c_den = 0.0
            
        # helper losses on the bones
        
        loss_bone_offset = (
            extra_info["loss_bone_offset"] if self.cfg.model.pos_enc.offset_net_enabled  else 0.0
        )
        loss_bone_occ = (
            extra_info["loss_bone_occ"] if self.cfg.loss_bone_occ_multi > 0 and self.cfg.dataset.version==2 else 0.0
        )
        
        loss_tv = (
            extra_info["loss_tv"] if self.cfg.loss_tv_multi > 0 else 0.0
        )
        loss_normals = (
                extra_info['loss_normals'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        
        sdf_consistency = (
                extra_info['loss_sdf_consistency'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        
        eikonal_loss = (
                extra_info['eikonal_loss'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        

        extra_info["loss_den"] = loss_den
        # extra_info["loss_reg_alpha"] = loss_reg_alpha
        # extra_info["loss_depth_reg"] = loss_depth_reg
        if self.cfg.dataset.version == 3:
            for param in self.optimizer.param_groups:
                if param["_name"] == "variance":
                    param["lr"] = lr * 10
                elif param["_name"] == "geometry":
                    param["lr"] = lr
                else:
                    param["lr"] = lr
            
        else:
            for param in self.optimizer.param_groups:
                if param["_name"] == "variance":
                    param["lr"] = lr * 10
                elif param["_name"] == "geometry":
                    param["lr"] = lr
                elif param["_name"] == "pos_enc":
                    param["lr"] = lr * 0.1
                else:
                    param["lr"] = lr
        self.optimizer.zero_grad()
        

        loss_all = (
            loss
            + loss_den
            + loss_bone_occ * self.cfg.loss_bone_occ_multi * 1
            + loss_tv * self.cfg.loss_tv_multi
            + eikonal_loss
            + sdf_consistency
            + loss_lpips #if self.cfg.dataset.version == 1 or self.cfg.dataset.version == 3 else 0.0
            + loss_sparse
            + loss_out_shape
            + loss_in_shape
            + loss_bone_offset*10
        )
        loss_all.backward()
        self.optimizer.step()
        if step % 10 == 0:
            del loss_all, ret, rays, rgb, acc, density  # replace 'variable' with the variable to be deleted
            torch.cuda.empty_cache()  # clear cache
            gc.collect()  # collect garbage

        return {
            "loss": loss,
            "psnr": psnr,
            "loss_c": loss_c,
            "psnr_c": psnr_c,
            **extra_info,
        }
    
    def pretrain_step(self, data, lr, step):
        self.model.train()
        meta_data = self.meta_data[self.cfg.train_split]
        data["bones_rest"] = namedtuple_map(
            lambda x: x.to(self.device), meta_data["bones_rest"]
        )
        idx = meta_data["meta_ids"].index(data["meta_id"])
        data["pose_latent"] = meta_data["pose_latent"][idx].to(self.device)
        smpl_params = data["pose_latent"].view(-1, 78)
        cond = {'smpl': smpl_params[:,3:72], 'smpl_params': smpl_params}
        extra_info = self.model.pos_enc.get_extra_losses(bones_rest = data["bones_rest"], rigid_clusters = data['rigid_clusters'], cond = cond, can_pc = None, geometry = self.model.geometry, normals = None, sdf_grads = None,sdf= None, pre = True)
        loss_bone_occ = (
            extra_info["loss_bone_occ"] if self.cfg.loss_bone_occ_multi > 0 else 0.0
        )
        loss_normals = (
                extra_info['loss_normals'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        
        eikonal_loss = (
                extra_info['eikonal_loss'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        
        sdf_consistency = (
                extra_info['loss_sdf_consistency'] if self.cfg.loss_coarse_mult > 0 else 0.0
            )
        
        loss_tv = (
            extra_info["loss_tv"] if self.cfg.loss_tv_multi > 0 else 0.0
        )
        self.optimizer.zero_grad()
        loss_all = (
            + loss_bone_occ * 0.1 #* self.cfg.loss_bone_occ_multi 
            + loss_normals * 10
            + eikonal_loss * 100
            + sdf_consistency *100
            + loss_tv * self.cfg.loss_tv_multi
        )
        loss_all.backward()
        self.optimizer.step()
        return {
            **extra_info,
        }

def plot_grad_flow(named_parameters, step):
    ave_grads = []
    layers = []
    norms = []
    for name, param in named_parameters:
        if(param.requires_grad) and ("bias" not in name):
            layers.append(name)
            if param.grad is not None:
                # ave_grads.append(param.grad.abs().mean().cpu().detach().numpy())
                norms.append(param.grad.norm().cpu().detach().numpy())
            else:
                # ave_grads.append(0.0)
                norms.append(0.0)
    
    plt.plot(norms, alpha=0.3, color="r")
    plt.hlines(0, 0, len(norms)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(norms), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(norms))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f'{step}_grad.png', bbox_inches = "tight")


def getattr_recursive(m, attr):
    for name in attr.split('.'):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []