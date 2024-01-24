from dataclasses import dataclass, field
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import IFPipeline
from diffusers.utils import pt_to_pil
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *
from threestudio.utils.perceptual import PerceptualLoss
from tqdm import trange, tqdm
# from threestudio.models.guidance.unet_2d_condition_hidden import unet_forward_hidden

@threestudio.register("deep-floyd-guidance")
class DeepFloydGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "DeepFloyd/IF-I-XL-v1.0"
        # FIXME: xformers error
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = True
        guidance_scale: float = 20.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        weighting_strategy: str = "sds"

        view_dependent_prompting: bool = True

        #(YK): DU guidance
        use_du: bool = False
        per_du_step: int = 10
        start_du_step: int = 1000
        cache_du: bool = False
        du_diffusion_steps: int = 20

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Deep Floyd ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # Create model
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                threestudio.warn(
                    f"Use DeepFloyd with xformers may raise error, see https://github.com/deep-floyd/IF/issues/52 to track this problem."
                )
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        if self.cfg.use_du:
            if self.cfg.cache_du:
                self.edit_frames = {}
            self.perceptual_loss = PerceptualLoss().eval().to(self.device)

        threestudio.info(f"Loaded Deep Floyd!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    def compute_grad_du(
        self,
        latents: Float[Tensor, "B 3 64 64"],
        rgb_BCHW_512: Float[Tensor, "B 3 512 512"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        use_perp_neg=False,
        neg_guidance_weights=None,
        **kwargs,
    ):
        batch_size, _, _, _ = latents.shape

        # need_diffusion = (
        #     self.global_step % self.cfg.per_du_step == 0
        #     and self.global_step > self.cfg.start_du_step
        # )
        need_diffusion = True

        if self.cfg.cache_du:
            if torch.is_tensor(kwargs["index"]):
                batch_index = kwargs["index"].item()
            else:
                batch_index = kwargs["index"]
            if (
                not (batch_index in self.edit_frames)
            ) and self.global_step > self.cfg.start_du_step:
                need_diffusion = True
        
        need_loss = self.cfg.cache_du or need_diffusion
        guidance_out = {}

        if need_diffusion:
            # sample number of steps to diffuse
            t = torch.randint(
                self.min_step,
                self.max_step,
                [1],
                dtype=torch.long,
                device=self.device,
            )
            self.scheduler.config.num_train_timesteps = t.item()
            self.scheduler.set_timesteps(self.cfg.du_diffusion_steps)

            # batched
            t = t.repeat(batch_size)
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t[0])  # type: ignore
                for i, timestep in tqdm(enumerate(self.scheduler.timesteps), total=self.cfg.du_diffusion_steps, desc="du guidance"):
                    # predict the noise residual with unet, NO grad!
                    noise_pred = self.get_noise_pred(latents, timestep.repeat(batch_size), text_embeddings, use_perp_neg, neg_guidance_weights)
                    # get previous sample, continue loop
                    latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            
            #(YK): can optionally add superresolution here
            gt_rgb = F.interpolate(
                latents, (512, 512), mode="bilinear"
            ).permute(0, 2, 3, 1)

            # debug (set t = self.max_step)
            # import cv2
            # import numpy as np
            # for i in range(batch_size):
            #     temp = ((gt_rgb.detach().cpu()[i].numpy() + 1) * 0.5 * 255).astype(np.uint8); cv2.imwrite(f".threestudio_cache/test{i}.jpg", temp[:, :, ::-1])
    
            w = 1 - self.alphas[t]
            guidance_out.update(
                {
                    "loss_du_l1": (w * torch.nn.functional.l1_loss(rgb_BCHW_512, gt_rgb.permute(0, 3, 1, 2), reduction="none").reshape(batch_size, -1).mean(-1)).mean(),
                    "loss_du_p": (w * self.perceptual_loss(rgb_BCHW_512.contiguous(),gt_rgb.permute(0, 3, 1, 2).contiguous()).reshape(batch_size, -1).mean(-1)).mean(),
                    "edit_images": gt_rgb,#.detach().cpu(),
                }
            )

            if self.cfg.cache_du:
                self.edit_frames[batch_index] = edit_images.detach().cpu()

        return guidance_out

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        assert rgb_as_latents == False, f"No latent space in {self.__class__.__name__}"
        rgb_BCHW = rgb_BCHW * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        latents = F.interpolate(
            rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
        )

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            with torch.no_grad():
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 4),
                    encoder_hidden_states=text_embeddings,
                )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)  # TODO: use torch generator
                latents_noisy = self.scheduler.add_noise(latents, noise, t)
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        """
        # thresholding, experimental
        if self.cfg.thresholding:
            assert batch_size == 1
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)
            noise_pred = custom_ddpm_step(self.scheduler,
                noise_pred, int(t.item()), latents_noisy, **self.pipe.prepare_extra_step_kwargs(None, 0.0)
            )
        """

        if self.cfg.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.cfg.weighting_strategy == "uniform":
            w = 1
        elif self.cfg.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sds": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_utils = {
                "use_perp_neg": prompt_utils.use_perp_neg,
                "neg_guidance_weights": neg_guidance_weights,
                "text_embeddings": text_embeddings,
                "t_orig": t,
                "latents_noisy": latents_noisy,
                "noise_pred": torch.cat([noise_pred, predicted_variance], dim=1),
            }
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        rgb_BCHW_512 = F.interpolate(rgb_BCHW, (512, 512), mode="bilinear", align_corners=False)
        if self.cfg.use_du:
            grad = self.compute_grad_du(
                latents, rgb_BCHW_512, text_embeddings, prompt_utils.use_perp_neg, neg_guidance_weights, **kwargs
            )
            guidance_out.update(grad)

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]
        if use_perp_neg:
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(batch_size)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 6, 64, 64)

            noise_pred_text, _ = noise_pred[:batch_size].split(3, dim=1)
            noise_pred_uncond, _ = noise_pred[batch_size : batch_size * 2].split(
                3, dim=1
            )
            noise_pred_neg, _ = noise_pred[batch_size * 2 :].split(3, dim=1)

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(batch_size)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (2B, 6, 64, 64)

            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return torch.cat([noise_pred, predicted_variance], dim=1)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(50)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = latents_noisy.shape[0]  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand(
            [bs, -1]
        ) > t_orig.unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = latents_noisy.permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(len(t)):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1]
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = latents_1step.permute(0, 2, 3, 1)
        imgs_1orig = pred_1orig.permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = latents_final.permute(0, 2, 3, 1)

        return {
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )
        
        self.global_step = global_step

    @torch.no_grad()
    def sample_latents(
        self,
        prompt_utils: PromptProcessorOutput,
        diffusion_latent=False,
        hidden_diffusion=True,
        num_inference_steps=1, 
        guidance_scale=0, 
        seed: int = -1,
        cross_attention_kwargs=None, 
        eta: float = 0.0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        """
        
        Returns:
            _type_: _description_
        """
        if diffusion_latent:
            generator = torch.Generator(device=self.device).manual_seed(seed) if seed > -1 else None
            do_classifier_free_guidance = guidance_scale > 1.0
            
            # 3. Encode input prompt
            text_embeddings = prompt_utils.text_embeddings
            
            # 4. Prepare timesteps 
            self.scheduler.set_timesteps(50, device=self.device)
            timesteps = self.scheduler.timesteps
            
            # 5. Prepare intermediate images 
            intermediate_images = self.pipe.prepare_intermediate_images(
               text_embeddings.shape[0],
               self.unet.config.in_channels, 
                self.unet.config.sample_size, 
                self.unet.config.sample_size, 
                text_embeddings.dtype,
                self.device, 
                generator    
            )
            
            # 6. Prepare extra step kwargs.
            extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)


            for i, t in enumerate(timesteps[:num_inference_steps-1]):
                model_input = (
                    torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
                )
                model_input = self.scheduler.scale_model_input(model_input, t)

                # TODO: check this cross_attention_kwargs
                # predict the noise residual
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                    noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

                if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                    noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)

                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]

                # DEBUG
                save_dir = os.path.dirname('out/debug/IF-denoise-step')
                os.makedirs(save_dir, exist_ok=True)
                pt_to_pil(intermediate_images)[0].save(os.path.join(save_dir, f'image-step{i}.jpg'))
        
            # TODO: or should this be the last time step???
            # TODO: evaluation mode of scheudler???
            # For the last step, get the output from the hidden feature map...
            t = timesteps[num_inference_steps-1]
            model_input = (
                torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
            )
            model_input = self.scheduler.scale_model_input(model_input, t)

            if not hidden_diffusion:
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
            else:
                noise_pred = unet_forward_hidden(self.unet, 
                                                 model_input, 
                                                 t,
                                                 encoder_hidden_states=text_embeddings,
                                                 cross_attention_kwargs=cross_attention_kwargs,
                                                 return_dict=False,
                                                 )
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
                noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

            if not hidden_diffusion:
                # compute the previous noisy sample x_t -> x_t-1
                intermediate_images = self.scheduler.step(
                    noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
                )[0]
            else:
                intermediate_images = noise_pred
            return intermediate_images
        else:
            return prompt_utils.text_embeddings

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def check_prompt(
        self,
        text_embeddings,
        neg_embeddings, 
        save_path='./debug/deep_floyd/guidance/'
    ):
        generator = torch.manual_seed(0)
        image = self.pipe(prompt_embeds=text_embeddings, negative_prompt_embeds=neg_embeddings, generator=generator, output_type="pt").images
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        pt_to_pil(image)[0].save(save_path)
        
        # t = torch.ones(self.batch_size, dtype=torch.long, device=self.device) * self.max_step
        # latents = torch.randn(self.batch_size, 4, 64, 64)
        # # Text embeds -> img latents
        # latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents,
        #             num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # # Img latents -> imgs
        # imgs = self.decode_latents(latents.to(
        # text_embeds.dtype))  # [1, 3, 512, 512]

        # # Img to Numpy
        # if to_numpy:
        # imgs = to_np_img(imgs)

"""
# used by thresholding, experimental
def custom_ddpm_step(ddpm, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, generator=None, return_dict: bool = True):
    self = ddpm
    t = timestep

    prev_t = self.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t].item()
    alpha_prod_t_prev = self.alphas_cumprod[prev_t].item() if prev_t >= 0 else 1.0
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    noise_thresholded = (sample - (alpha_prod_t ** 0.5) * pred_original_sample) / (beta_prod_t ** 0.5)
    return noise_thresholded
"""

# write a test main...