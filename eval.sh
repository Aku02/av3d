 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python launch.py --config-name=nerf_dyn_implicit dataset=zju pos_enc=fs dataset.bones_23=true\
 eval_every=10000 test_chunk=4096 dataset.num_rays=4096 \
  max_steps=25000 loss_bone_occ_multi=1.0 save_every=5000 \
  dataset.version=3 dataset.resize_factor=0.5  \
  pos_enc.opt.skinning_mode=mlp dataset.subject_id=313 resume=true resume_step=25000 eval_per_gpu=196 engine=evaluator