HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python launch.py --config-name=nerf_dyn_implicit dataset=zju_backup pos_enc=fs dataset.bones_23=true\
 eval_every=1000 test_chunk=4096 dataset.num_rays=4096 \
  max_steps=10000 loss_bone_occ_multi=1.0 save_every=5000  \
  dataset.version=2 dataset.resize_factor=0.5 \
  pos_enc.opt.skinning_mode=mlp dataset.subject_id=313 \
  # Eval stage I results!
  #resume=true resume_step=10000 #eval_per_gpu=196 engine=evaluator

# The only change in the stages is the type of sampling used and the lr is adjusted for geometry as to promote learning texture better!
 HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python launch.py --config-name=nerf_dyn_implicit dataset=zju_backup pos_enc=fs dataset.bones_23=true\
 eval_every=1000 test_chunk=4096 dataset.num_rays=4096 \
  max_steps=15000 loss_bone_occ_multi=1.0 save_every=5000 \
  dataset.version=3 dataset.resize_factor=0.5  \
  pos_enc.opt.skinning_mode=mlp dataset.subject_id=313 \ 
  # Eval stage II results!
  #resume=true resume_step=120000 eval_per_gpu=196 engine=evaluator