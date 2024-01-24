# Experiments

Before start, please download and process the data follow the instructions [docs/dataset.md](dataset.md). The directory structure of the data is expected to be:
```
data/
├── zju/
    ├── SMPL_NEUTRAL.pkl
    ├── CoreView_313/
    ├── ...
    └── CoreView_386/
```


## Our Method: AV3D

    "AvatarOne: Monocular 3D Human Animation"

The default output directory is:
`./outputs/<exp-name>/<dataset>/<subject_id>/<fs>/`. You can check the on-the-fly qualitative evalution results in the folder `eval_imgs_otf` and quantitative scores in the `val_xxx_metrics_otf.txt` file. There is also a tensorboard log file in this folder for you to check on the loss curves.

Additionally every 10k steps there is canonical and deformed `.obj` files generated for visualization.

### Training.

```
bash train_implicit.sh
```
### Rendering

```
bash eval.sh
```
