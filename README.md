#  PIN: Positional Insert Unlocks Object Localisation Abilities in VLMs

[Project Page](https://quva-lab.github.io/PIN/)
[ArXiv](https://arxiv.org/abs/2402.08657)


This is the official code repository. For a paper summary, check out our [project page](https://quva-lab.github.io/PIN/)!

> **Note:**
> More functionalities, scripts and pretrained PINs are released in the coming weeks!

## Installation

Please create a conda environment for running PIN on OpenFlamingo, run

```
conda env create -f environment.yml
```
currently we are working on incorporating BLIP-2 into a single environment file, please stay tuned.

## Evaluation

We can evaluate the trained PIN on OpenFlamingo using for COCO, PVOC and LVIS. For that, please set up the corresponding datasets according to their repo or website. Alternatively, one can set the flag download to true to downlaod via our code for PVOC. The metrics and visualizations are logged using wandb. The test script can be started using 

```
sh scripts/test_OF_PIN.sh
```

after adding the disk path to each dataset.

## Training

First we need to set up the training data. For background images we use the [BG20k](https://github.com/JizhiziLi/animal-matting) dataset, please download from their repo and save on disk. Please copy the lvis category list from [here](https://www.lvisdataset.org/dataset) to the utils folder. Synthetic images are generated following [XPaste](https://github.com/yoctta/XPaste). We create a synthetic dataset with 100 samples, after cleaning around 60k objects remained. We will release our generated synthetic images soon and share a link here. 

After setting up the datasets, you can start a training run for PIN using 
```
sh scripts/run_OF_PIN.sh
```

Training metrics are logged using wandb. 

## Contact

If you have questions or find a bug, feel free to open a GitHub issue or send a mail to m.l.dorkenwald at uva.nl. 

## BibTeX

```
@InProceedings{Dorkenwald_PIN_CVPR_2024,
    author    = {Dorkenwald, Michael and Barazani, Nimrod and Snoek, Cees G. M. and Asano, Yuki M.},
    title     = {PIN: Positional Insert Unlocks Object Localisation Abilities in VLMs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {13548-13558}
}
```