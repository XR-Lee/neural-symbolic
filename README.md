<div align="center">
<h2>🐎Chameleon: Fast-slow Neuro-symbolic Lane Topology Extraction</h2>

[**Zongzheng Zhang**](https://vveicao.github.io/)<sup>1,2*</sup> · [**Xinrun Li**](https://github.com/netbeifeng)<sup>2*</sup> · [**Sizhe Zou**](https://1zb.github.io/)<sup>1</sup> · [**Guoxuan Chi**](https://1zb.github.io/)<sup>1</sup> · [**Siqi Li**](https://1zb.github.io/)<sup>1</sup> <br>
[**Xuchong Qiu**](https://niessnerlab.org/members/matthias_niessner/profile.html)<sup>2</sup> · [**Guoliang Wang**](https://tangjiapeng.github.io/)<sup>1</sup> · [**Guantian Zheng**](https://1zb.github.io/)<sup>1</sup> · [**Leichen Wang**](https://1zb.github.io/)<sup>2</sup> · [**Hang Zhao**](https://1zb.github.io/)<sup>3</sup> and [**Hao Zhao**](https://1zb.github.io/)<sup>1</sup>

<sup>1</sup>Institute for AI Industry Research (AIR), Tsinghua University · <sup>2</sup>Bosch Corporate Research <br>
<sup>3</sup>Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University <br>
<sub>(* indicates equal contribution)</sub>

**ICRA 2025**

<a href="https://arxiv.org/abs/2401.06614"><img src='https://img.shields.io/badge/arXiv-M2V-firebrick?logo=arxiv' alt='Arxiv'></a>
<a href="https://arxiv.org/pdf/2401.06614.pdf"><img src='https://img.shields.io/badge/PDF-M2V-orange?logo=googledocs&logoColor=white' alt='PDF'></a>
<a href='https://vveicao.github.io/projects/Motion2VecSets/'><img src='https://img.shields.io/badge/Project_Page-M2V-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href='https://www.youtube.com/watch?v=VXI3y2o0SqY&ab_channel=MatthiasNiessner'><img src='https://img.shields.io/badge/Video-M2V-red?logo=youtube' alt='Youtube Video'></a>
</div>

[\[Arxiv\]](https://arxiv.org/abs/2401.06614) [\[Paper\]](https://arxiv.org/pdf/2401.06614.pdf) [\[Project Page\]](https://vveicao.github.io/projects/Motion2VecSets/) [\[Video\]](https://www.youtube.com/watch?v=VXI3y2o0SqY&ab_channel=MatthiasNiessner) 


<p>
    Lane topology extraction involves detecting lanes and traffic elements and determining their relationships, a key perception task for mapless autonomous driving. This task requires complex reasoning, such as determining whether it is possible to turn left into a specific lane. To address this challenge, we introduce neuro-symbolic methods powered by visionlanguage foundation models (VLMs). Existing approaches have notable limitations: (1) Dense visual prompting with VLMs can achieve strong performance but is costly in terms of both financial resources and carbon footprint, making it impractical for robotics applications. (2) Neuro-symbolic reasoning methods for 3D scene understanding fail to integrate visual inputs when synthesizing programs, making them ineffective in handling complex corner cases. To this end, we propose a fast-slow neuro-symbolic lane topology extraction algorithm, named Chameleon, which alternates between a fast system that directly reasons over detected instances using synthesized programs and a slow system that utilizes a VLM with a chain-of-thought design to handle corner cases. Chameleon leverages the strengths of both approaches, providing an affordable solution while maintaining high performance. We evaluate the method on the OpenLane-V2 dataset, showing consistent improvements across various baseline detectors.
</p>


## Install

Install the environment following `scripts/init_environment.sh`, to install with cuda 11.0, use the command `bash scripts/init_environment.sh`

## Data preparation and Pretrained models

You can find the data preparation scripts in `/home/liang/m2v/dataset/dataset_generate` or you can directly download the [preprocessed dataset](https://nextcloud.in.tum.de/index.php/s/PQWBSJQaWyH6jxN).

For pretrained models, you can directly put them in the `ckpts` directory, and there are two subfolders `DFAUST` and `DT4D` for two datasets respectively, you can get them from the [Google Drive](https://drive.google.com/drive/folders/1dvn-u2BCPkmRWH9wsDdxLOqTV8SzPb7i?usp=sharing).

## Training and evaluation

You can run the training with the following command, here we use the DFAUST dataset as an example. For DT4D dataset you may just change the path of config file.

```bash
# Shape
python core/run.py --config_path ./configs/DFAUST/train/dfaust_shape_ae.yaml # Shape AE
python core/run.py --config_path ./configs/DFAUST/train/dfaust_shape_diff_sparse.yaml # Diffusion Sparse Input
python core/run.py --config_path ./configs/DFAUST/train/dfaust_shape_diff_partial.yaml # Diffusion Partial Input

# Deformation
python core/run.py --config_path ./configs/DFAUST/train/dfaust_deform_ae.yaml # Deformation AE
python core/run.py --config_path ./configs/DFAUST/train/dfaust_deform_diff_sparse.yaml # Diffusion Sparse Input
python core/run.py --config_path ./configs/DFAUST/train/dfaust_deform_diff_partial.yaml # Diffusion Partial Input
```

For evaluation, you can run the following command:
```bash
python core/eval.py --config_path ./configs/DFAUST/eval/dfaust_eval_sparse.yaml # Test Sparse Unssen Sequence
python core/eval.py --config_path ./configs/DFAUST/eval/dfaust_eval_sparse.yaml --test_ui # Test Sparse Unssen Individual

python core/eval.py --config_path ./configs/DFAUST/eval/dfaust_eval_partial.yaml # Test Partial Unssen Sequence
python core/eval.py --config_path ./configs/DFAUST/eval/dfaust_eval_partial.yaml --test_ui # Test Partial Unssen Individual
```

## Demo

You can run `demo/infer.py` to get the predicted mesh sequence for the inputs located in `demo/inputs`, before running the demo, you need to download the [pretrained models](https://drive.google.com/drive/folders/1dvn-u2BCPkmRWH9wsDdxLOqTV8SzPb7i?usp=sharing) and put them in the `ckpts` directory.

## Citation
```
@inproceedings{Wei2024M2V,
    title={Motion2VecSets: 4D Latent Vector Set Diffusion for Non-rigid Shape Reconstruction and Tracking},
    author={Wei Cao and Chang Luo and Biao Zhang and Matthias Nießner and Jiapeng Tang},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    url={https://vveicao.github.io/projects/Motion2VecSets/},
    year={2024}
} 
```

## Reference
This project is based on several wonderful projects:
- 3DShape2Vecset: https://github.com/1zb/3DShape2VecSet
- CaDeX: https://github.com/JiahuiLei/CaDeX
- LPDC: https://github.com/Gorilla-Lab-SCUT/LPDC-Net
- Occupancy Flow: https://github.com/autonomousvision/occupancy_flow
- DeformingThings4D: https://github.com/rabbityl/DeformingThings4D