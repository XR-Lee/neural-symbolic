<div align="center">
<h2>🦎Chameleon: Fast-slow Neuro-symbolic Lane Topology Extraction</h2>

**Zongzheng Zhang**<sup>1,2*</sup> · **Xinrun Li**<sup>2*</sup> · **Sizhe Zou**<sup>1</sup> · **Guoxuan Chi**<sup>1</sup> · **Siqi Li**<sup>1</sup> <br>
**Xuchong Qiu**<sup>2</sup> · **Guoliang Wang**<sup>1</sup> · **Guantian Zheng**<sup>1</sup> · **Leichen Wang**<sup>2</sup> · [**Hang Zhao**](https://hangzhaomit.github.io/)<sup>3</sup> and [**Hao Zhao**](https://sites.google.com/view/fromandto/)<sup>1</sup>

<sup>1</sup>Institute for AI Industry Research (AIR), Tsinghua University · <sup>2</sup>Bosch Corporate Research <br>
<sup>3</sup>Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University <br>
<sub>(* indicates equal contribution)</sub>

<!-- > **ICRA 2025** -->

</div>

<!-- >
[\[Arxiv\]](https://arxiv.org/abs/2401.06614) [\[Paper\]](https://arxiv.org/pdf/2401.06614.pdf) [\[Project Page\]](https://vveicao.github.io/projects/Motion2VecSets/) [\[Video\]](https://www.youtube.com/watch?v=VXI3y2o0SqY&ab_channel=MatthiasNiessner)
-->

![teaser](./assets/teaser.png)

<p>
    Lane topology extraction involves detecting lanes and traffic elements and determining their relationships, a key perception task for mapless autonomous driving. This task requires complex reasoning, such as determining whether it is possible to turn left into a specific lane. To address this challenge, we introduce neuro-symbolic methods powered by visionlanguage foundation models (VLMs).
</p>

![mainimage](./assets/main_pic.png)

<p>
 We propose a fast-slow neuro-symbolic lane topology extraction algorithm, named Chameleon, which alternates between a fast system that directly reasons over detected instances using synthesized programs and a slow system that utilizes a VLM with a chain-of-thought design to handle corner cases. Chameleon leverages the strengths of both approaches, providing an affordable solution while maintaining high performance. We evaluate the method on the OpenLane-V2 dataset, showing consistent improvements across various baseline detectors.
</p>

<!-- > https://github.com/OpenDriveLab/DriveLM/assets/103363891/67495435-4a32-4614-8d83-71b5c8b66443 -->

<!-- > above is old demo video. demo scene token: cc8c0bf57f984915a77078b10eb33198 -->

https://github.com/user-attachments/assets/dddde21a-5ce3-4324-8fef-e3eb531a40d0

<!-- > above is new demo video. demo scene token: cc8c0bf57f984915a77078b10eb33198 -->

## Detailed Docs

[Vector VQA task](./docs/VQA.md)
[Code Generation task](./docs/code_gen.md)

## 📌 TODO List
- [x] Project Page Setup 
- [x] Example Data
- [x] vec VQA task test & Evaluation
- [ ] other VQA tasks & Evaluation
- [ ] Environment Preparation
- [X] Code Generation Tutorial
- [ ] Full Test Tutorial
- [ ] ...


## Citation
If you find this project useful, feel free to cite our work!
<div style="display:flex;">
<div>

```bibtex
@article{zhang2025chameleon,
  title={Chameleon: Fast-slow Neuro-symbolic Lane Topology Extraction},
  author={Zhang, Zongzheng and Li, Xinrun and Zou, Sizhe and Chi, Guoxuan and Li, Siqi and Qiu, Xuchong and Wang, Guoliang and Zheng, Guantian and Wang, Leichen and Zhao, Hang and others},
  journal={arXiv preprint arXiv:2503.07485},
  year={2025}
}
```


