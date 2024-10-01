<div align="center">
<h2>🦎Chameleon: Fast-slow Neuro-symbolic Lane Topology Extraction</h2>

[**Zongzheng Zhang**](https://vveicao.github.io/)<sup>1,2*</sup> · [**Xinrun Li**](https://github.com/netbeifeng)<sup>2*</sup> · [**Sizhe Zou**](https://1zb.github.io/)<sup>1</sup> · [**Guoxuan Chi**](https://1zb.github.io/)<sup>1</sup> · [**Siqi Li**](https://1zb.github.io/)<sup>1</sup> <br>
[**Xuchong Qiu**](https://niessnerlab.org/members/matthias_niessner/profile.html)<sup>2</sup> · [**Guoliang Wang**](https://tangjiapeng.github.io/)<sup>1</sup> · [**Guantian Zheng**](https://1zb.github.io/)<sup>1</sup> · [**Leichen Wang**](https://1zb.github.io/)<sup>2</sup> · [**Hang Zhao**](https://1zb.github.io/)<sup>3</sup> and [**Hao Zhao**](https://1zb.github.io/)<sup>1</sup>

<sup>1</sup>Institute for AI Industry Research (AIR), Tsinghua University · <sup>2</sup>Bosch Corporate Research <br>
<sup>3</sup>Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University <br>
<sub>(* indicates equal contribution)</sub>

<!-- > **ICRA 2025** -->

<a href='https://github.com/user-attachments/assets/dddde21a-5ce3-4324-8fef-e3eb531a40d0'><img src='https://img.shields.io/badge/Video-M2V-red?logo=youtube' alt='Youtube Video'></a>

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


## Data preparation

Before the data generation, you can directly get the lane segment and traffic element perception results of TopoMLP from the [Google Drive](https://drive.google.com/file/d/10FUIrxqSPai6eQlqlgIkmBjvtBAyCmJT/view?usp=drive_link). You can download the pickle file and save it in `/dataset`.

You can run the following command to convert the pickle file into timestamp-wise json files:

```python
# Convert pkl to json
python tools/pkl2json.py --input $PKL_PATH --output $OUTPUT_PATH --verbose

# For example, you can try this
python tools/pkl2json.py --input ./dataset/results_base.pkl --output ./dataset/output_json --verbose
```

Then you can generate the corresponding visual prompt data for different VQA sub-tasks. And you can find data generation scripts in `/VQA/sub-task/data`. For intersection task, you should generate BEV images and PV images respectively, and then make mosaics of them. Here, we take the connection VQA task for instance:

```python
# Generate BEV images for the connection VQA task
python ./VQA/connection/data/pairwise_conn_BEV.py --output $GENERATION_PATH --verbose
```

On top of that, you should make your text prompts for the VQA task and save it as a `txt` file. For instance:

```python
system_context = '''
The red lines in the photos are lane boundaries. Two segments in different lanes don't have any connection relationship. Only two segments in the same lane end to end adjacent are considered as directly connected.
'''
prompt = '''
You are an expert in determining adjacent lane segments in the image. Let's determine if the the green segment is directly connected with the blue segmemt. Please reply in a brief sentence starting with "Yes" or "No".
'''
```

So far, you get done all preparations for an VQA task.

## Testing and evaluation

You can run the VQA task with the following command, here we use the connection VQA task as an example. 

```python
# Run the connection VQA task
python ./VQA/connection/conn_VQA.py --txt $TEXT_PROMPT_PATH --visual $VISUAL_PROMPT_PATH --output $OUTPUT_RESULT_PATH --key $OPENAI_API_KEY --verbose
```

For evaluation, you can run the following command:
```python
# Evaluate your prediction
python ./VQA/connection/evaluate_conn.py --path $RESULT_PATH
```