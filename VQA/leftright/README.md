# VQA tutorial

## Generate Visual Prompt

1. open neural-symbolic/VQA/leftright/pairwise_BEV.py

2. change paths inside the python file

    - dataset path:

    gt_path = '/fs/scratch/Sgh_CR_RIX/rix3_shared/dataset-public/OpenLane-V2/raw/val/'

    - prediction path:

    pred_path = '/home/iix5sgh/workspace/llm/pkl2json_mini_batch/'

    - visual prompt save path:

    save_root_path = '/home/iix5sgh/workspace/llm/vqa_lr_0909'

3. execute the .py

4. you will see imgs generated in your specifiied folder

## Perform VQA tasks

1. change the corresponding path in neural-symbolic/VQA/leftright/lr_vqa.py

2. For LallVa, you need to rewrite this file.

## Evaluate the task

1. use the neural-symbolic/VQA/leftright/evaluate_lr.py

2. change the GT path using neural-symbolic/VQA/lr_annotation_part1.txt