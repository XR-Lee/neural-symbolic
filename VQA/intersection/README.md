# VQA tutorial

## Generate Visual Prompt

1. open neural-symbolic/VQA/intersection/enum_BEV.py

2. change paths inside the python file

    - dataset path:

    gt_path = '/fs/scratch/Sgh_CR_RIX/rix3_shared/dataset-public/OpenLane-V2/raw/val/'

    - prediction path:

    pred_path = '/home/iix5sgh/workspace/llm/pkl2json_mini_batch/'

    - visual prompt save path:

    save_root_path = '/home/iix5sgh/workspace/llm/vqa_lr_0909'

3. execute the .py

4. you will see imgs generated in your specifiied folder

5. repeat the above operation on enum_PV 

    note that the PV visual prompt should be in different folder name
    
6. merge these two visual prompt into one by using merge_imgs.py 

    check the path inside the .py for correct merge

## Perform VQA tasks

1. change the corresponding path in neural-symbolic/VQA/intersection/lr_vqa.py

2. For LallVa, you need to rewrite this file.

## Evaluate the task

1. use the neural-symbolic/VQA/leftright/evaluate_lr.py

2. change the GT path using neural-symbolic/VQA/lr_annotation_part1.txt