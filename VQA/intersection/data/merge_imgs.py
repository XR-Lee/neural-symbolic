from PIL import Image
import os

work_path1 = '/home/iix5sgh/workspace/llm/vqa_inter_0914_w'
work_path2 =  '/home/iix5sgh/workspace/llm/vqa_inter_pv_0914'
save_path = '/home/iix5sgh/workspace/llm/vqa_inter_merged_0914_w'


for scene in os.listdir(work_path1):
    scene_path1 = os.path.join(work_path1, scene)
    scene_path2 = os.path.join(work_path2, scene)
    scene_save = os.path.join(save_path, scene)

    if not os.path.exists(scene_save):
        os.makedirs(scene_save)

    for img in os.listdir(scene_path1):
        img = img.split('.')[0]
        img_name1 = img + '.png'
        img_name2 = img + '.jpg'
        img_path1 = os.path.join(scene_path1, img_name1)
        img_path2 = os.path.join(scene_path2, img_name2)
        img_save = os.path.join(scene_save, img_name2)

        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)

        width1, height1 = image1.size
        width2, height2 = image2.size

        new_height = max(height1, height2)

        if height1 < height2:
            ratio1 = new_height / height1
            new_width1 = int(width1 * ratio1)
            image1 = image1.resize((new_width1, new_height), Image.LANCZOS)
        elif height2 < height1:
            ratio2 = new_height / height2
            new_width2 = int(width2 * ratio2)
            image2 = image2.resize((new_width2, new_height), Image.LANCZOS)

        new_width = width1 + width2
        new_image = Image.new('RGB', (new_width, new_height), color='white')

        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1.width, 0))

        new_image = new_image.resize((new_width // 2, new_height // 2), Image.LANCZOS)

        new_image.save(img_save)

print('Done')
