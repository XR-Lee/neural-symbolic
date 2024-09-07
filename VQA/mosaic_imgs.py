from PIL import Image
import os

work_path1 = '/DATA_EDS2/wanggl/datasets/BEV_emun_polygon'
work_path2 = '/DATA_EDS2/wanggl/datasets/Opentest_mini_batch_area_enum2'
save_path = '/DATA_EDS2/wanggl/datasets/Mosaic_BEV_PV_downsampling'
# annotation_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/area_annotation_new.txt'

# with open(annotation_path, "r") as f:
#     data = f.read()
#     data = data.split('\n')

# scenes = set()
# imgs = {}
# for line in data:
#     line = line.split(' ')
#     scene = line[0]
#     if scene not in scenes:
#         imgs[scene] = []
#         scenes.add(scene)
#     imgs[scene].append(line[1])

# for scene in imgs:
#     scene_path1 = os.path.join(work_path1, scene)
#     scene_path2 = os.path.join(work_path2, scene)
#     scene_save = os.path.join(save_path, scene)

#     if not os.path.exists(scene_save):
#         os.makedirs(scene_save)
    
#     for img in imgs[scene]:
#         img_name1 = img + '.png'
#         img_name2 = img + '.jpg'
#         img_path1 = os.path.join(scene_path1, img_name1)
#         img_path2 = os.path.join(scene_path2, img_name2)
#         img_save = os.path.join(scene_save, img_name2)

#         image1 = Image.open(img_path1)
#         image2 = Image.open(img_path2)

#         width1, height1 = image1.size
#         width2, height2 = image2.size

#         new_height = max(height1, height2)

#         if height1 < height2:
#             ratio1 = new_height / height1
#             new_width1 = int(width1 * ratio1)
#             image1 = image1.resize((new_width1, new_height), Image.LANCZOS)
#         elif height2 < height1:
#             ratio2 = new_height / height2
#             new_width2 = int(width2 * ratio2)
#             image2 = image2.resize((new_width2, new_height), Image.LANCZOS)

#         new_width = width1 + width2
#         new_image = Image.new('RGB', (new_width, new_height), color='white')

#         new_image.paste(image1, (0, 0))

#         new_image.paste(image2, (image1.width, 0))

#         new_image.save(img_save)

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
