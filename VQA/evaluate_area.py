import os

result_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/area_result_filter.txt'
annotation_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/area_annotation_reduced.txt'

with open(result_path, "r") as f:
    results = f.read()
    results = results.split('\n')

with open(annotation_path, "r") as f:
    annotation = f.read()
    annotation = annotation.split('\n')


def load_dict(content, col):
    global num_img
    num_img = 0
    scenes = set()
    value = {}
    for line in content:
        line = line.split(' ')
        scene = line[0]
        if scene not in scenes:
            value[scene] = {}
        scenes.add(scene)
        if len(line) > 1:
            value[scene][line[1]] = line[col]
            num_img += 1

    return value

num_img = 0
label = load_dict(annotation, 2)
prediction = load_dict(results, 2)

num_0 = 0
num_1 = 0
tp = 0
for scene in prediction:
    if len(prediction[scene]) != 0:
        for img in prediction[scene]:
            if label[scene].get(img) == '0':
                num_0 += 1
            if label[scene].get(img) == '1':
                num_1 += 1
            if prediction[scene][img] == label[scene].get(img):
                tp += 1

print(f"number of img: {num_img}")
print(f"number of 0s: {num_0}")
print(f"number of 1s: {num_1}")
print(f"number of tps: {tp}")
print(f"Accuracy: {tp / num_img}")