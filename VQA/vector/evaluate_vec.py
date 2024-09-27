import os

result_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/topll_dire_result2.txt'
annotation_path = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/lr_annotation_comp.txt'

with open(result_path, "r") as f:
    results = f.read()
    results = results.split('\n')

with open(annotation_path, "r") as f:
    annotation = f.read()
    annotation = annotation.split('\n')


def load_dict(content, col):
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

    return value

num_img = 0
label = load_dict(annotation, 5)
prediction = load_dict(results, 2)

num_p = 0
check = 0
tp = 0
tn = 0
fp = 0
fn = 0
for scene in prediction:
    if len(prediction[scene]) != 0:
        for img in prediction[scene]:
            if label.get(scene) != None:
                if label[scene].get(img) == "1":
                    num_p += 1
                if prediction[scene][img] == "1" and label[scene].get(img) == "1":
                    tp += 1
                    num_img += 1
                if prediction[scene][img] == "0" and label[scene].get(img) == "0":
                    tn += 1
                    num_img += 1
                if prediction[scene][img] == "1" and label[scene].get(img) == "0":
                    fp += 1
                    num_img += 1
                if prediction[scene][img] == "0" and label[scene].get(img) == "1":
                    fn += 1
                    num_img += 1
                if prediction[scene][img] == label[scene].get(img):
                    check += 1
                
                

print(f"number of img: {num_img}")
print(f"number of checks: {check}")
print(f"number of ps: {num_p}")
print(f"number of tps: {tp}")
print(f"number of tns: {tn}")
print(f"number of fps: {fp}")
print(f"number of fns: {fn}")
print(f"Accuracy: {check / num_img}")