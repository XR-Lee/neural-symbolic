import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json

num_img = 0

def interp_arc(points, t=1000):

    # filter consecutive points with same coordinate
    temp = []
    for point in points:
        #point = point.tolist()
        if temp == [] or point != temp[-1]:
            temp.append(point)
    if len(temp) <= 1:
        return None
    points = np.array(temp)

    assert points.ndim == 2

    # the number of points on the curve itself
    n, _ = points.shape

    # equally spaced in arclength -- the number of points that will be uniformly interpolated
    eq_spaced_points = np.linspace(0, 1, t)

    # Compute the chordal arclength of each segment.
    # Compute differences between each x coord, to get the dx's
    # Do the same to get dy's. Then the hypotenuse length is computed as a norm.
    chordlen = np.linalg.norm(np.diff(points, axis=0), axis=1)  # type: ignore
    # Normalize the arclengths to a unit total
    chordlen = chordlen / np.sum(chordlen)
    # cumulative arclength

    cumarc = np.zeros(len(chordlen) + 1)
    cumarc[1:] = np.cumsum(chordlen)

    # which interval did each point fall in, in terms of eq_spaced_points? (bin index)
    tbins = np.digitize(eq_spaced_points, bins=cumarc).astype(int)  # type: ignore

    # #catch any problems at the ends
    tbins[np.where((tbins <= 0) | (eq_spaced_points <= 0))] = 1  # type: ignore
    tbins[np.where((tbins >= n) | (eq_spaced_points >= 1))] = n - 1

    s = np.divide((eq_spaced_points - cumarc[tbins - 1]), chordlen[tbins - 1])
    anchors = points[tbins - 1, :]
    # broadcast to scale each row of `points` by a different row of s
    offsets = (points[tbins, :] - points[tbins - 1, :]) * s.reshape(-1, 1)
    points_interp = anchors + offsets

    return points_interp


def _project(points, intrinsic, extrinsic):
    if points is None:
        return points
    points_in_cam_cor = np.linalg.pinv(np.array(extrinsic['rotation'])) \
        @ (points.T - np.array(extrinsic['translation']).reshape(3, -1))
    points_in_cam_cor = points_in_cam_cor[:, points_in_cam_cor[2, :] > 0]

    if points_in_cam_cor.shape[1] > 1:
        points_on_image_cor = np.array(intrinsic['K']) @ points_in_cam_cor
        points_on_image_cor = points_on_image_cor / (points_on_image_cor[-1, :].reshape(1, -1))
        points_on_image_cor = points_on_image_cor[:2, :].T
    else:
        points_on_image_cor = None
    return points_on_image_cor



def visual_prompt_gen(jsonname,jsonname_new,save_root_path ):####.....json
    print('----------')
    print(jsonname)
    # ?? JSON ??    ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/val/10000/info/xxx-ls.json'
    root,pathname,imgname = jsonname.split('/')[-3],jsonname.split('/')[-1],jsonname.split('val')[0]  ##???10000?json???  /DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/
    path = pathname.split('.')[0]   

    with open(jsonname) as f:
        data = json.load(f)
        
    with open(jsonname_new) as f:
        data_new = json.load(f)


    # ??????? 
    lane_ids = []
    centerlines = []      ### [0,10]
    leftlines = []
    rightlines = []
    for i in range(len(data_new["predictions"]['lane_segment'])):
        lane_id = data_new["predictions"]['lane_segment'][i]['id']
        centerline = data_new["predictions"]['lane_segment'][i]['centerline']
        leftline = data_new["predictions"]['lane_segment'][i]['left_laneline']
        rightline = data_new["predictions"]['lane_segment'][i]['right_laneline']
        lane_ids.append(lane_id)
        centerlines.append(centerline)
        leftlines.append(leftline)
        rightlines.append(rightline)
    


    intrinsic = data["sensor"]['ring_front_center']['intrinsic']
    extrinsic = data["sensor"]['ring_front_center']['extrinsic']   ##???? ?? 
    

    if "traffic_element" in data_new["predictions"]:
        LT = data_new["predictions"]['traffic_element']    ###?????

    num = len(centerlines)  # ????
        
    image = data['sensor']['ring_front_center']['image_path']   
    
    k_points = []
    attribute = []
    for i in range(len(data_new["predictions"]['traffic_element'])):
        point = data_new["predictions"]['traffic_element'][i]['points']
        attr = data_new["predictions"]['traffic_element'][i]['attribute']
        k_points.append(point)
        attribute.append(attr)
                
    k_points = np.array(k_points)   

    
    
    area_points = []
    area_cates = []  
    for i in range(len(data_new["predictions"]['area'])):
        point = data_new["predictions"]['area'][i]['points']
        area_cate = data_new["predictions"]['area'][i]['category']
        area_points.append(point)
        area_cates.append(area_cate)
        
    num_area = len(data_new["predictions"]['area'])
    colors = [(0.2, 0.2, 0.2),(1, 0, 0), (0, 0, 1), (0, 1, 0)]

    rootimg = save_root_path + '/' + root + '/' 
    if not os.path.exists(rootimg):
        os.makedirs(rootimg)
    
        
    lanetrue = []
    lefttrue = []
    righttrue = []
    
    print('num:',num)
    q = 0
    for i in range(num):     
        points = _project(interp_arc(centerlines[i]), intrinsic, extrinsic)
          
        cent_point = centerlines[i]
        left_point = leftlines[i]
        right_point = rightlines[i]
        
        lanetrue.append(cent_point)
        lefttrue.append(left_point)
        righttrue.append(right_point)  


    for index, (left, right) in enumerate(zip(lefttrue, righttrue)):
        ##BEV
        fig, ax = plt.subplots(figsize=(10, 20))
        ax.set_axis_off()  # ???????
        fig.patch.set_facecolor('white')
    
        for lanes in leftlines:  ##11??
            #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
            
            centerline_bevx = []
            centerline_bevy = []            
            
            for point in lanes:
                x, y, z = point
                new_x = 1 * x
                new_y = -1 * y
                centerline_bevx.append(new_y)   ##?????????  90?
                centerline_bevy.append(new_x)

            ax.plot(centerline_bevx, centerline_bevy, color=colors[0], linewidth=1.5, zorder=5)
    

        for lanes in rightlines:  ##11??
            #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
            
            centerline_bevx = []
            centerline_bevy = []            
            
            for point in lanes:
                x, y, z = point
                new_x = 1 * x
                new_y = -1 * y
                centerline_bevx.append(new_y)   ##?????????  90?
                centerline_bevy.append(new_x)

            ax.plot(centerline_bevx, centerline_bevy, color=colors[0], linewidth=1.5, zorder=5)

        points = []
        for point in left:
            x, y, z = point
            new_x = 1 * x
            new_y = -1 * y
            points.append((new_y, new_x))
        
        for point in reversed(right):
            x, y, z = point
            new_x = 1 * x
            new_y = -1 * y
            points.append((new_y, new_x))

        polygon = patches.Polygon(points, closed=True, fill=True, color='green')
        ax.add_patch(polygon)

        name = f'{save_root_path}/{root}/{path}-{lane_ids[index]}.png'   ###
        plt.savefig(name, bbox_inches='tight', pad_inches=0, facecolor='white')
        plt.close(fig)
        global num_img
        num_img += 1
        

def main():
    
    gt_path = '/fs/scratch/Sgh_CR_RIX/rix3_shared/dataset-public/OpenLane-V2/raw/val/'
    pred_path = '/home/iix5sgh/workspace/llm/pkl2json_mini_batch/'
    save_root_path = '/home/iix5sgh/workspace/llm/vqa_inter_0914_w'
    for i in range(10000,10150):
        gt_path_info = gt_path + str(i).zfill(5) + '/' + 'info'   ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/train/10000/info'
        pred_path_info = pred_path + str(i).zfill(5) + '/' + 'info'
        for frame in os.listdir(pred_path_info):      
            gt_path_frame = gt_path_info + '/' + frame 
            pred_path_frame = pred_path_info + '/' + frame   
            namepart = gt_path_frame.split("-")
            if namepart[-1] == "ls.json":
                visual_prompt_gen(gt_path_frame,pred_path_frame, save_root_path)      
        print(f'The scene: {i} done')
    print("All Scenes Done!")
    print(num_img)
    

if __name__ == '__main__':
    main()
