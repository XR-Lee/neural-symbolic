import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
from PIL import Image, ImageDraw
import logging
import math

def is_vertical_line(line):
    (x1, y1), (x2, y2) = line
    
    if x2 == x1:
        return True
    slope = (y2 - y1) / (x2 - x1)
    
    angle_with_x = math.degrees(math.atan(abs(slope)))
    
    angle_with_y = 90 - angle_with_x
    
    return angle_with_y < 45
    
    
def angle_with_y_axis(line):   
    (x1, y1), (x2, y2) = line
    
    
    if x1 == x2:
        if y2 > y1:
            return 0  
        else:
            return 180  
    else:
        if y2 > y1:
            slope = (y2 - y1) / (x2 - x1)
            angle_with_x_axis = math.degrees(math.atan(abs(slope)))
            angle_with_y_axis = 90 - angle_with_x_axis
        
        if y2 < y1:
            slope = (y2 - y1) / (x2 - x1)
            angle_with_x_axis = math.degrees(math.atan(abs(slope)))
            angle_with_y_axis = 180 - angle_with_x_axis 

        if y2 == y1:
            angle_with_y_axis = 90     
        
        return angle_with_y_axis > 40


def adjust_coordinates(points, target_length=10):
    if len(points) == target_length:
        return points
    elif len(points) < target_length:
        # ????????10,?????????
        x = np.linspace(0, len(points) - 1, num=len(points))
        xp = np.linspace(0, len(points) - 1, num=target_length)
        points_array = np.array(points)
        new_points_x = np.interp(xp, x, points_array[:, 0])  # ??x??
        new_points_y = np.interp(xp, x, points_array[:, 1])  # ??y??
        new_points_z = np.interp(xp, x, points_array[:, 2])  # ??z??
        new_points = list(zip(new_points_x, new_points_y, new_points_z))  # ??x, y, z??????
        return new_points
    else:
        # ????????10,?????10??
        indices = np.linspace(0, len(points) - 1, num=target_length, dtype=int)
        new_points = [points[i] for i in indices]
        return new_points



def interp_arc(points, t=15):

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
    
    
def find_optimal_area(area_points): # y value of BEV
    # ?????????area,????????
    default_large_x = 99999
    
    # ????????area???x?(????y??)
    min_x_val = default_large_x
    # ?????x?
    mid_x_val = default_large_x

    for points in area_points:
        # ??x???y??,???x,y???????y,x
        x_coords, y_coords = zip(*[(x, y) for x, y, z in points])

        # ????x??(????y??)???0
        if all(x > 0 for x in x_coords):
            # ???????????area(??????)
            width = max(y_coords) - min(y_coords)
            height = max(x_coords) - min(x_coords)
            if width > height:
                # ????x?(????y??)
                current_min_x = min(x_coords)
                current_max_x = max(x_coords)
                if current_min_x < min_x_val:
                    min_x_val = current_min_x
                    # ????x???x???
                    mid_x_val = 0.5 * (current_min_x + current_max_x)

    return min_x_val, mid_x_val
    
def preprocess_traffic_data(lane_lines, traffic_elements):
    """
    ??????????????????????
    
    ??:
        lane_lines: ????? (m, 10, 3) ???,?? m ????,??????? 10 ????,???? x, y, z ???
        traffic_elements: ????? (n, 2, 2) ???,?? n ?????,????????????????,????? x ? y ???
    
    ??:
        relationships_matrix: ?? m*n ???,???? NaN,????????????????????
        transformed_lane_lines: ????????????
    """
    # ?? m*n ? NaN ??
    m = lane_lines.shape[0]
    n = traffic_elements.shape[0]
    relationships_matrix = np.full((m, n), np.nan)
    
    # ???????
    transformed_lane_lines = np.array([[(-y, x, z) for (x, y, z) in line] for line in lane_lines], dtype=object)

    return relationships_matrix, transformed_lane_lines

def rule1(element_categories, relationships_matrix):  # unknown set to 0
    """
    ?????????(????????????),????????????
    ??????0,??????????0?
    
    ??:
        element_categories: ?????????,???????????????
        relationships_matrix: ????????????????
    
    ??:
        ?????????
    """
    for index, category in enumerate(element_categories):
        if category == 0:
            relationships_matrix[:, index] = 0  # ????????0

    return relationships_matrix
    
def rule2(transformed_lane_lines, relationships_matrix):  # opposite lane set to 0
    """
    ??Y?????????,??????????????
    
    ??:
        transformed_lane_lines: ?????????,???????10?????
        relationships_matrix: ????????????????
    
    ??:
        ?????????
    """
    for index, line in enumerate(transformed_lane_lines):
        # ??Y???????
        if all(line[i][1] > line[i+1][1] for i in range(len(line)-1)):
            relationships_matrix[index, :] = 0  # ????????0
    return relationships_matrix
    
def rule3(transformed_lane_lines, relationships_matrix):  # traversal lane set to 0
    """
    ?????????????????y???????????,???,???????????????????
    
    ??:
        transformed_lane_lines: ???????????,??? (n, 10, 3)?
        relationships_matrix: ????????????????
    
    ??:
        ?????????
    """
    for index, line in enumerate(transformed_lane_lines):
        # ????????????????
        front_pair = [(line[0][0], line[0][1]), (line[1][0], line[1][1])]
        back_pair = [(line[-2][0], line[-2][1]), (line[-1][0], line[-1][1])]
        
        # ???????????????
        if angle_with_y_axis(front_pair) or angle_with_y_axis(back_pair):
            relationships_matrix[index, :] = 0  # ????????0

    return relationships_matrix


def rule4(transformed_lane_lines, relationships_matrix, min_y, mid_y):  # area set to 0
    """
    ??????????y???0??????mid?????,
    ????????????????????
    
    ??:
        transformed_lane_lines: ?????????,???????10?????
        relationships_matrix: ????????????????
        min_y: ?????y?(???,?????????????)?
        mid_y: ?????y??
    
    ??:
        ?????????
    """
    for index, line in enumerate(transformed_lane_lines):
        last_y = line[-1][1]  # ???????????y??
        if last_y < 0 or last_y > mid_y:
            relationships_matrix[index, :] = 0  # ????????0

    return relationships_matrix
    

def draw_and_save_images_based_on_matrix(num_lanes, relationships_matrix, lane_ids, centerlines,leftlines,rightlines,lefttypes,righttypes,traffic_elements, imagename,save1,intrinsic, extrinsic,attribute,GT):
    """
    ????????????????,????
    ??????????????????,????'{???}-{???}.jpg'?

    ??:
        relationships_matrix: ????,m*n
        centerlines: ?????,???????????????
        traffic_elements: ??????,???????
        save_path: ???????
    """
    info_list = []
    lane_ids = lane_ids
    centerlines = centerlines
    leftlines = leftlines
    rightlines = rightlines
    k_points = traffic_elements
    imagename = imagename
    rootimg0 = save1   
    intrinsic = intrinsic
    extrinsic = extrinsic
    m, n = relationships_matrix.shape
    aaa = rootimg0.split('/')[-1]
    k_points2 = k_points.tolist()
    GT = GT
    for i in range(num_lanes):
        lane_id = lane_ids[i]
        points = _project(interp_arc(centerlines[i]), intrinsic, extrinsic)
        points_left = _project(interp_arc(leftlines[i]), intrinsic, extrinsic)
        points_right = _project(interp_arc(rightlines[i]), intrinsic, extrinsic)

        x_y = []
        if points is not None: 
            lenpo = len(points)   #### 
            for k in range(lenpo):
                x,y = points[k][0],points[k][1]
                x_y.append((x,y))   ###[(x,y),(x,y)......(x,y)]
        
        x_y_left = []       
        if points_left is not None:
            lenpoleft = len(points_left)
            for k in range(lenpoleft):
                x,y = points_left[k][0],points_left[k][1]
                x_y_left.append((x,y))
        
        x_y_right = []        
        if points_right is not None:
            lenporight = len(points_right)
            for k in range(lenporight):
                x,y = points_right[k][0],points_right[k][1]
                x_y_right.append((x,y))  
                
        image = Image.open(imagename)
        # image = image.convert('RGBA')
        image_width, image_height = image.size
        draw = ImageDraw.Draw(image)
        plot_polygon = False
        if points is not None and plot_polygon:        
            p2 =  points.tolist()      
            line_color = (0, 255, 0)  # ????? (R, G, B)
            line_color_lr = (255, 0, 0)
            line_width = 5  # ?????
            line_width_solid = 5
            line_width_dash = 3
            line_width_none = 7
            dash_pattern = (5, 5)
            outline_color = (0, 255, 0)  # ???? (R, G, B)
            outline_width = 2  # ????
            
            # if x_y:
            #     draw.line(x_y, fill=line_color, width=line_width)
                
            # if x_y_left and x_y_right:
            #     if lefttypes[i] == 1:
            #         draw.line(x_y_left, fill=line_color_lr, width=line_width_solid)
            #     elif lefttypes[i] == 2:
            #         draw.line(x_y_left, fill=line_color_lr, width=line_width_dash)
            #     elif lefttypes[i] == 0:
            #         draw.line(x_y_left, fill=line_color_lr, width=line_width_none)
  
                # if righttypes[i] == 1:
                #     draw.line(x_y_right, fill=line_color_lr, width=line_width_solid)
                # elif righttypes[i] == 2:
                #     draw.line(x_y_right, fill=line_color_lr, width=line_width_dash)
                # elif righttypes[i] == 0:
                #     draw.line(x_y_right, fill=line_color_lr, width=line_width_none)

                # front_side = [x_y_left[0], x_y_right[0]]
                # end_side = [x_y_left[-1], x_y_right[-1]]
                # draw.line(front_side, fill=line_color_lr, width=line_width_solid)
                # draw.line(end_side, fill=line_color_lr, width=line_width_solid)

            points = []
            for point in x_y_left:
                points.append(point)

            for point in reversed(x_y_right):
                points.append(point)

            draw.polygon(points, fill="green", outline='red')

        iname = f'{rootimg0}-{lane_ids[i]}.jpg'
        image.save(iname)
            
    return info_list
    



def visual_prompt_gen(jsonname,jsonname_new,save_path_root):####.....json
    print('----------')
    print(jsonname)
    # ?? JSON ??    ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/val/10000/info/xxx-ls.json'
    root,pathname,imgname = jsonname.split('/')[-3],jsonname.split('/')[-1],jsonname.split('val')[0]  ##???10000?json???  /DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/
    path = pathname.split('.')[0]   

    with open(jsonname) as f:
        data = json.load(f)

    with open(jsonname_new) as f:
        data_new = json.load(f)

    
    ###------------------------------------------------
    ###lane
    lane_ids = []
    centerlines = []      ### [0,10]
    leftlines = []
    rightlines = []
    lefttypes = []
    righttypes = []
    for i in range(len(data_new["predictions"]['lane_segment'])):
    # for i in range(len(data["annotation"]['lane_segment'])):
        # lane_id = data["annotation"]['lane_segment'][i]['id']
        # centerline = data["annotation"]['lane_segment'][i]['centerline']
        # leftline = data["annotation"]['lane_segment'][i]['left_laneline']
        # rightline = data["annotation"]['lane_segment'][i]['right_laneline']
        # lefttype = data["annotation"]['lane_segment'][i]['left_laneline_type']
        # righttype = data["annotation"]['lane_segment'][i]['right_laneline_type']
        lane_id = data_new["predictions"]['lane_segment'][i]['id']
        centerline = data_new["predictions"]['lane_segment'][i]['centerline']
        leftline = data_new["predictions"]['lane_segment'][i]['left_laneline']
        rightline = data_new["predictions"]['lane_segment'][i]['right_laneline']
        lane_ids.append(lane_id)
        centerlines.append(centerline)
        leftlines.append(leftline)
        rightlines.append(rightline)
        
    GT = data["annotation"]['topology_lste']      

    intrinsic = data["sensor"]['ring_front_center']['intrinsic']
    extrinsic = data["sensor"]['ring_front_center']['extrinsic']   
    
    centerlines2 = np.array(centerlines, dtype=object)

    if "traffic_element" in data["annotation"]:
        LT = data["annotation"]['traffic_element']    ###?????

    num = len(centerlines)  # ????
    num_lt = len(LT) if "traffic_element" in data["annotation"] else 0  # ??????
    
    image = data['sensor']['ring_front_center']['image_path']   ###??????? ???? 
    imagename = imgname + image ##????

    


    k_points = []
    attribute = []
    category = []

    for lt in LT:
        point = np.array(lt['points']).tolist()
        attr = np.array(lt['attribute'])
        cate = np.array(lt['category'])
        k_points.append(point)
        attribute.append(attr)
        category.append(cate)
        
    k_points1 = np.array(k_points)   ###???????
    
    
    area_points = []
    area_cates = []  
    
    for i in range(len(data["annotation"]['area'])):
        point = data["annotation"]['area'][i]['points']
        area_cate = data["annotation"]['area'][i]['category']
        area_points.append(point)
        area_cates.append(area_cate)
        
    num_area = len(data["annotation"]['area'])
    
    
    file0  = save_path_root + '/' + root
    
    # file0 = f'/DATA_EDS2/wanggl/datasets/Opentest_mini_batch_area_enum2/{root}'   ###### img  
    
    if not os.path.exists(file0): 
        os.makedirs(file0)
    

    # pathnn = f'{path}_{num}_{num_lt}'
    rootimg0 = file0 + '/' + path    ###/DATA_EDS2/wanggl/datasets/OpenLane-testlcteval2/10000_123456789_10_10
    
    
    #### llm start
   
    min_x,mid_x = find_optimal_area(area_points)    ###area min x
    mat,newlane = preprocess_traffic_data(centerlines2,k_points1)
    # mat1 = rule1(attribute,mat)
    # mat2 = rule2(newlane,mat1)
    # mat3 = rule3(newlane,mat2)
    # mat4 = rule4(newlane,mat3,min_x,mid_x)
    info_list = draw_and_save_images_based_on_matrix(len(centerlines),mat,lane_ids,centerlines,leftlines,rightlines,lefttypes,righttypes,k_points1,imagename,rootimg0,intrinsic, extrinsic,attribute,GT)  
     
    return info_list


def main():
    gt_path = '/fs/scratch/Sgh_CR_RIX/rix3_shared/dataset-public/OpenLane-V2/raw/val/'
    pred_path = '/home/iix5sgh/workspace/llm/pkl2json_mini_batch/'
    save_root_path = '/home/iix5sgh/workspace/llm/vqa_inter_pv_0914_img_only'
    
    for i in range(10000,10150):
        gt_path_info = gt_path + str(i).zfill(5) + '/' + 'info'   ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/train/10000/info'
        pred_path_info = pred_path + str(i).zfill(5) + '/' + 'info'
        for b in os.listdir(pred_path_info):
            gt_sub_frame = gt_path_info + '/' + b
            pred_sub_frame = pred_path_info + '/' + b
            namepart = gt_sub_frame.split("-")
            if namepart[-1] == "ls.json":
                info_list = visual_prompt_gen(gt_sub_frame,pred_sub_frame,save_root_path)
        print(f'The {i} epoch done')
    # with open('complete_infotest719.json', 'w') as f:
    #     json.dump(all_info, f, indent=4)
    print("Done!")


if __name__ == '__main__':
    main()