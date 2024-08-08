import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from shapely.geometry import Polygon
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
    
def is_vertical_line1(line):
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
        
        return angle_with_y_axis < 40



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



def IMG(jsonname,jsonname_new):####.....json
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
    centerlines = []      ### [0,10]
    leftlines = []
    rightlines = []
    lefttypes = []
    righttypes = []
    for i in range(len(data_new["predictions"]['lane_segment'])):
        centerline = data_new["predictions"]['lane_segment'][i]['centerline']
        leftline = data_new["predictions"]['lane_segment'][i]['left_laneline']
        rightline = data_new["predictions"]['lane_segment'][i]['right_laneline']
        lefttype = data_new["predictions"]['lane_segment'][i]['left_laneline_type']
        righttype = data_new["predictions"]['lane_segment'][i]['right_laneline_type']
        centerlines.append(centerline)
        leftlines.append(leftline)
        rightlines.append(rightline)
        lefttypes.append(lefttype)
        righttypes.append(righttype)
    


    intrinsic = data["sensor"]['ring_front_center']['intrinsic']
    extrinsic = data["sensor"]['ring_front_center']['extrinsic']   ##???? ?? 
    

    if "traffic_element" in data_new["predictions"]:
        LT = data_new["predictions"]['traffic_element']    ###?????

    num = len(centerlines)  # ????
    num_lt = len(LT) if "traffic_element" in data_new["predictions"] else 0  # ??????
    
    image = data['sensor']['ring_front_center']['image_path']   
    imagename = imgname + image ##????

    
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
    colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
   
    file = '/DATA_EDS2/wanggl/datasets/BEV_polygon'
    rootimg = file + '/' + root + '/' 
    if not os.path.exists(rootimg):
        os.makedirs(rootimg)
    
    
    aaaaa = []
    bbbbb = []
    areakey = []
    diffkey = []
    AREA = False
    min_x = 99999
    max_x = 99999
    mid_x = 99999
    #print('num_area : ',num_area)   #####area???
    if num_area == 0:
        AREA = True
    elif num_area > 0:
        key = 0
        xval = 99999
        for l in range(num_area):
            if area_cates[l] == 1:  
                point_area = area_points[l] 
                point_areax = [x for x, y ,z in point_area] ##x???
                point_areay = [y for x, y ,z in point_area]
                max_x = max(point_areax)
                min_x = min(point_areax)
                max_y = max(point_areay)
                min_y = min(point_areay)
                difference = max_x - min_x
                difference_y = max_y - min_y
                if difference < difference_y:   ###???????????
                    if min_x < xval and min_x > 0:
                        xval = min_x
                        key = l        
                    
        area_val = area_points[key] 
        #print('area_val :',area_val)  
        y_values = [y for x, y ,z in area_val]
        x_values = [x for x, y ,z in area_val]
        min_y = min(y_values)  ###??0 ????
        min_x = min(x_values)
        max_x = max(x_values)
        if min_x <= 0:
            min_x = 9999
        if max_x <= 0:
            max_x = 9999
        mid_x = 0.5 * (min_x + max_x)
        if min_x - max_x > 1000:
            mid_x = max_x 
        #print('min_y : ' ,min_y)  ??area??????
        #print('min_x : ' ,min_x)
        
        
    lanetrue = []
    lefttrue = []
    righttrue = []
    lanetrue1 = []
    lefttrue1 = []
    righttrue1 = []
    print('num:',num)
    q = 0
    for i in range(num):     
        points = _project(interp_arc(centerlines[i]), intrinsic, extrinsic)
        x_y = []
        new_xy = []
        ori_xy = []
        ori_xy1 = []
        ori_xy2 = []
        sss = False
        
        cent_point = centerlines[i]
        left_point = leftlines[i]
        right_point = rightlines[i]
        
        cent_y0 = cent_point[0][1]
        cent_x0 = cent_point[0][0]
        cent_y3 = cent_point[-1][1]
        cent_x3 = cent_point[-1][0]
        mid_x1 = cent_point[2][0]
        mid_y1 = cent_point[2][1]
        mid_x2 = cent_point[-3][0]
        mid_y2 = cent_point[-3][1]        
        ori_xy.append((-1 * cent_y0,cent_x0))
        ori_xy.append((-1 * cent_y3,cent_x3))
        ori_xy1.append((-1 * cent_y0,cent_x0))
        ori_xy1.append((-1 * mid_y1,mid_x1))   ##1
        ori_xy2.append((-1 * mid_y2,mid_x2))
        ori_xy2.append((-1 * cent_y3,cent_x3)) ###2
        
        
        sss1 = angle_with_y_axis(ori_xy1)
        sss2 = angle_with_y_axis(ori_xy2)         ##???
        sssall = angle_with_y_axis(ori_xy) 
          
        if sss1 == True and sss2 == True and sssall == True:
            ssss_z = True
            if cent_x0 < min_x and cent_x3 < mid_x and cent_x3 > 0:
                AREA = True  
                lanetrue.append(cent_point)
                lefttrue.append(left_point)
                righttrue.append(right_point)  

    ##BEV
    fig, ax = plt.subplots(figsize=(10, 20))
    ax.set_axis_off()  # ???????
    fig.patch.set_facecolor('black')
    

    # for lanes in centerlines:  
    #     #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
        
    #     centerline_bevx = []
    #     centerline_bevy = []            
        
    #     for point in lanes:
    #         x, y, z = point
    #         new_x = 1 * x
    #         new_y = -1 * y
    #         centerline_bevx.append(new_y)   ##?????????  90?
    #         centerline_bevy.append(new_x)
        
                     
    #     ax.plot(centerline_bevx, centerline_bevy, color=colors[0], linewidth=1.5, zorder=5)
    #     ax.plot(centerline_bevx[0], centerline_bevy[0], 'o', color=(1, 0, 0), markersize=5, zorder=5)  
    #     ax.plot(centerline_bevx[-1], centerline_bevy[-1], 'o', color=(1, 0, 0), markersize=5, zorder=5) 
        
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
           

    for index, (left, right) in enumerate(zip(lefttrue, righttrue)):
        palette = [
            'red',
            'green',
            'blue',
            'yellow',
            'cyan',
            'magenta',
            'white',
            'aqua',
            'fuchsia',
            'gray',
            'lime',
            'maroon',
            'navy',
            'olive',
            'purple',
            'silver',
            'teal',
        ]
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

        polygon = patches.Polygon(points, closed=True, fill=True, color=palette[index])
        ax.add_patch(polygon)

    
    # for lanes in lanetrue:  ##11??
    #     #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
        
    #     centerline_bevx = []
    #     centerline_bevy = []            
        
    #     for point in lanes:
    #         x, y, z = point
    #         new_x = 1 * x
    #         new_y = -1 * y
    #         centerline_bevx.append(new_y)   ##?????????  90?
    #         centerline_bevy.append(new_x)
        
      
            
    #     ax.plot(centerline_bevx, centerline_bevy, color=colors[-1], linewidth=5, zorder=10)
    #     ax.plot(centerline_bevx[0], centerline_bevy[0], 'o', color=colors[-1], markersize=8, zorder=10)  
    #     ax.plot(centerline_bevx[-1], centerline_bevy[-1], 'o', color=colors[-1], markersize=8, zorder=10) 
    
    # for lanes in lefttrue:  ##11??
    #     #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
        
    #     centerline_bevx = []
    #     centerline_bevy = []            
        
    #     for point in lanes:
    #         x, y, z = point
    #         new_x = 1 * x
    #         new_y = -1 * y
    #         centerline_bevx.append(new_y)   ##?????????  90?
    #         centerline_bevy.append(new_x)

            
        
    #     ax.plot(centerline_bevx, centerline_bevy, color=colors[1], linewidth=2.5, zorder=10)
    #     ax.plot(centerline_bevx[0], centerline_bevy[0], 'o', color=colors[1], markersize=8, zorder=10)  
    #     ax.plot(centerline_bevx[-1], centerline_bevy[-1], 'o', color=colors[1], markersize=8, zorder=10)        

    # for lanes in righttrue:  ##11??
    #     #lanes = lanes[int(len(centerline) * 0.02):int(len(centerline) * 0.98)]
        
    #     centerline_bevx = []
    #     centerline_bevy = []            
        
    #     for point in lanes:
    #         x, y, z = point
    #         new_x = 1 * x
    #         new_y = -1 * y
    #         centerline_bevx.append(new_y)   ##?????????  90?
    #         centerline_bevy.append(new_x)

  
    #     ax.plot(centerline_bevx, centerline_bevy, color=colors[1], linewidth=2.5, zorder=10)
    #     ax.plot(centerline_bevx[0], centerline_bevy[0], 'o', color=colors[1], markersize=8, zorder=10)  
    #     ax.plot(centerline_bevx[-1], centerline_bevy[-1], 'o', color=colors[1], markersize=8, zorder=10)            

        
             
    filename = f'{file}/{root}/{path}.png'   ###
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close(fig)
        
        

def main():
    
    rootpath = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/val/'
    rootpath_new = '/DATA_EDS2/wanggl/datasets/pkl2json72te/'

    for i in range(10000,10150):
        rootpath2 = rootpath + str(i).zfill(5) + '/' + 'info'   ##'/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/train/10000/info'
        rootpath_new2 = rootpath_new + str(i).zfill(5) + '/' + 'info'
        for b in os.listdir(rootpath_new2):      
            rootpath3 = rootpath2 + '/' + b 
            rootpath_new3 = rootpath_new2 + '/' + b   
            namepart = rootpath3.split("-")
            if namepart[-1] == "ls.json":
                IMG(rootpath3,rootpath_new3)      
        print(f'The {i} epoch done')
    print("Done!")
    

if __name__ == '__main__':
    main()