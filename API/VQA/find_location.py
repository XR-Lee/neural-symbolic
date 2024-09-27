import json
import os

def write_location(scene_id,timestamp,lane_left,lane_right,lane_middle=None):
    json_pth= './output_topomlp_complete/'+scene_id+'_'+timestamp+'.json'

    with open(json_pth, 'r') as file:  

        data = json.load(file) 

        for information in data["information"]:
            if information["lane_id"]==lane_left :
                information["location"]="left"
            if information["lane_id"]==lane_right :
                information["location"]="right"
            if lane_middle!= None: 
                for i in lane_middle:
                    if  information["lane_id"]==i:
                        information["location"]="middle"

    with open(json_pth, 'w') as file:  

        json.dump(data, file, indent=4)  

def get_result_from_stack(left_stack,right_stack):
    lane_left= max(left_stack,key=left_stack.count)
    lane_right=max(right_stack,key=right_stack.count)
    lane_middle=[]
    left_stack.extend(right_stack)
    lane_middle = list(set(left_stack)) 
    lane_middle.remove(lane_left)
    lane_middle.remove(lane_right)
    return lane_left,lane_right,lane_middle


    # if 'lane_middle' in locals():
def add_location(lr_result_pth):
    with open(lr_result_pth, 'r', encoding='utf-8') as file:  
        left_stack=[]
        right_stack=[]

        for i,line in enumerate(file):
            
                  
            string=line.split(' ')

            scene_id=string[0]
            timestamp = string[1].split('-')[0]
            lane_0 = string[1].split('-')[2]
            lane_1 = string[1].split('-')[3]
            tag=int(string[2].replace("\n",""))
            if tag==1:
                read_lane_left=lane_0
                read_lane_right=lane_1
            elif tag==2:
                read_lane_right=lane_0
                read_lane_left=lane_1

            write_location(scene_id,timestamp,read_lane_left,read_lane_right)

            left_stack.append(read_lane_left)
            right_stack.append(read_lane_right)
            if i!=0:
                if scene_id==last_scene_id and timestamp==last_timestamp:
                     pass
                elif len(left_stack)>2:
                        left_stack.pop(-1)
                        right_stack.pop(-1)
                        
                        # lane_left= max(left_stack,key=left_stack.count)
                        # lane_right=max(right_stack,key=right_stack.count)
                        # lane_middle=[]
                        # left_stack.extend(right_stack)
                        # lane_middle = list(set(left_stack)) 
                        # lane_middle.remove(lane_left)
                        # lane_middle.remove(lane_right)
                        lane_left,lane_right,lane_middle=get_result_from_stack(left_stack,right_stack)
                        write_location(last_scene_id,last_timestamp,lane_left,lane_right,lane_middle)

                        left_stack=[read_lane_left]
                        right_stack=[read_lane_right]
                elif len(left_stack)==2:
                        left_stack.pop(0)
                        right_stack.pop(0)
                        lane_left=read_lane_left
                        lane_right=read_lane_right
    
            else:
                lane_left=read_lane_left
                lane_right=read_lane_right
                

            if i == len(line) - 1 and len(left_stack)>2 :  
                left_stack.pop(-1)
                right_stack.pop(-1)
                
                lane_left,lane_right,lane_middle=get_result_from_stack(left_stack,right_stack)
                write_location(last_scene_id,last_timestamp,lane_left,lane_right,lane_middle)



            last_scene_id=scene_id
            last_timestamp=timestamp

    return 



if __name__ == '__main__':
    add_location('mapless/ego_para_lr_result_complete.txt')