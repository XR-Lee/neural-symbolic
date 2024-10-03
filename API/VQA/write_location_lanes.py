import json
import os
import fnmatch



class Node():
    def __init__(self, val = None):
        self.val = val
        self.children = []

    def add_child(self,node):
        self.children.append(node)

def traverse_recursive(results,node):  
    if node is not None:  
        
        results.append(node.val) 

        for child in node.children:  

            traverse_recursive(results,child)  # 递归遍历子节点  



def build_tree(informations):
    nodes = {}
    # root=Node(root_id)
    # nodes[root.val] = root
    # parent=root
    for info in informations:
        id=info['lane_id']
        if id not in nodes:
            nodes[id] = Node(id)
        connections =info['connection'].split(',')
        if connections==['']:
            continue
        for connection in connections:
            if connection not in nodes:
                nodes[connection] = Node(connection)
            parent=Node(id)
            children=Node(connection)
            parent.add_child(children)
            nodes.pop(connection)

    return nodes

def write_location(json_pth,lanes,txt):
        with open(json_pth, 'r') as file:  
            data = json.load(file) 
            for lane in lanes:
                for information in data["information"]:
                    if information['lane_id']==lane:
                        information['location']=txt
                        
        with open(json_pth, 'w') as file:  

            json.dump(data, file, indent=4)  




def parse_file(filename, index):
    with open(filename, 'r') as file:
        lines =file.readlines()

        line = lines[index]
        parts = line.strip().split()
        scene_id = parts[0]

        part = parts[1].split('-ls-')
        timestamp = part[0]
        lane_id = part[1]
        intersection = int(parts[-1])
    return scene_id, timestamp, lane_id, intersection

def find_file(directory, scene_id, timestamp):
    filename = scene_id + '_' + timestamp + '.json'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if fnmatch.fnmatch(file, filename):
                return os.path.join(root, file)

if __name__ == '__main__':
    filename = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/mapless/ego_para_area_result_complete.txt'  
    directory = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/output_topomlp_complete'

    with open(filename, 'r') as file:
        lines =file.readlines()
    for index in range(len(lines)):
        data = parse_file(filename, index)
        scene_id = data[0]
        timestamp = data[1]
        lane_id = data[2]
        intersection = data[3]
        #print(scene_id,timestamp,lane_id,intersection)
        
        
        if intersection == 0:
            continue
        else:
            found_file = find_file(directory, scene_id, timestamp)
            print(found_file)
            
            with open(found_file, 'r') as file:
                json_data = json.load(file)
            tree=build_tree(json_data['information'])
            results=[]
            for lane in tree:
                traverse_recursive(results,tree[lane])

                for information in json_data["information"]:
                    if information['lane_id']==lane_id:
                        location=information['location']
                if location=='':
                    location='single'
                write_location(found_file,results,location)
                # pth
            output_file='./write_location_lanes_output.txt'
            # output_file=  './write_location_lanes_output/'+
            with open(output_file,'a') as file:
                for item in results:
                    file.write('/DATA_EDS2/wanggl/datasets/Mosaic_BEV_PV_downsampling/'+scene_id+'/'+timestamp +'-ls-'+  item+ '.jpg'+'\n')
            # print(results)


