import mmcv
import pickle
import numpy as np
import copy
from shapely.geometry import LineString


def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

def format_openlanev2_gt(data_infos):
        gt_dict = {}
        for idx in range(len(data_infos)):
            info = copy.deepcopy(data_infos[idx])
            key = ('val', info['segment_id'], str(info['timestamp']))
            areas = []
            for area in info['annotation']['area']:
                if area['category'] == 1:
                    points = area['points']
                    left_boundary = fix_pts_interpolate(points[[0, 1]], 10)
                    right_boundary = fix_pts_interpolate(points[[2, 3]], 10)
                    area['points'] = np.concatenate([left_boundary, right_boundary], axis=0)
                    areas.append(area)
            info['annotation']['area'] = areas
            gt_dict[key] = info
        return gt_dict


ann_file = '/DATA_EDS2/zhangzz2401/zhangzz2401/OpenLane-V2-master/data/OpenLane-V2/data_dict_subset_A_val_lanesegnet.pkl'
data_infos = mmcv.load(ann_file, file_format='pkl')
if isinstance(data_infos, dict):
    data_infos = list(data_infos.values())
gt_dict = format_openlanev2_gt(data_infos)
pickle.dump(gt_dict, open('topll_gt.pkl', 'wb'))