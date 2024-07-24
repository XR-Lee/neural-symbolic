result_pkl_path = "/home/iix5sgh/workspace/hdmap/Online-HDMap-Perception/work_dirs/baseline_smerf_lanesegnet/test/results.pkl"
ann_file_path = "/fs/scratch/Sgh_CR_RIX/rix3_shared/dataset-public/OpenLane-V2/raw/data_dict_subset_A_val_lanesegnet.pkl"

import mmcv 
import numpy as np 
from shapely.geometry import LineString

def fix_pts_interpolate(lane, n_points):
    ls = LineString(lane)
    distances = np.linspace(0, ls.length, n_points)
    lane = np.array([ls.interpolate(distance).coords[0] for distance in distances])
    return lane

def load_annotations(ann_file):
    """Load annotation from a olv2 pkl file.

    Args:
        ann_file (str): Path of the annotation file.

    Returns:
        list[dict]: Annotation info from the json file.
    """
    data_infos = mmcv.load(ann_file, file_format='pkl')
    if isinstance(data_infos, dict):
        data_infos = list(data_infos.values())
    return data_infos


def format_results(results,data_infos,jsonfile_prefix=None):
    points_num =10 
    pred_dict = {}
    pred_dict['method'] = 'LaneSegNet'
    pred_dict['authors'] = []
    pred_dict['e-mail'] = 'dummy'
    pred_dict['institution / company'] = 'OpenDriveLab'
    pred_dict['country / region'] = 'CN'
    pred_dict['results'] = {}
    for idx, result in enumerate(results):
        info = data_infos[idx]
        key = ("val", info['segment_id'], str(info['timestamp']))

        pred_info = dict(
            lane_segment = [],
            area = [],
            traffic_element = [],
            topology_lsls = None,
            topology_lste = None
        )

        if result['lane_results'] is not None:
            lane_results = result['lane_results']
            scores = lane_results[1]
            valid_indices = np.argsort(-scores)
            lanes = lane_results[0][valid_indices]
            labels = lane_results[2][valid_indices]
            scores = scores[valid_indices]
            lanes = lanes.reshape(-1, lanes.shape[-1] // 3, 3)

            left_type_scores = lane_results[3][valid_indices]
            left_type_labels = lane_results[4][valid_indices]
            right_type_scores = lane_results[5][valid_indices]
            right_type_labels = lane_results[6][valid_indices]

            pred_area_index = []
            for pred_idx, (lane, score, label) in enumerate(zip(lanes, scores, labels)):
                if label == 0:
                    points = lane.astype(np.float32)
                    pred_lane_segment = {}
                    pred_lane_segment['id'] = 20000 + pred_idx
                    pred_lane_segment['centerline'] = fix_pts_interpolate(points[:points_num], 10)
                    pred_lane_segment['left_laneline'] = fix_pts_interpolate(points[points_num:points_num * 2], 10)
                    pred_lane_segment['right_laneline'] = fix_pts_interpolate(points[points_num * 2:], 10)
                    pred_lane_segment['left_laneline_type'] = left_type_labels[pred_idx]
                    pred_lane_segment['right_laneline_type'] = right_type_labels[pred_idx]
                    pred_lane_segment['confidence'] = score.item()
                    pred_info['lane_segment'].append(pred_lane_segment)

                elif label == 1:
                    points = lane.astype(np.float32)
                    pred_ped = {}
                    pred_ped['id'] = 20000 + pred_idx
                    pred_points = np.concatenate((fix_pts_interpolate(points[points_num:points_num * 2], 10),
                                                    fix_pts_interpolate(points[points_num * 2:][::-1], 10)), axis=0)
                    pred_ped['points'] = pred_points
                    pred_ped['category'] = label
                    pred_ped['confidence'] = score.item()
                    pred_info['area'].append(pred_ped)
                    pred_area_index.append(pred_idx)

                elif label == 2:
                    raise NotImplementedError

        if result['bbox_results'] is not None:
            te_results = result['bbox_results']
            scores = te_results[1]
            te_valid_indices = np.argsort(-scores)
            tes = te_results[0][te_valid_indices]
            scores = scores[te_valid_indices]
            class_idxs = te_results[2][te_valid_indices]
            for pred_idx, (te, score, class_idx) in enumerate(zip(tes, scores, class_idxs)):
                te_info = dict(
                    id = 20000 + pred_idx,
                    category = 1 if class_idx < 4 else 2,
                    attribute = class_idx,
                    points = te.reshape(2, 2).astype(np.float32),
                    confidence = score
                )
                pred_info['traffic_element'].append(te_info)

        if result['lsls_results'] is not None:
            topology_lsls_area = result['lsls_results'].astype(np.float32)[valid_indices][:, valid_indices]
            topology_lsls_area = np.delete(topology_lsls_area, pred_area_index, axis=0)
            topology_lsls = np.delete(topology_lsls_area, pred_area_index, axis=1)
            pred_info['topology_lsls'] = topology_lsls
        else:
            pred_info['topology_lsls'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['lane_segment'])), dtype=np.float32)

        if result['lste_results'] is not None:
            topology_lste_area = result['lste_results'].astype(np.float32)[valid_indices]
            topology_lste = np.delete(topology_lste_area, pred_area_index, axis=0)
            pred_info['topology_lste'] = topology_lste
        else:
            pred_info['topology_lste'] = np.zeros((len(pred_info['lane_segment']), len(pred_info['traffic_element'])), dtype=np.float32)

        pred_dict['results'][key] = dict(predictions=pred_info)

    return pred_dict



result_pkl = mmcv.load(result_pkl_path)
ann_file = load_annotations(ann_file_path)

res = format_results(results=result_pkl,data_infos=ann_file)
