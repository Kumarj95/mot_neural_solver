import os.path as osp
import os
import cv2
import pandas as pd
import shutil
import numpy as np

DET_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')
DET_COL_NAMES_CUSTOM=('frame','class','conf','bb_left','bb_top','bb_width','bb_height')
GT_COL_NAMES = ('frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf')


def _build_seq_info_dict_custom(seq_name, data_root_path, dataset_params):

    seq_path = dataset_params['seq_path']
    imgs_path = dataset_params['img_path']
    seq_len = len(set(os.listdir(imgs_path)))
    frame_height, frame_width= dataset_params['frame_height'], dataset_params['frame_width']

    seq_info_dict = {'seq': seq_name,
                     'seq_path': seq_path,
                     'det_file_name': dataset_params['detection_file_path'],
                     'frame_height': frame_height,
                     'frame_width': frame_width,
                     'seq_len': seq_len,
                     'fps': dataset_params['fps'],
                     'mov_camera': False,
                     'has_gt': dataset_params['hasgt']
                    }
    return seq_info_dict


def get_custom_det_df(seq_name, data_root_path, dataset_params):

    seq_path = dataset_params['seq_path']
    detections_file_path =dataset_params['detection_file_path']
    det_df = pd.read_csv(detections_file_path, header=None, sep=" ")
    x1=det_df[3]
    x2=det_df[5]
    y1=det_df[4]
    y2=det_df[6]
    bb_width= x2-x1
    bb_height=y2-y1
    det_df[5]=bb_width
    det_df[6]=bb_height
    # print(det_df)
    # frames=det_df[0]
    # ids=-1*np.ones(len(det_df))
    # det_df[1]=ids
    # print(frames)
    # input()
    # Number and order of columns is always assumed to be the same
    det_df = det_df[det_df.columns[:len(DET_COL_NAMES_CUSTOM)]].copy()
    det_df.columns = DET_COL_NAMES_CUSTOM
    det_df['bb_left'] -= 1 # Coordinates are 1 based
    det_df['bb_top'] -= 1

    # If id already contains an assignment (e.g. using tracktor output), keep it
    # if len(det_df['id'].unique()) > 1:
    #     det_df['tracktor_id'] = det_df['id']

    # Add each frame's path
    frame_num=0
    add_frame_path = lambda frame_num: osp.join(dataset_params['img_path'], f'frame{int(frame_num):06}.jpg')
    det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

    seq_info_dict = _build_seq_info_dict_custom(seq_name, data_root_path, dataset_params)
    seq_info_dict['is_gt'] = False

    if seq_info_dict['has_gt']: # Return the corresponding ground truth, if available
        gt_file_path = dataset_params['gt_file_path']
        gt_df = pd.read_csv(gt_file_path, header=None)
        gt_df = gt_df[gt_df.columns[:len(GT_COL_NAMES)]]
        gt_df.columns = GT_COL_NAMES
        gt_df['bb_left'] -= 1  # Coordinates are 1 based
        gt_df['bb_top'] -= 1
        gt_df = gt_df[gt_df['conf'] == 1].copy()
        gt_df['bb_bot'] = (gt_df['bb_top'] + gt_df['bb_height']).values
        gt_df['bb_right'] = (gt_df['bb_left'] + gt_df['bb_width']).values

        # Store the gt file in the common evaluation path
        # gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
        # os.makedirs(gt_to_eval_path, exist_ok=True)
        # shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

    else:
        gt_df = None
    return det_df, seq_info_dict, gt_df


# def get_custom_gt_df(seq_name, data_root_path, dataset_params):

#     # Create a dir to store Ground truth data in case if does not exist yet
#     seq_path = dataset_params['seq_path']
#     # if not osp.exists(seq_path):
#     #     os.mkdir(seq_path)
#     #     non_gt_seq_path = osp.join(data_root_path, seq_name[:-3])
#     #     shutil.copytree(osp.join(non_gt_seq_path, 'gt'), osp.join(seq_path, 'gt'))

#     detections_file_path = dataset_params['gt_file_path']
#     det_df = pd.read_csv(detections_file_path, header=None)

#     # Number and order of columns is always assumed to be the same
#     det_df = det_df[det_df.columns[:len(GT_COL_NAMES)]]
#     det_df.columns = GT_COL_NAMES
#     det_df['bb_left'] -= 1 # Coordinates are 1 based
#     det_df['bb_top'] -= 1

#     # VERY IMPORTANT: Only take active annotations (see: https://arxiv.org/abs/1504.01942, page 7)
#     det_df = det_df[det_df['conf'] == 1].copy()

#     det_df['bb_bot'] = (det_df['bb_top'] + det_df['bb_height']).values
#     det_df['bb_right'] = (det_df['bb_left'] + det_df['bb_width']).values
#     det_df['bb_size'] = det_df['bb_height']*det_df['bb_width']

#     # det_df = drop_occluded_gt_annotations(det_df, dataset_params)

#     # Add each image's path
#     add_frame_path = lambda frame_num: osp.join(data_root_path, seq_name[:-3], f'img1/{frame_num:06}.jpg')
#     det_df['frame_path'] = det_df['frame'].apply(add_frame_path)

#     seq_info_dict = _build_seq_info_dict_custom(seq_name[:-3], data_root_path, dataset_params)

#     # Correct the detections file name to contain the 'gt' as well as other attributes
#     seq_info_dict['det_file_name'] = 'gt'
#     seq_info_dict['seq_path'] += '-GT'
#     seq_info_dict['seq'] += '-GT'
#     seq_info_dict['is_gt'] = True

#     # Store the gt file in the common evaluation path
#     gt_file_path = osp.join(seq_path, f"gt/gt.txt")
#     gt_to_eval_path = osp.join(DATA_PATH, 'MOT_eval_gt', seq_name, 'gt')
#     os.makedirs(gt_to_eval_path, exist_ok=True)
#     shutil.copyfile(gt_file_path, osp.join(gt_to_eval_path, 'gt.txt'))

#     return det_df, seq_info_dict, None
