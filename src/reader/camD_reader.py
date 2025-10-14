import os
import json
import pickle
import random
import logging
import numpy as np
from tqdm import tqdm

ACTIVITIES = ['B1', 'F1', 'B2', 'F2', 'B3',
              'F3', 'B4', 'F4', 'B5', 'F5']

class CamD_Reader():
    def __init__(self, dataset_root_folder, out_folder, split_ratio=0.7, **kwargs):
        self.max_channel = 3
        self.max_frame = 100
        self.max_joint = 17
        self.max_person = 6

        self.dataset_root_folder = dataset_root_folder
        self.out_folder = out_folder

        # Divide train and eval samples
        pose_dir = os.path.join(dataset_root_folder,'pose')
        smp_idx = range(len(os.listdir(pose_dir)))
        random.shuffle(smp_idx)
        split_idx = int(len(smp_idx) * split_ratio)
        self.training_samples = smp_idx[:split_idx]
        self.eval_samples = smp_idx[split_idx:]

        # Create label-to-idx map
        self.class2idx = {name: i for i, name in enumerate(ACTIVITIES)}


    def read_pose_and_object(self, pose_path, obj_path):
        """
        Reads pose and object JSON files and returns:
        - skeleton_data: np.array of shape (T, M, V, C)
        where C = [x, y, confidence]
        - object_info: dict {"obj_name": [[x,y,confidence], [x,y,confidence], ...]}

        Args:
            pose_path (str): Path to pose JSON files.
            obj_path (str): Directory containingPath to object JSON files.
            T (int): Total number of frames (optional fixed length).
            V (int): Number of joints per person.
            M (int): Number of persons (actors).
            C (int): Channels per joint (x, y, conf).

        Returns:
            skeleton_data (np.ndarray): shape (T, M, V, C)
            object_info (dict): {"obj_name": [[x, y], ...]}
        """

        T, M, V, C = self.max_frame, self.max_person, self.max_joint, self.max_channel

        # --- Initialize pose data array ---
        skeleton_data = np.zeros((T, M, V, C), dtype=np.float32)

        # --- Load Pose JSON ---
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)

        # Fill skeleton_data
        for frame_data in pose_data.get("frames", []):
            t = frame_data["frame_index"]
########################## Might need to introduce my downsampling function here ##################################################
            if t >= T:
                continue
            for m, person in enumerate(frame_data["poses"][:M]):
                kpts = np.array(person["keypoints"], dtype=np.float32)
                conf = np.array(person["confidence"], dtype=np.float32)
                # Shape: (V, 2) → stack with conf to get (V, 3)
                joint = np.concatenate([kpts, conf[:, None]], axis=1)
                v = min(V, joint.shape[0])
                skeleton_data[t, m, :v, :] = joint[:v]

        # --- Load Object JSON ---
        with open(obj_path, 'r') as f:
            obj_data = json.load(f)

        # Assuming one object of interest per video
        obj_name = None
        obj_coords = []

        for frame_data in obj_data.get("frames", []):
            objects = frame_data.get("objects", [])
            if not objects:
                obj_coords.append([np.nan, np.nan])  # no object in frame
                continue
            # Pick first object (or highest confidence)
            obj = max(objects, key=lambda o: o.get("confidence", 0))
            if obj_name is None:
                obj_name = obj["object_name"]
            cx, cy = obj["center"]
            conf = obj["confidence"]
            obj_coords.append([float(cx), float(cy), float(conf)])

        object_info = {obj_name or "unknown": obj_coords}

        return skeleton_data, object_info


    def gendata(self, phase):

        res_skeleton = []
        res_obj = []
        group_label = []
        video_list = os.listdir(path) # use the directory for pose
        videos = video_list[self.training_samples] if phase == 'train' else video_list[self.eval_samples]
        
        iterizer = tqdm(videos, dynamic_ncols=True)
        for filename in iterizer:

            video_id = filename.split('.')[0].split('_')[0] # loading from a .json file

            # Skip the random walking files
            if video_id[:-5] not in ['RED', 'YELLOW', 'BLACK', 'GREEN', 'BLUE', 'WHITE']:
                continue

            # path to joints and object files
            joint_path = os.path.join(
                self.dataset_root_folder, 'pose', filename)
            object_path = os.path.join(
                self.dataset_root_folder, 'object', filename)
            
            # save group name for each video sample
            group_label.append([self.class2idx[video_id[-5:-3]], video_id])
                
                
            # Get joint/+object information
            joint_data, object_data = self.read_pose_and_object(joint_path, object_path)
            res_skeleton.append(joint_data)
            res_obj.append(object_data)
                
        # Save label
        os.makedirs(self.out_folder, exist_ok=True)
        with open(os.path.join(self.out_folder, phase + '_label.pkl'), 'wb') as f:
            pickle.dump(group_label, f)
        
        # Save pose data
        res_skeleton = np.array(res_skeleton)
        np.save(os.path.join(self.out_folder, phase + '_data.npy'), res_skeleton)
        
        # Save obj data
        with open(os.path.join(self.out_folder, phase + '_object_data.json', "w")) as f:
            json.dump(res_obj, f)
        
    def start(self):
        for phase in ['train', 'eval']:
            logging.info('Phase: {}'.format(phase))
            self.gendata(phase)
