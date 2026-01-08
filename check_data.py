import torch
import numpy as np

# path = "/home/sangjunpark/Desktop/ACCAD_SMPL/ACCAD/ACCAD/Female1General_c3d/A1 - Stand_poses.npz"
path = "/home/sangjunpark/Downloads/ACCAD_SMPLX/Female1General_c3d/A1_-_Stand_stageii.npz"
# path = ""

data = np.load(path, allow_pickle=True)
print(data.files)
breakpoint()

"""
For SMPL files:

['trans', 'gender', 'mocap_framerate', 'betas', 'dmpls', 'poses']

data['trans'].shape (N, 3)
data['gender'].shape ()
data['mocap_framerate'].shape ()
data['betas'].shape (16,)
data['dmpls'].shape (N, 8)
data['poses'].shape (N, 156)


For SMPLX files:
['gender', 'surface_model_type', 'mocap_frame_rate', 'mocap_time_length', 
'markers_latent', 'latent_labels', 'markers_latent_vids', 'trans', 'poses', 
'betas', 'num_betas', 'root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'pose_eye']

BUT GMR requires:
['trans', 'gender', 'mocap_frame_rate', 'betas', 'root_orient', 'pose_body']
data['trans'].shape (N, 3)
data['gender'].shape ()
data['mocap_frame_rate'].shape ()
data['betas'].shape (16,)
data['root_orient'].shape (N, 3)
data['pose_body'].shape (N, 63)
"""