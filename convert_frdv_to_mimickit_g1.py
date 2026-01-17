import numpy as np
import pickle
import sys
import os

def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof

def convert_npy_to_pkl(input_path, output_path):
    print(f"Loading input NPY: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    if data.shape == ():
        content = data.item()
        if isinstance(content, list):
            content = content[0]
    else:
        if isinstance(data, np.ndarray) and len(data) > 0:
             content = data[0] if isinstance(data[0], dict) else data

    if 'q' not in content:
        raise ValueError("Input file does not contain 'q' key.")

    q_data = content['q']
    fps = content.get('fps', 30)
    
    print(f"Data shape (q): {q_data.shape}")
    print(f"FPS: {fps}")
    
    if q_data.shape[1] == 36:
        print("Detected 36 dimensions. Converting Quaternion (cols 3:7) to Axis-Angle (3 dims)...")
        root_pos = q_data[:, 0:3]
        
        quats = q_data[:, 3:7]
        
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / (norms + 1e-8)
        
        x, y, z, w = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        v = np.stack([x, y, z], axis=1)
        v_norm = np.linalg.norm(v, axis=1)
        
        angle = 2.0 * np.arctan2(v_norm, w)
        

        v_norm_safe = np.where(v_norm < 1e-8, 1.0, v_norm)
        factor = angle / v_norm_safe
        
        factor = np.where(v_norm < 1e-6, 2.0, factor)
        
        root_rot = v * factor[:, np.newaxis]
        
        joints = q_data[:, 7:]
        
        q_converted = np.hstack([root_pos, root_rot, joints])
        print(f"Converted shape: {q_converted.shape}")
        
        frames_list = [row.tolist() for row in q_converted]
        
    else:
        print(f"Dimensions {q_data.shape[1]} do not match expected 36. Passing through unmodified.")
        frames_list = [row.tolist() for row in q_data]
    
    output_dict = {
        'loop_mode': 0,
        'fps': int(fps),
        'frames': frames_list
    }
    
    print(f"Writing output PKL: {output_path}")
    print(f"Number of frames: {len(frames_list)}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)
        
    print("Conversion complete.")

if __name__ == "__main__":
    input_file = '/home/sangjunpark/Documents/main_branch/mocap_retargeting/log/kin_trajectories/unitree_g1/unitree_g1_aiming1_subject1_0_7184_2026-01-16_16-57-44.npy'
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    output_file = 'unitree_g1_converted.pkl' 
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        
    convert_npy_to_pkl(input_file, output_file)
