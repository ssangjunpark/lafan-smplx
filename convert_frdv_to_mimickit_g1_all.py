import numpy as np
import pickle
import sys
import os
import glob

def extract_pose_data(frame):
    root_pos = frame[..., 0:3]
    root_rot = frame[..., 3:6]
    joint_dof = frame[..., 6:]
    return root_pos, root_rot, joint_dof

def convert_npy_to_pkl(input_path, output_path):
    print(f"Loading input NPY: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    content = None
    if data.shape == ():
        content = data.item()
        if isinstance(content, list):
            content = content[0]
    else:
        if isinstance(data, np.ndarray) and len(data) > 0:
             content = data[0] if isinstance(data[0], dict) else data

    if content is None:
         if isinstance(data, dict):
             content = data
         else:
             try:
                 if data.size == 1:
                     content = data.item()
             except:
                 pass

    if content is None or not isinstance(content, dict) or 'q' not in content:
        if isinstance(data, dict) and 'q' in data:
            content = data
        elif isinstance(data, np.ndarray) and data.dtype == 'O' and data.size == 1:
             possible_content = data.item()
             if isinstance(possible_content, dict) and 'q' in possible_content:
                 content = possible_content
    
    if content is None or 'q' not in content:
        raise ValueError(f"Input file {input_path} does not contain 'q' key or valid dictionary structure.")

    q_data = content['q']
    fps = content.get('fps', 30)
    
    if q_data.shape[1] == 36:
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
        frames_list = [row.tolist() for row in q_converted]
        
    else:
        print(f"Dimensions {q_data.shape[1]} do not match expected 36. Passing through unmodified.")
        frames_list = [row.tolist() for row in q_data]
    
    output_dict = {
        'loop_mode': 0,
        'fps': int(fps),
        'frames': frames_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_frdv_to_mimickit_g1_all.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    npy_files = glob.glob(os.path.join(input_folder, "*.npy"))
    
    if not npy_files:
        print(f"No .npy files found in {input_folder}")
        sys.exit(0)
        
    print(f"Found {len(npy_files)} files to convert.")

    for npy_file in npy_files:
        base_name = os.path.basename(npy_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        pkl_file_name = file_name_without_ext + ".pkl"
        output_path = os.path.join(output_folder, pkl_file_name)
        
        print(f"Processing: {base_name}...")
        try:
            convert_npy_to_pkl(npy_file, output_path)
            print(f"  -> Saved to {pkl_file_name}")
        except Exception as e:
            print(f"  -> ERROR converting {base_name}: {e}")
