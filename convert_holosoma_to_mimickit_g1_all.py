import numpy as np
import pickle
import sys
import os
import glob


def convert_npz_to_pkl(input_path, output_path):
    print(f"Loading input NPZ: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    if 'qpos' not in data:
        raise ValueError(f"Input file {input_path} does not contain 'qpos' key.")
    
    qpos = data['qpos']
    fps = int(data.get('fps', 30))
    
    print(f"  qpos shape: {qpos.shape}, fps: {fps}")
    
    if qpos.shape[1] == 36:
        root_pos = qpos[:, 0:3]
        
        quats = qpos[:, 3:7]
        
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / (norms + 1e-8)

        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        
        v = np.stack([x, y, z], axis=1)
        v_norm = np.linalg.norm(v, axis=1)
        
        angle = 2.0 * np.arctan2(v_norm, w)
        
        v_norm_safe = np.where(v_norm < 1e-8, 1.0, v_norm)
        factor = angle / v_norm_safe
        
        factor = np.where(v_norm < 1e-6, 2.0, factor)
        
        root_rot = v * factor[:, np.newaxis]

        joints = qpos[:, 7:]
        
        q_converted = np.hstack([root_pos, root_rot, joints])
        frames_list = [row.tolist() for row in q_converted]
        
    else:
        print(f"  Dimensions {qpos.shape[1]} do not match expected 36. Passing through unmodified.")
        frames_list = [row.tolist() for row in qpos]
    
    output_dict = {
        'loop_mode': 0,
        'fps': fps,
        'frames': frames_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_holosoma_to_mimickit_g1_all.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    npz_files = glob.glob(os.path.join(input_folder, "*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        sys.exit(0)
        
    print(f"Found {len(npz_files)} files to convert.")

    success_count = 0
    error_count = 0

    for npz_file in sorted(npz_files):
        base_name = os.path.basename(npz_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        pkl_file_name = file_name_without_ext + ".pkl"
        output_path = os.path.join(output_folder, pkl_file_name)
        
        print(f"Processing: {base_name}...")
        try:
            convert_npz_to_pkl(npz_file, output_path)
            print(f"  -> Saved to {pkl_file_name}")
            success_count += 1
        except Exception as e:
            print(f"  -> ERROR converting {base_name}: {e}")
            error_count += 1

    print(f"\nConversion complete: {success_count} succeeded, {error_count} failed.")
