import numpy as np
import pickle
import sys
import os
import glob
import argparse


def quat_to_exp_map(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    sign = np.sign(w)
    sign[sign == 0] = 1
    w = w * sign
    x, y, z = x * sign, y * sign, z * sign

    sin_half_angle = np.sqrt(x**2 + y**2 + z**2)
    
    angle = 2.0 * np.arctan2(sin_half_angle, w)
    
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    
    eps = 1e-8
    safe_sin = np.where(sin_half_angle > eps, sin_half_angle, 1.0)
    
    axis_x = x / safe_sin
    axis_y = y / safe_sin
    axis_z = z / safe_sin

    small_angle_mask = sin_half_angle < eps
    axis_x = np.where(small_angle_mask, 0.0, axis_x)
    axis_y = np.where(small_angle_mask, 0.0, axis_y)
    axis_z = np.where(small_angle_mask, 1.0, axis_z)

    exp_map = np.stack([
        axis_x * angle,
        axis_y * angle,
        axis_z * angle
    ], axis=-1)
    
    return exp_map


def convert_gmr_to_mimickit(input_path, output_path, loop_mode="wrap"):
    print(f"Loading input PKL: {input_path}")
    
    with open(input_path, 'rb') as f:
        gmr_data = pickle.load(f)
    
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']
    root_rot_quat = gmr_data['root_rot']
    dof_pos = gmr_data['dof_pos']
    
    print(f"  FPS: {fps}")
    print(f"  root_pos shape: {root_pos.shape}")
    print(f"  root_rot shape: {root_rot_quat.shape}")
    print(f"  dof_pos shape: {dof_pos.shape}")

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
    
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot shape (num_frames, 4), got {root_rot_quat.shape}")
    
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")
    
    root_rot_exp = quat_to_exp_map(root_rot_quat)
    print(f"  root_rot_exp shape: {root_rot_exp.shape}")
    
    frames = np.concatenate([root_pos, root_rot_exp, dof_pos], axis=1)
    print(f"  frames shape: {frames.shape}")
    
    frames_list = [row.tolist() for row in frames]
    
    if loop_mode == "wrap":
        loop_mode_val = 1
    else:
        loop_mode_val = 0
    
    output_dict = {
        'loop_mode': loop_mode_val,
        'fps': fps,
        'frames': frames_list
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR retargeted motion data to MimicKit format."
    )
    parser.add_argument("input_folder", help="Path to folder containing GMR .pkl files")
    parser.add_argument("output_folder", help="Path to output folder for MimicKit .pkl files")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"],
                        help="Loop mode for output motion (default: wrap)")
    
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    loop_mode = args.loop
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    pkl_files = glob.glob(os.path.join(input_folder, "*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {input_folder}")
        sys.exit(0)
    
    print(f"Found {len(pkl_files)} files to convert.")
    print(f"Loop mode: {loop_mode}")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for pkl_file in sorted(pkl_files):
        base_name = os.path.basename(pkl_file)
        file_name_without_ext = os.path.splitext(base_name)[0]
        output_name = file_name_without_ext + ".pkl"
        output_path = os.path.join(output_folder, output_name)
        
        print(f"\nProcessing: {base_name}...")
        try:
            convert_gmr_to_mimickit(pkl_file, output_path, loop_mode)
            print(f"  -> Saved to {output_name}")
            success_count += 1
        except Exception as e:
            print(f"  -> ERROR converting {base_name}: {e}")
            error_count += 1
    
    print("\n" + "=" * 60)
    print(f"Conversion complete: {success_count} succeeded, {error_count} failed.")


if __name__ == "__main__":
    main()
