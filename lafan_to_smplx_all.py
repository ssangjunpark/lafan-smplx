import argparse
import os
import glob

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
# from general_motion_retargeting.utils.lafan1 import (
    # load_lafan1_file as _load_lafan1_file,
# )
from general_motion_retargeting.utils.lafan1 import load_bvh_file as _load_lafan1_file

def load_lafan1_file(file_path):
    return _load_lafan1_file(file_path)[0]


def get_smplx_local_joint(
    lafan_frame_name, lafan_frame_quat, lafan_parent_frame_name, lafan_parent_frame_quat
):
    parent_frame_global_rot = (
        R.from_quat(lafan_parent_frame_quat, scalar_first=True)
        * lafan_frame_offsets[lafan_parent_frame_name]
    )
    frame_global_quat = (
        R.from_quat(lafan_frame_quat, scalar_first=True)
        * lafan_frame_offsets[lafan_frame_name]
    )
    frame_local_quat = parent_frame_global_rot.inv() * frame_global_quat
    return frame_local_quat.as_rotvec()


lafan_frame_offsets = {
    "world": R.from_euler("x", 0),
    "Hips": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "LeftFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "LeftToe": R.from_euler("y", np.pi / 2),
    "RightUpLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightLeg": R.from_euler("z", np.pi / 2) * R.from_euler("y", np.pi / 2),
    "RightFoot": R.from_euler("z", 0.37117860986509) * R.from_euler("y", np.pi / 2),
    "RightToe": R.from_euler("y", np.pi / 2),
    "Spine": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine1": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Spine2": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Neck": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "Head": R.from_euler("z", -np.pi / 2) * R.from_euler("y", -np.pi / 2),
    "LeftShoulder": R.from_euler("x", np.pi / 2),
    "LeftArm": R.from_euler("x", np.pi / 2),
    "LeftForeArm": R.from_euler("x", np.pi / 2),
    "LeftHand": R.from_euler("x", np.pi / 2),
    "RightShoulder": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightForeArm": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
    "RightHand": R.from_euler("z", np.pi) * R.from_euler("x", -np.pi / 2),
}

smplx_to_lafan_map = {
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "spine1": "Spine",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "spine2": "Spine1",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "spine3": "Spine2",
    "left_foot": "LeftToe",
    "right_foot": "RightToe",
    "neck": "Neck",
    "left_collar": "LeftShoulder",
    "right_collar": "RightShoulder",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


betas = {
    "smpl": torch.tensor(
        [
            [
                0.9597,
                1.0887,
                -2.1717,
                -0.8611,
                1.3940,
                0.1401,
                -0.2469,
                0.3182,
                -0.2482,
                0.3085,
            ]
        ],
        dtype=torch.float32,
    ),
    "smplx": torch.tensor(
        [
            [
                1.4775,
                0.6674,
                -1.1742,
                0.4731,
                1.2984,
                -0.2159,
                1.5276,
                -0.3152,
                -0.6441,
                -0.2986,
                0.5089,
                -0.6354,
                0.3321,
                -0.1099,
                -0.3060,
                -0.7330,
            ]
        ],
        dtype=torch.float32,
    ),
}

def process_single_bvh(bvh_file, output_file, smplx_model_path, model_type, rerun):
    # Load the LAFAN BVH file
    try:
        data = read_bvh(bvh_file)
        frames = load_lafan1_file(bvh_file)
    except Exception as e:
        print(f"Skipping {bvh_file} - failed to load: {e}")
        return

    n_frames = len(frames)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    body_model = smplx.create(
        model_path=smplx_model_path,
        model_type=model_type,
        use_pca=False,
        betas=betas[model_type].to(device),
    )

    trans_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    root_orient_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    body_pose_tensor = torch.zeros(
        (n_frames, body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
    )
    lafan_centroids = []
    
    for frame in range(n_frames):
        root_orient = (
            R.from_quat(frames[frame]["Hips"][1], scalar_first=True)
            * lafan_frame_offsets["Hips"]
        )

        body_pose = torch.zeros(
            (body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
        )
        for i, joint_name in enumerate(
            JOINT_NAMES[1:22]
        ):
            if joint_name not in smplx_to_lafan_map:
                continue
            lafan_joint_name = smplx_to_lafan_map[joint_name]
            parent_name = data.bones[data.parents[data.bones.index(lafan_joint_name)]]
            body_pose[i] = torch.from_numpy(
                get_smplx_local_joint(
                    lafan_joint_name,
                    frames[frame][lafan_joint_name][1],
                    parent_name,
                    frames[frame][parent_name][1],
                )
            )

        lafan_positions = torch.from_numpy(
            np.array(
                [frames[frame][name][0] for name in frames[frame] if "Mod" not in name],
                dtype=np.float32,
            )
        ).to(device)
        lafan_centroids.append(lafan_positions.mean(dim=0))
        
        root_orient_tensor[frame] = torch.tensor(
            root_orient.as_rotvec(), device=device
        ).float()
        body_pose_tensor[frame] = body_pose.clone()

    lafan_centroids = torch.stack(lafan_centroids)

    with torch.no_grad():
        smplx_kwargs = {
            "betas": betas[model_type].to(device).view(1, -1).expand(n_frames, -1),
            "global_orient": root_orient_tensor,
            "body_pose": body_pose_tensor.flatten(1),
            "return_full_pose": True,
        }
        if model_type == "smplx":
            smplx_kwargs.update({
                "jaw_pose": torch.zeros(n_frames, 3, device=device, dtype=torch.float32),
                "leye_pose": torch.zeros(n_frames, 3, device=device, dtype=torch.float32),
                "reye_pose": torch.zeros(n_frames, 3, device=device, dtype=torch.float32),
                "left_hand_pose": torch.zeros(n_frames, 45, device=device, dtype=torch.float32),
                "right_hand_pose": torch.zeros(n_frames, 45, device=device, dtype=torch.float32),
                "expression": torch.zeros(n_frames, 10, device=device, dtype=torch.float32),
            })
        smplx_output = body_model(**smplx_kwargs)

    smplx_centroids = smplx_output.joints[:, :22].mean(dim=1)
    trans_tensor = lafan_centroids - smplx_centroids

    smplx_data = {
        "betas": body_model.betas.detach().cpu().squeeze().numpy(),
        "root_orient": root_orient_tensor.cpu().numpy(),
        "pose_body": body_pose_tensor.flatten(1).cpu().numpy(),
        "trans": trans_tensor.cpu().numpy(),
    }

    num_frames = smplx_data["pose_body"].shape[0]
    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"]).float().view(1, -1).expand(num_frames, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(), # (N, 3)
        body_pose=torch.tensor(smplx_data["pose_body"]).float(), # (N, 63)
        transl=torch.tensor(smplx_data["trans"]).float(), # (N, 3)
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )

    if rerun:
        for frame in range(n_frames):
            rr.set_time_sequence("frame", frame)
            rr.log(
                "nameless_motion",
                rr.Mesh3D(
                    vertex_positions=smplx_output.vertices[frame]
                    .detach()
                    .cpu()
                    .numpy(),
                    triangle_indices=body_model.faces,
                ),
            )

    if output_file is not None:
        full_poses_tensor = torch.cat(
            [root_orient_tensor.unsqueeze(1), body_pose_tensor],
            dim=1,
        )
        if model_type == "smplx":
            # Just mimicking variable name logic here, not strictly needed for np.savez below which uses specific keys
            pass 
        
        np.savez(
            output_file,
            trans=trans_tensor.cpu().numpy(),
            gender="neutral",
            mocap_frame_rate=30,
            betas=body_model.betas.detach().cpu().squeeze().numpy(),
            root_orient=root_orient_tensor.cpu().numpy(),
            pose_body=body_pose_tensor.flatten(1).cpu().numpy(),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_folder", type=str, required=True, help="Folder containing .bvh files")
    parser.add_argument("--smplx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_type",
        type=str,
        default="smplx",
        choices=["smpl", "smplx"],
    )
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output .npz files")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Whether to visualize the result in Rerun",
    )
    args = parser.parse_args()

    if args.rerun:
        import rerun as rr
        rr.init("lafan_to_smplx_all", spawn=True)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    bvh_files = glob.glob(os.path.join(args.bvh_folder, "*.bvh"))
    print(f"Found {len(bvh_files)} BVH files in {args.bvh_folder}")

    for bvh_file in bvh_files:
        filename = os.path.basename(bvh_file)
        name, ext = os.path.splitext(filename)
        output_file = os.path.join(args.output_folder, name + ".npz")
        
        print(f"Processing {filename} -> {output_file}")
        try:
            process_single_bvh(
                bvh_file, 
                output_file, 
                args.smplx_model_path, 
                args.model_type, 
                args.rerun
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
