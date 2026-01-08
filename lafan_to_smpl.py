import argparse

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R
from smplx.joint_names import JOINT_NAMES

from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
from general_motion_retargeting.utils.lafan1 import load_bvh_file as _load_lafan1_file


def load_lafan1_file(file_path):
    return _load_lafan1_file(file_path)[0]


def get_smpl_local_joint(
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


# LAFAN to SMPL coordinate frame offset rotations
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

smpl_to_lafan_map = {
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

default_betas = torch.tensor(
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
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    ],
    dtype=torch.float32,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LAFAN BVH motion capture to SMPL format"
    )
    parser.add_argument(
        "--bvh_file",
        type=str,
        required=True,
        help="Path to the input LAFAN BVH file",
    )
    parser.add_argument(
        "--smpl_model_path",
        type=str,
        required=True,
        help="Path to SMPL model directory",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output .npz file",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="Gender of the SMPL model",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Whether to visualize the result in Rerun",
    )
    args = parser.parse_args()

    data = read_bvh(args.bvh_file)
    frames = load_lafan1_file(args.bvh_file)
    n_frames = len(frames)

    device = "cpu"

    body_model = smplx.create(
        model_path=args.smpl_model_path,
        model_type="smpl",
        gender=args.gender,
        num_betas=10,
    )

    if args.rerun:
        import rerun as rr
        rr.init("lafan_to_smpl", spawn=True)

    trans_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    global_orient_tensor = torch.zeros((n_frames, 3), device=device, dtype=torch.float32)
    
    body_pose_tensor = torch.zeros(
        (n_frames, body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
    )
    lafan_centroids = []

    for frame in range(n_frames):
        global_orient = (
            R.from_quat(frames[frame]["Hips"][1], scalar_first=True)
            * lafan_frame_offsets["Hips"]
        )

        body_pose = torch.zeros(
            (body_model.NUM_BODY_JOINTS, 3), device=device, dtype=torch.float32
        )
        
        for i, joint_name in enumerate(JOINT_NAMES[1:24]):
            if joint_name not in smpl_to_lafan_map:
                continue
            lafan_joint_name = smpl_to_lafan_map[joint_name]
            parent_name = data.bones[data.parents[data.bones.index(lafan_joint_name)]]
            body_pose[i] = torch.from_numpy(
                get_smpl_local_joint(
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

        global_orient_tensor[frame] = torch.tensor(
            global_orient.as_rotvec(), device=device
        ).float()
        body_pose_tensor[frame] = body_pose.clone()

    lafan_centroids = torch.stack(lafan_centroids)

    with torch.no_grad():
        smpl_output = body_model(
            betas=default_betas[:, :10].to(device).expand(n_frames, -1),
            global_orient=global_orient_tensor,
            body_pose=body_pose_tensor.flatten(1),
            return_full_pose=True,
        )

    smpl_centroids = smpl_output.joints[:, :24].mean(dim=1)
    trans_tensor = lafan_centroids - smpl_centroids
    
    global_orient_flat = global_orient_tensor.cpu().numpy()  # (N, 3)
    body_pose_flat = body_pose_tensor.flatten(1).cpu().numpy()  # (N, 69)

    poses = np.zeros((n_frames, 156), dtype=np.float32)
    poses[:, :3] = global_orient_flat  # global_orient
    poses[:, 3:72] = body_pose_flat  # body_pose (23 joints * 3 = 69)

    betas_output = np.zeros(16, dtype=np.float32)
    betas_output[:10] = default_betas[0, :10].numpy()

    dmpls = np.zeros((n_frames, 8), dtype=np.float32)

    if args.rerun:
        for frame in range(n_frames):
            rr.set_time_sequence("frame", frame)
            rr.log(
                "smpl_mesh",
                rr.Mesh3D(
                    vertex_positions=smpl_output.vertices[frame].detach().cpu().numpy(),
                    triangle_indices=body_model.faces,
                ),
            )

    np.savez(
        args.output_file,
        trans=trans_tensor.cpu().numpy(),  # (N, 3)
        gender=args.gender,  # scalar string
        mocap_framerate=30,  # LAFAN is 30 fps
        betas=betas_output,  # (16,)
        dmpls=dmpls,  # (N, 8)
        poses=poses,  # (N, 156)
    )
    
    print(f"Saved SMPL data to {args.output_file}")
    print(f"  - trans: {trans_tensor.shape}")
    print(f"  - poses: {poses.shape}")
    print(f"  - betas: {betas_output.shape}")
    print(f"  - dmpls: {dmpls.shape}")
    print(f"  - gender: {args.gender}")
    print(f"  - mocap_framerate: 30")
