import time
import torch
import os
import gymnasium as gym

# Import GSWorld utilities
from gsworld.utils.gs_utils import transform_gaussians
from gsworld.utils.pcd_utils import extract_rigid_transform
from gsworld.utils.gaussian_merger import GaussianModelMerger
from gsworld.constants import (
    fr3_gs_semantics, 
    sim2gs_arm_trans, 
    obj_gs_semantics, 
    robot_scan_qpos, 
    object_offset,
    sim2gs_object_transforms,
    object_scale,
    CFG_DIR
)
from mani_skill.utils.structs.pose import Pose


class GSReconstructor:
    """
    Utility class to reconstruct the 14-channel 3D Gaussian Splatting representation
    from robot `qpos` and `actor_poses`.
    """
    def __init__(
        self, 
        scene_gs_cfg_name="fr3_stack", 
        robot_uid="fr3_umi_wrist435", 
        device="cuda", 
        num_gaussians=1024, 
    ):
        self.device = torch.device(device)
        self.scene_gs_cfg_name = scene_gs_cfg_name
        self.num_gaussians = num_gaussians

        # Constants from GSWorld
        self.gs_semantics = fr3_gs_semantics
        self.sim2gs_arm_trans = torch.tensor(sim2gs_arm_trans, dtype=torch.float32, device=self.device)
        self.obj_gs_semantics = obj_gs_semantics
        
        # Load the base unified Gaussian model
        merger = GaussianModelMerger()
        path = os.path.join(CFG_DIR, f"{scene_gs_cfg_name}.json")
        merger.load_models_from_config(path)
        self.base_model = merger.merge_models()

        # Temporary ManiSkill environment to compute Forward Kinematics (FK)
        # Instead of implementing FK entirely by hand, we use the SAPIEN engine
        env4moving = gym.make(
            "Empty-v1",
            obs_mode="none",
            reward_mode="none",
            enable_shadow=False,
            robot_uids=robot_uid,
            render_mode=None,
            sim_config=dict(sim_freq=100, control_freq=20),
            sim_backend="auto",
        )
        env4moving.reset(seed=0)
        self.env4moving = env4moving.unwrapped

        self.env4moving.agent.robot.set_qpos(robot_scan_qpos[robot_uid])
        
        # Cache the initial transformation matrices of the links in the GS scan pose
        self.gs_link_pose_mats = []
        for link in self.env4moving.agent.robot.get_links():
            link_mat = link.pose.to_transformation_matrix().to(self.device).squeeze()
            if link_mat.dim() == 2:
                self.gs_link_pose_mats.append(link_mat)
            else:
                self.gs_link_pose_mats.append(link_mat.squeeze(0))

        # Cache semantic indices to avoid integer `==` comparisons during the training loop
        semantics = self.base_model._semantics.long().squeeze(-1).to(self.device).contiguous()
        self._semantic_indices = {} # key -> index tensor for transform_gaussians
        
        for link in self.env4moving.agent.robot.get_links():
            if link.name in self.gs_semantics:
                target = torch.tensor(self.gs_semantics[link.name], device=semantics.device).long()
                self._semantic_indices[link.name] = torch.where(torch.isin(semantics, target))[0]
                
        for actor_key, sem_id in self.obj_gs_semantics.items():
            self._semantic_indices[actor_key] = torch.where(semantics == sem_id)[0]

        # Performance optimization: cache all moving indices and object mapping
        all_ordered_indices = []
        gaussian_id_to_object_id = []
        self._moving_objects = [] # list of dicts with metadata
        
        object_idx = 0
        
        # 1. Robot Links
        robot_links = self.env4moving.agent.robot.get_links()
        for i, link in enumerate(robot_links):
            if link.name in self._semantic_indices:
                indices = self._semantic_indices[link.name]
                all_ordered_indices.append(indices)
                gaussian_id_to_object_id.append(torch.full((indices.shape[0],), object_idx, device=self.device, dtype=torch.long))
                self._moving_objects.append({
                    "type": "robot",
                    "name": link.name,
                    "link_idx": i
                })
                object_idx += 1
                
        # 2. Actors (Objects)
        for actor_key in self.obj_gs_semantics:
            if actor_key in self._semantic_indices:
                indices = self._semantic_indices[actor_key]
                all_ordered_indices.append(indices)
                gaussian_id_to_object_id.append(torch.full((indices.shape[0],), object_idx, device=self.device, dtype=torch.long))
                self._moving_objects.append({
                    "type": "actor",
                    "name": actor_key
                })
                object_idx += 1
                
        if all_ordered_indices:
            self._all_moving_indices = torch.cat(all_ordered_indices)
            self._gaussian_id_to_object_id = torch.cat(gaussian_id_to_object_id)
        else:
            self._all_moving_indices = torch.empty(0, dtype=torch.long, device=self.device)
            self._gaussian_id_to_object_id = torch.empty(0, dtype=torch.long, device=self.device)
            
        # Pre-cache inverses and transforms
        self.gs_link_pose_mats_inv = torch.stack([torch.linalg.inv(m) for m in self.gs_link_pose_mats])
        self.sim2gs_arm_trans_inv = torch.linalg.inv(self.sim2gs_arm_trans)

    def reconstruct(self, qpos: torch.Tensor, actor_poses: dict, gaussian_indices=None):
        """
        Reconstructs the 3D Gaussian cloud for a single timestep.
        qpos: (9,) tensor
        actor_poses: dict mapping actor_name -> (13,) tensor
        gaussian_indices: (Optional) 1D tensor of indices to sample. If None, runs FPS on the full scene and returns (cloud, indices).
        Returns:
            A tuple of (sampled_gaussians, indices):
                sampled_gaussians: (num_gaussians, 14) tensor representing the 3d gaussian representation
                indices: (num_gaussians,) tensor of indices used for FPS sampling, or None if gaussian_indices were provided
        """
        
        if qpos.dim() > 1:
            qpos = qpos.squeeze()
            
        # 1. Update robot kinematics
        self.env4moving.agent.robot.set_qpos(qpos)
        
        # 2. Collect parts
        start_time = time.time()
        
        K = len(self._moving_objects)
        batch_rot = torch.zeros((K, 3, 3), device=self.device)
        batch_trans = torch.zeros((K, 3), device=self.device)
        batch_scale = torch.ones(K, device=self.device)
        
        robot_links = self.env4moving.agent.robot.get_links()
        
        for i, src in enumerate(self._moving_objects):
            if src["type"] == "robot":
                # Extract pure (4, 4) transform from SAPIEN
                link_mat = robot_links[src["link_idx"]].pose.to_transformation_matrix()[0].to(self.device)
                
                # Note: Hardcoded xarm object offset
                if "xarm" in self.env4moving.agent.uid:
                    for j in range(3):
                        link_mat[j, 3] += object_offset["xarm_arm"][j]
                
                # Compute T_sim2gs
                T = self.sim2gs_arm_trans @ link_mat @ self.gs_link_pose_mats_inv[src["link_idx"]] @ self.sim2gs_arm_trans_inv
                batch_rot[i] = T[:3, :3]
                batch_trans[i] = T[:3, 3]
                
            elif src["type"] == "actor":
                actor_key = src["name"]
                if actor_key not in actor_poses or actor_key not in sim2gs_object_transforms:
                    continue
                    
                full_pose = actor_poses[actor_key]
                pose_tensor = full_pose[:7].to(self.device).view(-1) # Extract position and quaternion
                
                # SAPIEN Pose.create expects batched input
                mat = Pose.create(pose=pose_tensor.unsqueeze(0)).to_transformation_matrix()[0].to(self.device)
                
                if actor_key in object_offset:
                    for j in range(3):
                        mat[j, 3] += object_offset[actor_key][j]
                        
                sim2gs_obj_trans_inv = torch.tensor(sim2gs_object_transforms[actor_key], device=self.device, dtype=torch.float32).inverse()
                T = self.sim2gs_arm_trans @ mat @ sim2gs_obj_trans_inv
                
                M_rigid, scale, R_rigid, t = extract_rigid_transform(T)
                batch_scale[i] = scale * object_scale[actor_key]
                
        # Determine subset of points to transform
        if gaussian_indices is not None:
            active_moving_indices = self._all_moving_indices[gaussian_indices]
            active_object_ids = self._gaussian_id_to_object_id[gaussian_indices]
        else:
            active_moving_indices = self._all_moving_indices
            active_object_ids = self._gaussian_id_to_object_id
            
        # Expand transforms to active gaussians
        final_rot = batch_rot[active_object_ids]   # (N, 3, 3)
        final_trans = batch_trans[active_object_ids] # (N, 3)
        final_scale = batch_scale[active_object_ids] # (N,)
        
        # 3. Apply Unified Transformation
        gs_xyz, gs_scaling, gs_rotation, gs_opacity = transform_gaussians(
            self.base_model,
            selected_indices=active_moving_indices,
            scale=final_scale,
            rot_mat=final_rot,
            translation=final_trans,
            new_opacity=None,
        )
        
        # Assemble channels natively
        opacity = gs_opacity.view(-1, 1)
        f_dc = self.base_model._features_dc[active_moving_indices].view(-1, 3)
            
        gaussian_cloud = torch.cat([gs_xyz, gs_scaling, gs_rotation, opacity, f_dc], dim=-1)
        
        # 4. Handle Sampling
        if gaussian_indices is None:
            pts = gaussian_cloud[:, :3]
            feats = gaussian_cloud[:, 3:]
            
            from pytorch3d.ops import sample_farthest_points
            pts_batch = pts.unsqueeze(0)  # (1, N, 3)
            _, indices = sample_farthest_points(pts_batch, K=min(self.num_gaussians, pts_batch.shape[1]))
            indices = indices.squeeze(0)  # (ngaussian,)
            
            final_pts = pts[indices]
            final_feats = feats[indices]
            
            assert (final_pts.shape[0] == self.num_gaussians), "Less Gaussians than requested for sampling!" 
                
            gaussian_cloud = torch.cat([final_pts, final_feats], dim=-1)
            return gaussian_cloud, indices
        else:
            # We already ran on the subset!
            # The gaussian_cloud is strictly the `num_gaussians` we wanted.   
            return gaussian_cloud, None
