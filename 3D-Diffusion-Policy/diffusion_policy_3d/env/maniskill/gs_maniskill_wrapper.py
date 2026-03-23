import gym
from gym import spaces
import sys
import os
import time
import torch

from pytorch3d.ops import sample_farthest_points

def crop_workspace_gs(pts, feats, bounds):
    """Crop point cloud to workspace bounds. All inputs/outputs are torch tensors."""
    mask = (pts[:, 0] >= bounds[0]) & (pts[:, 0] <= bounds[1]) & \
           (pts[:, 1] >= bounds[2]) & (pts[:, 1] <= bounds[3]) & \
           (pts[:, 2] >= bounds[4]) & (pts[:, 2] <= bounds[5])
    return pts[mask], feats[mask]


class GSManiskillDP3Wrapper(gym.Env):
    """
    Wrapper that expects its underlying environment to be a `GSWorldWrapper`.
    It extracts the 14-channel Gaussian splats from `gs_movable_pts` for the policy.
    """
    def __init__(self, env, num_points=1024, bounds=[-0.2, 0.8, -0.5, 0.5, 0.01, 1.0]):
        super().__init__()
        self.env = env
        self.num_points = num_points
        self.bounds = bounds

        # Cast Gymnasium action space to legacy Gym action space for DP3's wrapper math
        orig_as = self.env.action_space
        self.action_space = spaces.Box(
            low=orig_as.low,
            high=orig_as.high,
            shape=orig_as.shape,
            dtype=orig_as.dtype
        )

        obs, _ = self.env.reset()
        dummy_state = self._extract_state(obs)
        self.obs_sensor_dim = dummy_state.shape[0]

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-float('inf'), high=float('inf'),
                shape=(self.obs_sensor_dim,),
                dtype='float32'
            ),
            'point_cloud': spaces.Box(
                low=-float('inf'), high=float('inf'),
                # (xyz:3, scaling:3, rotation:4, opacity:1, features_dc:3) -> 14
                shape=(self.num_points, 14), 
                dtype='float32'
            ),
        })

    def _extract_state(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        qpos = obs['agent']['qpos']
        if len(qpos.shape) > 1:
            qpos = qpos[0]
        return qpos.float()

    def _get_obs_dict(self, obs):
        state = self._extract_state(obs)

        # Base 3DGS Model extraction
        # GSWorldWrapper updates `self.env.gs_movable_pts` internally
        # which is a dict mapping link_name -> (link_xyz, link_scaling, link_rotation, link_opacity)
        # Note: we need the SH DC features too, which are static from `initial_merger_robot`
        
        all_channels = []
        base_merger = self.env.unwrapped.initial_merger_robot # Unwrapped to bypass multi-step wrappers if any
        
        # But wait, self.env is just GSWorldWrapper, its unwrapped is BaseEnv
        # Let's cleanly access the GSWorldwrapper
        gs_wrapper = self.env
        while not hasattr(gs_wrapper, "gs_movable_pts"):
            gs_wrapper = gs_wrapper.env

        base_merger = gs_wrapper.initial_merger_robot
        semantics = base_merger._semantics.long().squeeze(-1)
        
        for key, value in gs_wrapper.gs_movable_pts.items():
            xyz, scaling, rotation, opacity = value
            
            # Re-fetch semantic indices to grab features_dc identically
            mask = gs_wrapper._semantic_masks[key]
            indices = gs_wrapper._semantic_indices[key]
            f_dc = base_merger._features_dc[indices]
            
            if len(f_dc.shape) == 3:
                f_dc = f_dc.squeeze(1)
                
            # If batch dim exists, index [0]
            if xyz.dim() == 3:
                xyz = xyz[0]
            if scaling.dim() == 3:
                scaling = scaling[0]
            if rotation.dim() == 3:
                rotation = rotation[0]
            if opacity.dim() == 2:
                opacity = opacity[0]
            if f_dc.dim() == 3:
                f_dc = f_dc[0]
                
            opacity = opacity.unsqueeze(-1) if opacity.dim() == 1 else opacity
                
            channels = torch.cat([xyz, scaling, rotation, opacity, f_dc], dim=-1)
            all_channels.append(channels)

        merged_gaussians = torch.cat(all_channels, dim=0)

        pts = merged_gaussians[:, :3]
        feats = merged_gaussians[:, 3:]

        # Crop
        pts, feats = crop_workspace_gs(pts, feats, self.bounds)

        # FPS Sample
        pts_batch = pts.unsqueeze(0)  # (1, N, 3)
        num_avail = pts_batch.shape[1]
        K = min(self.num_points, num_avail)
        
        if K > 0:
            _, indices = sample_farthest_points(pts_batch, K=K)
            indices = indices.squeeze(0)  # (npoint,)
            
            final_pts = pts[indices]
            final_feats = feats[indices]
        else:
            final_pts = torch.empty((0,3), device=pts.device)
            final_feats = torch.empty((0,11), device=pts.device)

        # Pad if short
        if final_pts.shape[0] < self.num_points:
            padding_size = self.num_points - final_pts.shape[0]
            pad_pts = torch.zeros((padding_size, 3), device=pts.device)
            pad_feats = torch.zeros((padding_size, 11), device=pts.device)
            final_pts = torch.cat([final_pts, pad_pts], dim=0)
            final_feats = torch.cat([final_feats, pad_feats], dim=0)

        pc_14d = torch.cat([final_pts, final_feats], dim=-1).float()

        # Save RGB for rendering
        if 'sensor_data' in obs and 'wrist_cam' in obs['sensor_data']:
            rgb = obs['sensor_data']['wrist_cam']['rgb']
            if len(rgb.shape) == 4:
                rgb = rgb[0]
            self._last_rgb = rgb.clone().to(torch.uint8)

        return {
            'agent_pos': state,
            'point_cloud': pc_14d,
        }

    def step(self, action):
        if hasattr(self.env.unwrapped, "num_envs") and self.env.unwrapped.num_envs > 0:
            if not isinstance(action, torch.Tensor):
                action = torch.from_numpy(action)
            action = action.unsqueeze(0)

        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self._get_obs_dict(obs)

        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        if hasattr(truncated, "item"):
            truncated = bool(truncated.item())
        if hasattr(reward, "item"):
            reward = float(reward.item())

        if isinstance(terminated, (torch.Tensor,)) and terminated.ndim > 0:
            terminated = bool(terminated[0].item())
        if isinstance(truncated, (torch.Tensor,)) and truncated.ndim > 0:
            truncated = bool(truncated[0].item())
        if isinstance(reward, (torch.Tensor,)) and reward.ndim > 0:
            reward = float(reward[0].item())

        done = bool(terminated or truncated)
        reward = float(reward)
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_dict = self._get_obs_dict(obs)
        return obs_dict

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()
