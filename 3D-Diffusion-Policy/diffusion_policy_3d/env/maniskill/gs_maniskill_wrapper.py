import gym
from gym import spaces
import sys
import os
import time
import torch

from pytorch3d.ops import sample_farthest_points


class GSManiskillDP3Wrapper(gym.Env):
    """
    Wrapper that expects its underlying environment to be a `GSWorldWrapper`.
    It extracts the 14-channel Gaussian splats from `gs_movable_pts` for the policy.
    """
    def __init__(self, env, num_gaussians=1024):
        super().__init__()
        self.env = env
        self.num_gaussians = num_gaussians

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
            'gsplats': spaces.Box(
                low=-float('inf'), high=float('inf'),
                # (xyz:3, scaling:3, rotation:4, opacity:1, features_dc:3) -> 14
                shape=(self.num_gaussians, 14), 
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
        
        all_gaussians = []
        base_merger = self.env.initial_merger_robot
        semantics = base_merger._semantics.long().squeeze(-1)
        
        for key, value in self.env.gs_movable_pts.items():
            xyz, scaling, rotation, opacity = value
            
            # Re-fetch semantic indices to grab features_dc identically
            indices = self.env._semantic_indices[key]
            f_dc = base_merger._features_dc[indices]
            
            if len(f_dc.shape) == 3:
                f_dc = f_dc.squeeze(1)
                
            xyz = xyz.view(-1, 3)
            scaling = scaling.view(-1, 3)
            rotation = rotation.view(-1, 4)
            opacity = opacity.view(-1, 1)
            f_dc = f_dc.view(-1, 3)
                
            gs_channels = torch.cat([xyz, scaling, rotation, opacity, f_dc], dim=-1)
            all_gaussians.append(gs_channels)

        merged_gaussians = torch.cat(all_gaussians, dim=0)

        pts = merged_gaussians[:, :3]
        feats = merged_gaussians[:, 3:]

        # FPS Sample for persistent temporal consistency
        if self.gaussian_indices is None:
            pts_batch = pts.unsqueeze(0)  # (1, N, 3)
            num_avail = pts_batch.shape[1]
            assert (num_avail >= self.num_gaussians), "Less Gaussians than requested for sampling!"
            
            _, indices = sample_farthest_points(pts_batch, K=self.num_gaussians)
            self.gaussian_indices = indices.squeeze(0)  # (npoint,)
            
        final_pts = pts[self.gaussian_indices]
        final_feats = feats[self.gaussian_indices]
        
        gsplats = torch.cat([final_pts, final_feats], dim=-1).float()

        # Save RGB for rendering
        if 'sensor_data' in obs and 'right_cam' in obs['sensor_data']:
            rgb = obs['sensor_data']['right_cam']['rgb'].view(-1, 3)
            self._last_rgb = rgb.clone().to(torch.uint8)

        return {
            'agent_pos': state,
            'gsplats': gsplats,
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
        # Reset sampled gaussian indices for new episode
        self.gaussian_indices = None

        obs, info = self.env.reset(**kwargs)
        obs_dict = self._get_obs_dict(obs)
        return obs_dict

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb.cpu().numpy()
        return torch.zeros((256, 256, 3), dtype=torch.uint8).numpy()
