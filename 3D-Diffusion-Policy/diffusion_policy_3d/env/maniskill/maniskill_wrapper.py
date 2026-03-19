import gym
import numpy as np
from gym import spaces
import sys
import os

def depth_to_point_cloud(depth, rgb, intrinsic, extrinsic_cv):
    H, W = depth.shape[:2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    z = depth[..., 0]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=-1).reshape(-1, 4)
    rgb_flat = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    
    ext_4x4 = np.eye(4)
    ext_4x4[:3, :] = extrinsic_cv
    cam2world = np.linalg.inv(ext_4x4)
    
    pts_world = (cam2world @ pts_cam.T).T[:, :3]
    
    valid = (z.flatten() > 0.01) & (z.flatten() < 3.0)
    return pts_world[valid], rgb_flat[valid]

def crop_workspace(pts, rgb, bounds):
    mask = (pts[:, 0] >= bounds[0]) & (pts[:, 0] <= bounds[1]) & \
           (pts[:, 1] >= bounds[2]) & (pts[:, 1] <= bounds[3]) & \
           (pts[:, 2] >= bounds[4]) & (pts[:, 2] <= bounds[5])
    return pts[mask], rgb[mask]

def farthest_point_sample(pts, rgb, npoint):
    N = pts.shape[0]
    if N == 0:
        return np.zeros((npoint, 3)), np.zeros((npoint, 3))
    
    if N <= npoint:
        indices = np.random.choice(N, npoint, replace=True)
        return pts[indices], rgb[indices]
        
    centroids = np.zeros(npoint, dtype=np.int32)
    distance = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = pts[farthest, :]
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
        
    return pts[centroids], rgb[centroids]


class ManiSkillDP3Wrapper(gym.Env):
    def __init__(self, env, num_points=1024, bounds=[-0.2, 0.8, -0.5, 0.5, 0.01, 1.0]):
        super().__init__()
        self.env = env
        self.num_points = num_points
        self.bounds = bounds
        self.cams = ['wrist_cam', 'right_cam']
        
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
                low=-np.inf, high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_points, 6),
                dtype=np.float32
            ),
        })

    def _extract_state(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        # In mani_skill, wrapper might return unbatched.
        qpos = obs['agent']['qpos'].cpu().numpy()
        if len(qpos.shape) > 1:
            qpos = qpos[0]
        return qpos.astype(np.float32)

    def _get_obs_dict(self, obs):
        state = self._extract_state(obs)
        
        step_pts, step_rgb = [], []
        for cam in self.cams:
            depth = obs['sensor_data'][cam]['depth'].cpu().numpy()
            rgb = obs['sensor_data'][cam]['rgb'].cpu().numpy()
            intr = obs['sensor_param'][cam]['intrinsic_cv'].cpu().numpy()
            extr = obs['sensor_param'][cam]['extrinsic_cv'].cpu().numpy()
            
            # handle batch dimensions if the env returns (1, H, W, C)
            if len(depth.shape) == 4:
                depth = depth[0]
                rgb = rgb[0]
                intr = intr[0]
                extr = extr[0]
                
            if cam == self.cams[0]:
                self._last_rgb = rgb.copy().astype(np.uint8)
                
            pts, colors = depth_to_point_cloud(depth, rgb, intr, extr)
            step_pts.append(pts)
            step_rgb.append(colors)
        
        merged_pts = np.concatenate(step_pts, axis=0)
        merged_rgb = np.concatenate(step_rgb, axis=0)
        
        # Process pc
        merged_pts, merged_rgb = crop_workspace(merged_pts, merged_rgb, self.bounds)
        merged_pts, merged_rgb = farthest_point_sample(merged_pts, merged_rgb, npoint=self.num_points)
        
        pc_6d = np.concatenate([merged_pts, merged_rgb], axis=-1).astype(np.float32)
        
        return {
            'agent_pos': state,
            'point_cloud': pc_6d,
        }

    def step(self, action):
        # SAPIEN might need batched action if num_envs=1
        if hasattr(self.env.unwrapped, "num_envs") and self.env.unwrapped.num_envs > 0:
            action = np.expand_dims(action, axis=0)
            
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_dict = self._get_obs_dict(obs)
        
        if hasattr(terminated, "item"):
            terminated = bool(terminated.item())
        if hasattr(truncated, "item"):
            truncated = bool(truncated.item())
        if hasattr(reward, "item"):
            reward = float(reward.item())
            
        if isinstance(terminated, np.ndarray) and terminated.ndim > 0:
            terminated = bool(terminated[0])
        if isinstance(truncated, np.ndarray) and truncated.ndim > 0:
            truncated = bool(truncated[0])
        if isinstance(reward, np.ndarray) and reward.ndim > 0:
            reward = float(reward[0])
            
        # Ensure primitive python types to prevent np.all() crashes later on
        done = bool(terminated or truncated)
        reward = float(reward)
        return obs_dict, reward, done, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_dict = self._get_obs_dict(obs)
        return obs_dict

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_rgb'):
            return self._last_rgb
        return np.zeros((256, 256, 3), dtype=np.uint8)
