import gym
from gym import spaces
import torch

from pytorch3d.ops import sample_farthest_points
from gsworld.constants import fr3_gs_semantics, obj_gs_semantics


STATIC_ACTORS = {"table-workspace", "ground"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ManiSkillDP3Wrapper(gym.Env):
    def __init__(self, env, num_points=1024):
        super().__init__()
        self.env = env
        self.num_points = num_points
        self.cam = 'right_cam'

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
                shape=(self.num_points, 6),
                dtype='float32'
            ),
        })

        # Preprocessing env actor and robot strings to object ids for later pcd masking through segmentation
        all_actor_names = list(self.env.unwrapped.scene.actors.keys())
        moving_actors = sorted([a for a in all_actor_names if a not in STATIC_ACTORS])

        actor_ids = torch.tensor([obj_gs_semantics[actor] for actor in moving_actors], device=DEVICE, dtype=torch.long)
        robot_ids = []
        for value in fr3_gs_semantics.values():
            if isinstance(value, list):
                robot_ids.extend(value)
            else:
                robot_ids.append(value)
        robot_ids = torch.tensor(robot_ids, device=DEVICE, dtype=torch.long)
        self.all_target_ids = torch.cat([robot_ids, actor_ids])        

    def _extract_state(self, obs):
        # We assume the env returns unbatched shapes or batched arrays of size (1, ...)
        qpos = obs['agent']['qpos']
        if len(qpos.shape) > 1:
            qpos = qpos[0]
        return qpos.float()

    def _get_obs_dict(self, obs):
        state = self._extract_state(obs)
       
        depth = obs['sensor_data'][self.cam]['depth']
        rgb = obs['sensor_data'][self.cam]['rgb']
        seg = obs['sensor_data'][self.cam]['segmentation']
        intr = obs['sensor_param'][self.cam]['intrinsic_cv']
        extr = obs['sensor_param'][self.cam]['gl_cam2sapien_world']

        # handle batch dimensions if the env returns (1, H, W, C)
        if len(depth.shape) == 4:
            depth = depth[0]
            rgb = rgb[0]
            seg = seg[0]
            intr = intr[0]
            extr = extr[0]

        self._last_rgb = rgb.clone().to(torch.uint8)

        pcd_pts = get_sapien_world_pcd(
            depth / 1000.0,     # extrinsics are in meter!
            intr, 
            extr
        )

        # Process pcd - filter out static actors and background + FPS
        flat_seg = seg.reshape(-1)
        mask = torch.isin(flat_seg, self.all_target_ids)
        masked_pcd_pts = pcd_pts[mask]

        final_pcd = farthest_point_sample(masked_pcd_pts, npoint=self.num_points)

        return {
            'agent_pos': state,
            'point_cloud': final_pcd,
        }

    def step(self, action):
        # SAPIEN might need batched action if num_envs=1
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

        # Ensure primitive python types to prevent crashes later on
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


def get_sapien_world_pcd(depth, intrinsic, gl_cam2sapien_world):
    """
    Converts depth to pcd in sapien world coordinates assuming graphics convention (x-axis right, y-axis upward, z-axis backward) in extrinsics. 
    All inputs/outputs are torch tensors on the same device.
    """
    
    H, W = depth.shape[:2]
    device = depth.device

    u, v = torch.meshgrid(torch.arange(W, device=device, dtype=depth.dtype) + 0.5,  # pixel center depth
                          torch.arange(H, device=device, dtype=depth.dtype) + 0.5,
                          indexing='xy')

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    d = depth[..., 0]
    
    z = - d     # assuming positive depth values
    x = (u - cx) * d / fx   # X is right -> (u - cx) is positive on the right
    y = -(v - cy) * d / fy   # Y is upward -> (v - cy) is positive downward!

    pts_cam = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    rotation_mat = gl_cam2sapien_world[:3, :3]
    translation_vec = gl_cam2sapien_world[:3, 3].unsqueeze(0)
    
    pcd_pts_world = torch.matmul(pts_cam, rotation_mat.transpose(-2, -1)) + translation_vec
    
    return pcd_pts_world


def farthest_point_sample(pts, npoint):
    pts_batch = pts.unsqueeze(0)  # (1, N, 3)
    _, indices = sample_farthest_points(pts_batch, K=npoint)
    indices = indices.squeeze(0)  # (npoint,)
    return pts[indices]