from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.common.gs_reconstructor import GSReconstructor


class GSManiskillDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            scene_gs_cfg_name="fr3_stack",
            num_gaussians=1024
            ):
        super().__init__()
        self.task_name = task_name
        self.num_gaussians = num_gaussians
        self.scene_gs_cfg_name = scene_gs_cfg_name
        
        # Determine actor poses keys based on the task name/scene config
        self.actor_keys = []
        if "stack" in self.scene_gs_cfg_name.lower():
            self.actor_keys = ['005_tomato_soup_can', 'dtc_red_tomato_can_fr3']
        else:
            raise ValueError("Other scenes are not yet implemented!")    

        load_keys = ['state', 'action']
        for act in self.actor_keys:
            load_keys.append(f'actor_pose_{act}')
            
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=load_keys)
            
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
            
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
            
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        
        # Initialize GSReconstructor once
        self.reconstructor = GSReconstructor(
            scene_gs_cfg_name=self.scene_gs_cfg_name, 
            robot_uid="fr3_umi_wrist435",
            num_gaussians=self.num_gaussians
        )

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # TODO: How do I normalize the gsplat representation? 
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """ Returns data as dict of torch tensors """
        agent_pos = torch.from_numpy(sample['state']).to(torch.float32)

        # Instead of using the 6-D pointcloud from the Zarr file, 
        # we generate the 14-D Gaussian representation
        # Note: sequence sampling returns (T, D) arrays. We need to reconstruct for each timestep.
        
        T = agent_pos.shape[0]
        gsplats_over_t = []
        
        # Track persistent indices across the sequence for temporal consistency
        gaussian_indices = None
        
        for t in range(T):
            qpos_t = agent_pos[t]
            actor_poses_t = {}
            for act in self.actor_keys:
                pose_key = f'actor_pose_{act}'
                actor_poses_t[act] = torch.from_numpy(sample[pose_key][t])
                
            gs_t, new_indices = self.reconstructor.reconstruct(qpos_t, actor_poses_t, gaussian_indices=gaussian_indices)
            # reconstruct provides new_indices if gaussian_indices is None, else it uses gaussian_indices
            if gaussian_indices is None and new_indices is not None:
                gaussian_indices = new_indices
                
            gsplats_over_t.append(gs_t)
            
        gsplats = torch.stack(gsplats_over_t, dim=0)

        data = {
            'obs': {
                'gsplats': gsplats,
                'agent_pos': agent_pos,
            },
            'action': torch.from_numpy(sample['action']).to(torch.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        torch_data = self._sample_to_data(sample)
        return torch_data
