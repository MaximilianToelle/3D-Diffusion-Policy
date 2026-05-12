import torch
import torch.nn.functional as F

import open3d as o3d
import numpy as np


def compute_gs_normals(rotations_9d: torch.Tensor, log_scales: torch.Tensor) -> torch.Tensor:
    """
    Computes temporally-consistent surface normals from 3DGS parameters.
    Extracts the axis corresponding to the minimum scale (the 'flat' direction).
    
    Since rotations_9d rigidly follows the object in the environment, this normal 
    is intrinsically consistent over time without needing dynamic camera-alignment 
    that could flip signs mid-trajectory.
    
    Args:
        rotations_9d: Tensor of shape (..., 9) representing flattened 3x3 matrices
        log_scales: Tensor of shape (..., 3)
        
    Returns:
        normals: Tensor of shape (..., 3) representing unit normals.
    """
    original_shape = rotations_9d.shape[:-1]
    rot_matrices = rotations_9d.view(*original_shape, 3, 3)
    
    # 1. Find the shortest axis (the "flat" direction of the Gaussian)
    _, min_scale_idx = torch.min(log_scales, dim=-1)
    
    # Extract the specific column from the rotation matrix
    axis_selector = F.one_hot(min_scale_idx, num_classes=3).to(dtype=rot_matrices.dtype).unsqueeze(-1)
    normals = torch.matmul(rot_matrices, axis_selector).squeeze(-1)
    
    # Guarantee unit length (should already be unit length since R is orthogonal, but safe)
    normals = F.normalize(normals, p=2, dim=-1)
    
    return normals


def visualize_normals_o3d(positions: torch.Tensor, normals: torch.Tensor, rgb: torch.Tensor = None):
    """
    Blocks execution to visualize a point cloud and its surface normals using Open3D.
    Press 'Q' or close the window to resume the script!
    
    Args:
        positions: Tensor of shape (N, 3)
        normals: Tensor of shape (N, 3)
        rgb: Optional Tensor of shape (N, 3) in range [0, 1]
    """
    
    pcd = o3d.geometry.PointCloud()
    
    # Ensure inputs are 2D (N, 3)
    if len(positions.shape) > 2:
        positions = positions.view(-1, 3)
        normals = normals.view(-1, 3)
        if rgb is not None:
            rgb = rgb.view(-1, 3)
            
    pcd.points = o3d.utility.Vector3dVector(positions.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
    
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb.detach().cpu().numpy())
        
    print("Visualizing PointCloud with Normals. Close the Open3D window to continue execution...")
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
