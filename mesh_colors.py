import torch
import numpy as np
from pytorch3d.ops import knn_points
from tqdm import tqdm


colmap_points= np.load('xyzs.npy')
mesh_points=np.load('mesh_points.npy')
colmap_colors=np.load('rgbs.npy')

colmap_points_t = torch.tensor(colmap_points, dtype=torch.float32)
center_t = colmap_points_t.mean(dim=0)


def radial_propagation_torch(colmap_points, colmap_colors, mesh_points, center, k=4, p=2, local_search_radius=0.8):
    # Convert data to PyTorch tensors and move to GPU
    colmap_points_t = torch.tensor(colmap_points, dtype=torch.float32).cuda()
    colmap_colors_t = torch.tensor(colmap_colors, dtype=torch.float32).cuda()
    mesh_points_t = torch.tensor(mesh_points, dtype=torch.float32).cuda()
    center_t = torch.tensor(center, dtype=torch.float32).cuda()

    # Sort mesh points based on distance from the center
    distances_from_center_t = torch.norm(mesh_points_t - center_t, dim=1)
    sorted_indices_t = torch.argsort(distances_from_center_t)

    # Initialize colors for mesh points
    mesh_colors_t = torch.zeros((mesh_points_t.shape[0], 3), device='cuda')
    colored_t = torch.zeros(mesh_points_t.shape[0], dtype=torch.bool, device='cuda')

    # Initialize a list to keep track of points to process
    points_to_process = [sorted_indices_t[0].item()]
    colored_t[sorted_indices_t[0]] = True

    # Create a boolean mask for points within the local search radius
    def within_radius(points, center, radius):
        distances = torch.norm(points - center, dim=1)
        return distances < radius

    # Add tqdm progress bar
    pbar = tqdm(total=len(points_to_process), desc='Processing Points')

    while points_to_process:
        current_idx = points_to_process.pop(0)
        current_point_t = mesh_points_t[current_idx]

        # Update tqdm progress bar
        pbar.update(1)
        pbar.set_postfix({'remaining': len(points_to_process)})

        # Find all points within the local search radius
        mask = within_radius(colmap_points_t, current_point_t, local_search_radius)
        nearby_points_t = colmap_points_t[mask]
        nearby_colors_t = colmap_colors_t[mask]

        if nearby_points_t.size(0) > 0:
            # Perform k-NN search using PyTorch3D
            current_point_t_expanded = current_point_t[None, None, :]
            nearby_points_t_expanded = nearby_points_t[None, :, :]
            knn_result = knn_points(current_point_t_expanded, nearby_points_t_expanded, K=k)

            # Gather k-NN points and colors
            knn_indices = knn_result.idx.squeeze()
            knn_points_t = nearby_points_t[knn_indices]
            knn_colors_t = nearby_colors_t[knn_indices]

            # Calculate distances and weights
            dists_t = torch.norm(knn_points_t - current_point_t, dim=1)
            weights_t = 1 / (dists_t ** p)
            weights_t /= weights_t.sum()

            # Calculate weighted average color
            mesh_colors_t[current_idx] = torch.sum(weights_t[:, None] * knn_colors_t, dim=0)

        # Add uncolored neighboring mesh points to the processing list
        mask = within_radius(mesh_points_t, current_point_t, local_search_radius)
        for i in mask.nonzero(as_tuple=True)[0]:
            if not colored_t[i]:
                points_to_process.append(i.item())
                colored_t[i] = True

        # Update tqdm total
        pbar.total = len(points_to_process) + 1
        pbar.refresh()

    pbar.close()
    return mesh_colors_t.cpu().numpy()
mesh_colors = radial_propagation_torch(colmap_points, colmap_colors, mesh_points, center_t)
np.save('mesh_colors_small', mesh_colors)


