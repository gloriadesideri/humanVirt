import numpy as np
import cv2

def project_points(points, camera_matrix):
    """
    Projects 3D points onto the 2D image plane using the camera matrix.

    Args:
        points (np.ndarray): Nx3 array of 3D points.
        camera_matrix (np.ndarray): 3x4 or 3x3 intrinsic camera matrix.

    Returns:
        np.ndarray: Nx2 array of 2D points in the image plane.
    """
    # Convert points to homogeneous coordinates
    if points.shape[1] == 3:
        points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        points_homogeneous = points

    # Ensure the camera matrix is 3x4 for proper matrix multiplication
    if camera_matrix.shape == (3, 3):
        camera_matrix = np.hstack([camera_matrix, np.zeros((3, 1))])

    # Project points
    points_2d_homogeneous = camera_matrix @ points_homogeneous.T

    # Convert from homogeneous to Cartesian coordinates
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

    return points_2d.T

def get_colors_from_image(points_2d, image):
    """
    Queries the color for each 2D point from the image.

    Args:
        points_2d (np.ndarray): Nx2 array of 2D points.
        image (np.ndarray): Image from which to query the colors.

    Returns:
        np.ndarray: Nx3 array of colors.
    """
    # Ensure points are within image bounds
    points_2d = np.round(points_2d).astype(int)
    h, w, _ = image.shape
    valid_indices = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
    valid_points_2d = points_2d[valid_indices]

    # Query colors
    colors = np.zeros((points_2d.shape[0], 3), dtype=np.uint8)
    colors[valid_indices] = image[valid_points_2d[:, 1], valid_points_2d[:, 0]]

    return colors

# Example usage:
# Replace with actual colmap points, camera matrix, and image
colmap_points = np.load('xyzs.npy') # Replace with actual colmap points
mesh_points =np.load('mesh_points.npy')    # Replace with actual mesh points
camera_mats= np.load('Multiview-3DMM-Fitting/single_cam/cameras/0055/camera_00.npz')
# Replace with actual intrinsic camera matrix
camera_matrix = camera_mats['intrinsic']

# Replace with the actual image
image = cv2.imread('Multiview-3DMM-Fitting/single_cam/images/0055/image_00.png')

# Project mesh points onto the image plane
mesh_points_2d = project_points(mesh_points, camera_matrix)

# Get colors from the image
mesh_colors = get_colors_from_image(mesh_points_2d, image)
np.save('mesh_colors_proj.npy',mesh_colors )

# mesh_colors now contains the colors for each mesh point
