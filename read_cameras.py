import numpy as np
import struct
import collections
import os

# Define data structures
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    0: CameraModel(0, "SIMPLE_PINHOLE", 3),
    1: CameraModel(1, "PINHOLE", 4),
    2: CameraModel(2, "SIMPLE_RADIAL", 4),
    3: CameraModel(3, "RADIAL", 5),
    4: CameraModel(4, "OPENCV", 8),
    5: CameraModel(5, "OPENCV_FISHEYE", 8),
    6: CameraModel(6, "FULL_OPENCV", 12),
    7: CameraModel(7, "FOV", 5),
    8: CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    9: CameraModel(9, "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_cameras_binary(path):
    """Read cameras from a binary file."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODELS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODELS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=np.array(params))
    return cameras

def read_images_binary(path):
    """Read images from a binary file."""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = b""
            while True:
                byte = fid.read(1)
                if byte == b"\x00":
                    break
                image_name += byte
            image_name = image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, num_points2D * 24, "ddq" * num_points2D)
            xys = np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]])
            point3D_ids = np.array(x_y_id_s[2::3])
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def quaternion_to_rotation_matrix(qvec):
    """Convert quaternion to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
    ])

# Paths to the binary files
cameras_bin_path = "capture1/colmap/sparse/0/cameras.bin"
images_bin_path = "capture1/colmap/sparse/0/images.bin"

# Read cameras and images from binary files
cameras = read_cameras_binary(cameras_bin_path)
images = read_images_binary(images_bin_path)

# Extract intrinsic and extrinsic matrices for each camera
intrinsic_matrices = {}
extrinsic_matrices = {}

for camera_id, camera in cameras.items():
    if camera.model == "SIMPLE_PINHOLE":
        fx = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0, 1]
        ])
    elif camera.model == "PINHOLE":
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
        intrinsic_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    else:
        raise NotImplementedError(f"Camera model {camera.model} is not implemented.")
    intrinsic_matrices[camera_id] = intrinsic_matrix

for image_id, image in images.items():
    R = quaternion_to_rotation_matrix(image.qvec)
    t = image.tvec
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))
    #extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))
    extrinsic_matrices[image.camera_id] = extrinsic_matrix

# Save intrinsic and extrinsic matrices to npz files
output_dir = "camera_npy"
os.makedirs(output_dir, exist_ok=True)

for camera_id in intrinsic_matrices.keys():
    intrinsic_matrix = intrinsic_matrices[camera_id]
    extrinsic_matrix = extrinsic_matrices[camera_id]
    
    np.savez(os.path.join(output_dir, f"{camera_id}.npz"), intrinsic=intrinsic_matrix, extrinsic=extrinsic_matrix)
print(intrinsic_matrix)
print(extrinsic_matrix)
print("Matrices saved successfully.")
