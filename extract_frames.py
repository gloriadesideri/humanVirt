import cv2
import os
import pycolmap
#import open3d as o3d


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = cap.read()
    count = 0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    while success:
        cv2.imwrite(f"{output_folder}/frame{count:04d}.png", image)
        success, image = cap.read()
        count += 1

    cap.release()
    return frame_count
if __name__ == '__main__':
    # Example usage
    video_path = 'data/tazza.mp4'
    output_folder = 'extracted_frames'
    frame_count = extract_frames(video_path, output_folder)

    # Paths
    image_path = output_folder
    database_path = 'database.db'
    output_path = 'output'
    mvs_path = os.path.join(output_path , "mvs")

    # Make sure the output path exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Feature extraction
    # Correcting the function call for feature extraction
    #pycolmap.extract_features(database_path, image_path, sift_options={"max_num_features": 32})


    # Feature matching
    pycolmap.match_sequential(
        database_path=database_path
    )

    # Dense reconstruction
    maps = pycolmap.incremental_mapping(database_path, image_path, output_path)
    maps[0].write(output_path)
    # dense reconstruction
    pycolmap.undistort_images(mvs_path, output_path, image_path)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

    # Load the point cloud
    point_cloud = o3d.io.read_point_cloud(os.path.join(mvs_path, 'dense.ply'))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
