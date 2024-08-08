import os
import shutil
from PIL import Image

# Define the source directories and target directories
npz_src_folder = 'camera_npy'
png_src_folder = 'capture1/rgb/1x'
cameras_target_folder = 'Multiview-3DMM-Fitting/single_cam/cameras'
images_target_folder = 'Multiview-3DMM-Fitting/single_cam/images'

# Create the target directories if they don't exist
os.makedirs(cameras_target_folder, exist_ok=True)
os.makedirs(images_target_folder, exist_ok=True)

camera_ids = []


# Process .npz files
for i in range(1, 95):  # Adjusted range to start from 1
      # Create folder name with leading zeros
    npz_file_name = f"{i}.npz"  # File name with 6 digits
    folder_name = f"{i:04d}"
    camera_subfolder = os.path.join(cameras_target_folder, folder_name)
    os.makedirs(camera_subfolder, exist_ok=True)

    # Move the .npz file
    npz_src_path = os.path.join(npz_src_folder, npz_file_name)
    npz_dest_path = os.path.join(camera_subfolder, f"camera_00.npz")
    if os.path.exists(npz_src_path):
        shutil.copy(npz_src_path, npz_dest_path)
        camera_ids.append(f"{i:04d}")
    

# Process .png files

for i in range(1, 95):  # Adjusted range to start from 1
      # Create folder name with leading zeros
    png_file_name = f"{i:06d}.png"  # File name with 6 digits

    folder_name = f"{i:04d}"
    image_subfolder = os.path.join(images_target_folder, folder_name)
    os.makedirs(image_subfolder, exist_ok=True)
    

    # Crop the image to a square and move it
    png_src_path = os.path.join(png_src_folder, png_file_name)
    png_dest_path = os.path.join(image_subfolder, f"image_00.png")
    camera_id = f"{i:04d}"
    if os.path.exists(png_src_path):
        with Image.open(png_src_path) as img:
            # Crop the image to a square
            width, height = img.size
            min_side = min(width, height)
            left = (width - min_side) / 2
            top = (height - min_side) / 2
            right = (width + min_side) / 2
            bottom = (height + min_side) / 2
            img_cropped = img.crop((left, top, right, bottom))
            
            # Save the cropped image
            img_cropped.save(png_dest_path)
            
            # Print the image size
            print(f"Image image_{i:04d}.png size: {img_cropped.size}")

    

print("Files have been moved and cropped successfully.")
print("Image IDs:", camera_ids)