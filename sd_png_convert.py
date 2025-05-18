from PIL import Image
from pathlib import Path
import time

def convert_png_to_jpg_in_place(directory, max_images=100):
    """
    Convert up to 'max_images' PNG images in the specified directory to JPEG format in place.
    Deletes the original PNG after successful conversion.
    """
    counter = 0
    for png_file in Path(directory).glob('*.jpg'):
        # Construct the JPEG filename
        jpg_file = png_file.with_suffix('.JPEG')

        # Open and convert the PNG to JPEG
        with Image.open(png_file) as img:
            # Convert to RGB to handle transparency
            rgb_img = img.convert('RGB')
            # Save as JPEGc
            rgb_img.save(jpg_file, 'JPEG')

        # Delete the original PNG file to free up space
        png_file.unlink()

        # print(f"Converted {png_file.name} to {jpg_file.name} and deleted original PNG.")
        counter += 1
        
        # Stop after processing the specified number of images
        if counter >= max_images:
            break


def main():
    base_path = Path('/scratch-shared/scur0551/GenImage_download/GenImage/VQDM')
    # base_path = Path('/gpfs/scratch1/shared/scur0551/GenImage_download/GenImage/stable_diffusion_v_1_5/train')
    # Process 'ai' subfolders in 'train' and 'val' for 100 images each
    sub_dirs = ['test_val_backup/val/ai'] #, 'val/ai']
    
    for sub_dir in sub_dirs:
        directory_path = base_path / sub_dir
        if directory_path.exists():
            print(f"Processing directory: {directory_path}")
            # convert_png_to_jpg_in_place(directory_path, max_images=100)

            start_time = time.time()
            convert_png_to_jpg_in_place(directory_path, max_images=100)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing time for {directory_path}: {elapsed_time:.2f} seconds", flush=True)
        else:
            print(f"Directory not found: {directory_path}")

if __name__ == "__main__":
    main()
