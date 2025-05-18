from PIL import Image
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import os


def convert_image(png_file):
    """
    Convert a single PNG image to JPEG and delete the original PNG file.
    """
    try:
        # print(f"Processing {png_file.name} on PID: {os.getpid()}")
        # Construct the JPEG filename
        jpg_file = png_file.with_suffix('.JPEG')
        # Open and convert the PNG to JPEG
        with Image.open(png_file) as img:
            # Convert to RGB to handle transparency
            rgb_img = img.convert('RGB')
            # Save as JPEG
            rgb_img.save(jpg_file, 'JPEG')
        # Delete the original PNG file to free up space
        png_file.unlink()
    except Exception as e:
        print(f"Error processing {png_file.name}: {str(e)}")


def process_directory(directory, num_workers=64):
    """
    Convert all PNG images in the specified directory to JPEG format using parallel processing.
    """
    png_files = list(directory.glob('*.[pP][nN][gG]'))  # Match PNG files

    print(f"Starting multiprocessing with {num_workers} workers.")
    print(f"Number of PNG files to process: {len(png_files)}")
    print(f"Number of CPU cores available: {cpu_count()}")
    print(f"PID of main process: {os.getpid()}")

    chunksize = 10 #max(1, len(png_files) // num_workers)
    print(f"Chunk size for processing: {chunksize}")
    # with Pool(num_workers) as pool:
    #     print(f"Active workers: {len(pool._pool)}")
    #     pool.map(convert_image, png_files, chunksize=chunksize)

    with Pool(num_workers) as pool:
        for _ in pool.imap_unordered(convert_image, png_files, chunksize=chunksize):
            pass  # This will keep the workers busy without blocking

def main():
    base_path = Path('/scratch-shared/scur0551/GenImage_download/GenImage/stable_diffusion_v_1_5')
    # Process 'ai' subfolders in 'train' and 'val'
    sub_dirs = ['train/ai'] #] #'val/ai', 

    for sub_dir in sub_dirs:
        directory_path = base_path / sub_dir
        if directory_path.exists():
            print(f"Processing directory: {directory_path}")
            start_time = time.time()
            process_directory(directory_path, num_workers=64)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Processing time for {directory_path}: {elapsed_time:.2f} seconds", flush=True)
        else:
            print(f"Directory not found: {directory_path}")


if __name__ == "__main__":
    main()
