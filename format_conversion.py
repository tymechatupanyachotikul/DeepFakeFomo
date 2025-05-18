from PIL import Image
from pathlib import Path

# Convert a PNG image to JPG format from images in a given path

def convert_png_to_jpg(input_path, output_path, max_images=100):
    # Create the output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Iterate through all PNG files in the input directory
    counter = 0
    for png_file in Path(input_path).glob('*.png'):
        # Open the PNG image
        with Image.open(png_file) as img:
            # Convert to RGB (JPG does not support transparency)
            rgb_img = img.convert('RGB')
            # Save as JPG in the output directory
            jpg_file = Path(output_path) / (png_file.stem + '.jpg')
            rgb_img.save(jpg_file, 'JPEG')
        counter += 1
        # Stop after processing the specified number of images
        if counter >= max_images:
            break

def copy_images(input_path, output_path):
    # Create the output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Iterate through all files in the input directory
    counter = 0
    for file in Path(input_path).glob('*'):
        # Copy the file to the output directory
        destination = Path(output_path) / file.name
        file.replace(destination)
        counter += 1

        if counter >= 100:
            break

if __name__ == "__main__":
    input_directory = './LaRE/GenImage/BigGAN/val/ai_full/'  # Replace with your input directory
    output_directory = './LaRE/GenImage/BigGAN/val/ai/'  # Replace with your output directory
    # convert_png_to_jpg(input_directory, output_directory)
    # print(f"Converted PNG images from {input_directory} to JPG format in {output_directory}.")
    copy_images(input_directory, output_directory)