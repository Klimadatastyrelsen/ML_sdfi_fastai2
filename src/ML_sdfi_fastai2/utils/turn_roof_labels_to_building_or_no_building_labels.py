import argparse
import os
import rasterio
import numpy as np

def process_geotiff(input_path, output_path):
    # Open the GeoTIFF file
    with rasterio.open(input_path) as src:
        # Read the image data as a numpy array
        img_array = src.read(1)  # Read the first band
        
        # Replace all values larger than 1 with 2
        img_array[img_array > 1] = 2
        
        # Define the metadata for the output file
        metadata = src.meta.copy()
        metadata.update({
            'dtype': 'int32',  # Update the data type to accommodate the new values
            'count': 1  # We are only processing one band
        })
        
        # Write the modified image to a new GeoTIFF file
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(img_array, 1)

def main():
    parser = argparse.ArgumentParser(description="Process GeoTIFF images by replacing pixel values.")
    parser.add_argument('--input_folder', type=str, help="Path to the input folder containing GeoTIFF images.")
    parser.add_argument('--output_folder', type=str, help="Path to the output folder to save processed images.")

    parser.add_argument('--path_to_txt', type=str, default = None, help="optional: Path to the txt file listing files to process.")
    
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if args.path_to_txt:
        with open(args.path_to_txt) as f:
            filenames = [line.rstrip() for line in f.readlines()]
    else:
        filenames = os.listdir(input_folder)
    
    # Process each GeoTIFF
    for filename in filenames:
        if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_geotiff(input_path, output_path)
            print(f"Processed and saved {filename} to {output_folder}")

if __name__ == "__main__":
    main()
