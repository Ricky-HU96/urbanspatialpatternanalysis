# core/shape_index.py
import rasterio
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def calculate_shape_index(input_height_tif, output_tif_path, threshold_min=1, threshold_max=None, output_csv_path=None, output_png_path=None):
    try:
        print(f"Start processing Shape Index: {input_height_tif}")
        with rasterio.open(input_height_tif) as src:
            # Read all data (To ensure the accuracy of connected component analysis, it is recommended to process the whole image. If memory is insufficient, process in blocks. Here, the whole image logic is demonstrated)
            data = src.read(1)
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform

            # 1. Corresponds to Figure 1: Threshold processing (Create binary mask)
            if threshold_max is None:
                threshold_max = np.max(data)
            
            # Generate binary mask: Pixels within the threshold range are marked as True (Building)
            mask = (data >= threshold_min) & (data <= threshold_max)

            # 2. Corresponds to Figure 1: Connected component labeling (Label connected patches)
            # connectivity=2 means 8-connectivity (diagonals are also considered connected)
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)
            
            print(f"Detected {num_labels} building patches")

            # Initialize output array, background set to NaN or 0
            shape_index_array = np.zeros((height, width), dtype=np.float32)
            patch_metrics = []

            # 3. Corresponds to Figure 1: Execute loop to calculate shape index (Calculate shape index for each patch)
            if num_labels > 0:
                props = regionprops(labeled_mask)
                for prop in props:
                    # Filter out noise
                    if prop.area <= 0: continue

                    area = prop.area
                    perimeter = prop.perimeter

                    # Core formula: SI = P / (2 * sqrt(pi * A))
                    # Note: If the shape is very regular (circle), value is 1; the more irregular, the larger the value
                    shape_index = perimeter / (2 * np.sqrt(np.pi * area))

                    # Assign calculation results back to the corresponding position in the raster
                    # This way, all pixels of each patch will have the SI value of that patch
                    coords = prop.coords # Get coordinates of all pixels in the patch (row, col)
                    shape_index_array[coords[:, 0], coords[:, 1]] = shape_index

                    # Collect statistical data
                    patch_metrics.append({
                        'Patch_ID': prop.label,
                        'Patch_Area_pixels': area,
                        'Perimeter': perimeter,
                        'Shape_Index': shape_index
                    })

        # --- Subsequent output logic remains unchanged ---
        if output_csv_path and patch_metrics:
            pd.DataFrame(patch_metrics).to_csv(output_csv_path, index=False)
            print(f"Shape Index CSV result saved to: {output_csv_path}")

        if output_png_path and patch_metrics:
            plt.figure(figsize=(12, 7))
            df = pd.DataFrame(patch_metrics)
            plt.hist(df['Shape_Index'], bins=50, color='#d62728', alpha=0.7, edgecolor='black')
            plt.xlabel('Shape Index (1.0 = Circle)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Patch Shape Index')
            plt.grid(axis='y', alpha=0.5)
            plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
            plt.close()

        # Save result raster
        with rasterio.open(
            output_tif_path, 'w', driver='GTiff',
            height=height, width=width, count=1,
            dtype=shape_index_array.dtype, crs=crs,
            transform=transform, compress='lzw', nodata=0) as dst:
            dst.write(shape_index_array, 1)

        return True
    except Exception as e:
        print(f"Error occurred while calculating Shape Index: {e}")
        import traceback
        traceback.print_exc()
        return False