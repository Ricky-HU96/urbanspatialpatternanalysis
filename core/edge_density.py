# core/edge_density.py
import rasterio
import numpy as np
import pandas as pd
from skimage.measure import label, find_contours
from skimage.transform import resize

def calculate_edge_density(input_height_tif, output_tif_path, threshold_min=2, grid_size=50, output_csv_path=None):
    try:
        print(f"Starting Edge Density Analysis with Grid Size: {grid_size}px")
        
        with rasterio.open(input_height_tif) as src:
            data = src.read(1)
            meta = src.meta.copy()
            transform = src.transform
            # Handle NoData
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0, data)
        
        original_rows, original_cols = data.shape

        # 1. Preprocessing: Binarization and Connected Component Labeling (Identify Patches)
        # Assume areas with height > threshold_min are the patches we want to analyze (e.g., buildings)
        binary_map = (data >= threshold_min).astype(int)
        # Label the entire image to distinguish different buildings/patches
        labeled_image = label(binary_map, connectivity=2)

        # 2. Calculate the number of grid rows and columns
        n_rows = original_rows // grid_size
        n_cols = original_cols // grid_size
        
        # Result matrix, size reduced (because it became a grid map)
        density_result = np.zeros((n_rows, n_cols), dtype=np.float32)

        print(f"Processing {n_rows} x {n_cols} grids...")

        # 3. Core loop: Iterate through each grid (Iterate through each unit area)
        for i in range(n_rows):
            for j in range(n_cols):
                # Slice to get current grid
                r_start, r_end = i * grid_size, (i + 1) * grid_size
                c_start, c_end = j * grid_size, (j + 1) * grid_size
                
                # Data of current grid
                grid_patch = labeled_image[r_start:r_end, c_start:c_end]
                
                # Get all patch IDs in this grid (exclude background 0)
                unique_ids = np.unique(grid_patch)
                unique_ids = unique_ids[unique_ids != 0]
                
                total_edge_length = 0.0
                
                # Calculate edges for each patch within the grid
                for uid in unique_ids:
                    # Generate binary mask for current patch
                    patch_mask = (grid_patch == uid)
                    # find_contours finds contours
                    # level=0.5 is the standard contour finding threshold for binary images
                    contours = find_contours(patch_mask, 0.5)
                    
                    for contour in contours:
                        total_edge_length += len(contour)
                
                # Calculate density: Total edge length / Grid area
                # grid_area is usually grid_size * grid_size
                area = (r_end - r_start) * (c_end - c_start)
                if area > 0:
                    density_result[i, j] = total_edge_length / area

        # 4. Output processing
        if output_csv_path:
            # Save raw grid value matrix
            pd.DataFrame(density_result).to_csv(output_csv_path, header=False, index=False)

        # 5. Key step: Restore size (Upscaling)
        # Upscale the calculated small grid heatmap back to the original map size, creating a mosaic-like effect (as shown on the right side of Figure 1)
        # order=0 uses nearest neighbor interpolation to maintain the "grid" visual effect
        final_output = resize(density_result, (original_rows, original_cols), order=0, preserve_range=True, anti_aliasing=False)

        # Update metadata and save
        meta.update({
            "dtype": "float32",
            "count": 1,
            "compress": "lzw"
        })

        with rasterio.open(output_tif_path, "w", **meta) as dst:
            dst.write(final_output.astype(np.float32), 1)

        print(f"Edge Density calculation completed: {output_tif_path}")
        return True

    except Exception as e:
        print(f"Error in edge_density: {e}")
        import traceback
        traceback.print_exc()
        return False