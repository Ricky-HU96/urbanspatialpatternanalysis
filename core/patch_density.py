# core/patch_density.py
import rasterio
import numpy as np
import pandas as pd
from skimage.measure import label
from skimage.transform import resize

def calculate_patch_density(input_height_tif, output_tif_path, threshold_min=2, grid_size=256, output_csv_path=None):
    try:
        print(f"Start processing Patch Density: {input_height_tif}, Grid size: {grid_size}, Threshold: {threshold_min}")
        
        with rasterio.open(input_height_tif) as src:
            data = src.read(1)
            # Handle NoData values, convert them to 0
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0, data)
                
            transform = src.transform
            crs = src.crs
            original_shape = data.shape
        
        # 1. Binarization: Extract patches based on threshold (e.g., areas with height > 2m are 1, others are 0)
        mask = (data >= threshold_min)
        binary_image = np.where(mask, 1, 0)
        
        # 2. Connected Component Labeling: Assign a unique ID to each independent patch
        # connectivity=2 means 8-connectivity (diagonals are also considered connected)
        labeled_image = label(binary_image, connectivity=2)

        # 3. Grid Analysis (Moving Window / Grid Analysis)
        rows, cols = original_shape
        # Calculate number of grid rows and columns
        num_rows = int(np.ceil(rows / grid_size))
        num_cols = int(np.ceil(cols / grid_size))
            
        # Initialize result matrix
        patch_density_grid = np.zeros((num_rows, num_cols), dtype=np.float32)

        print(f"Performing grid statistics, grid rows x cols: {num_rows} x {num_cols}")

        for i in range(num_rows):
            for j in range(num_cols):
                # Determine slice range for current grid
                start_row, end_row = i * grid_size, min((i + 1) * grid_size, rows)
                start_col, end_col = j * grid_size, min((j + 1) * grid_size, cols)
                
                # Slice out current grid
                grid_window = labeled_image[start_row:end_row, start_col:end_col]
                
                if grid_window.size == 0:
                    continue

                # 4. Core Algorithm (Corresponding to Fig 3 code): Count how many unique patch IDs are in the grid
                unique_patches = np.unique(grid_window)
                
                # Remove background (0 is usually background)
                count = np.count_nonzero(unique_patches)
                if 0 in unique_patches:
                    count -= 1 # Subtract background itself
                
                # If strict compliance with PD = N/A is needed, divide by area.
                # But for visualization (heatmap), directly storing quantity N is also common practice.
                # Here we store patch count N.
                patch_density_grid[i, j] = max(0, count)

        # 5. Output CSV (Optional)
        if output_csv_path:
            pd.DataFrame(patch_density_grid).to_csv(output_csv_path, index=False, header=False)

        # 6. Scale low-resolution grid results back to original image size (Nearest Neighbor interpolation, preserving grid shape)
        # So when loaded in QGIS, it looks like a grid heatmap effect
        output_array = resize(patch_density_grid, original_shape, order=0, preserve_range=True, anti_aliasing=False)

        # 7. Save results
        with rasterio.open(
            output_tif_path, 'w', driver='GTiff',
            height=original_shape[0], width=original_shape[1],
            count=1, dtype=rasterio.float32, # Use float type to store density
            crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(output_array.astype(rasterio.float32), 1)
        
        print(f"Patch Density calculation completed: {output_tif_path}")
        return True, np.mean(patch_density_grid) # Return success status and global average density

    except Exception as e:
        print(f"Error occurred while calculating Patch Density: {e}")
        import traceback
        traceback.print_exc()
        return False