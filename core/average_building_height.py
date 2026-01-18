# core/average_building_height.py
import os
import rasterio
import numpy as np
import pandas as pd

# Modification 1: Changed threshold_min default from 80 to 0 (any height is considered a building)
def calculate_average_building_height(input_height_tif, output_tif_path, threshold_min=0, grid_size=256, output_csv_path=None):
    try:
        print(f"Start processing average building height: {input_height_tif}")
        print(f"Calculation parameters - Grid pixel size: {grid_size}, Height threshold: {threshold_min}")
        
        with rasterio.open(input_height_tif) as src:
            data = src.read(1)
            # Handle NoData values, convert them to 0 to prevent interference with calculation
            if src.nodata is not None:
                data[data == src.nodata] = 0
            
            transform = src.transform
            crs = src.crs
            original_shape = data.shape

        rows, cols = original_shape
        
        # Safety check: If calculated grid_size is less than 1 (e.g., when resolution is very coarse), force it to 1
        if grid_size < 1:
            grid_size = 1

        # Handle cases where image size is smaller than grid_size
        if rows < grid_size or cols < grid_size:
            print(f"Warning: Image size ({rows}x{cols}) is smaller than grid size ({grid_size}), the entire image will be used for calculation.")
            grid_rows, grid_cols = 1, 1
            grids = [data]
        else:
            grid_rows, grid_cols = rows // grid_size, cols // grid_size
            grids = [data[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
                     for i in range(grid_rows) for j in range(grid_cols)]

        average_height_indices = []
        for idx, grid in enumerate(grids):
            # Modification 2: Logic optimization
            # Filter all pixels within the grid with height > threshold_min
            building_pixels = grid[grid > threshold_min]
            
            # Calculate the average value of these building pixels
            # If there are no buildings in the grid (size==0), set average height to 0
            if building_pixels.size > 0:
                average_height = building_pixels.mean()
            else:
                average_height = 0
                
            average_height_indices.append({'grid_id': idx, 'average_height': average_height})

        if output_csv_path:
            pd.DataFrame(average_height_indices).to_csv(output_csv_path, index=False)
            print(f"Average building height CSV result saved to: {output_csv_path}")

        height_values = [item['average_height'] for item in average_height_indices]
        height_matrix = np.array(height_values).reshape((grid_rows, grid_cols))
        
        # Use np.kron for efficient upscaling (generate mosaic effect)
        output_array = np.kron(height_matrix, np.ones((grid_size, grid_size)))
        # Crop to original size to handle non-perfect divisions
        output_array = output_array[:original_shape[0], :original_shape[1]].astype(np.float32)

        with rasterio.open(
            output_tif_path, 'w', driver='GTiff',
            height=original_shape[0], width=original_shape[1],
            count=1, dtype=output_array.dtype,
            crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(output_array, 1)
            
        print(f"Average building height raster result saved to: {output_tif_path}")
        return True
    except Exception as e:
        print(f"Error occurred while calculating average building height: {e}")
        import traceback
        traceback.print_exc()
        return False