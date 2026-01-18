# core/skyline_index.py
import os
import rasterio
import numpy as np
import pandas as pd

def calculate_skyline_index(input_height_tif, output_tif_path, grid_size=500, output_csv_path=None):
    try:
        print(f"Start processing Skyline Index: {input_height_tif}")
        with rasterio.open(input_height_tif) as src:
            height_data = src.read(1, masked=True) # Read masked data (nodata is not included in calculation)
            transform = src.transform
            crs = src.crs
            original_shape = height_data.shape
            
            # Get resolution (assuming pixels are square)
            pixel_size_x = abs(transform[0])
            
            # 1. Convert user input 'meters' to 'pixel count'
            grid_size_px = int(grid_size / pixel_size_x)
            if grid_size_px < 1:
                grid_size_px = 1
            
            print(f"Grid setting: {grid_size} meters => {grid_size_px} pixels (Resolution: {pixel_size_x})")

        rows, cols = original_shape
        
        # Calculate number of grid rows and columns
        n_grid_rows = rows // grid_size_px
        n_grid_cols = cols // grid_size_px
        
        # Prepare result matrix (store SI value for each grid)
        si_matrix = np.zeros((n_grid_rows, n_grid_cols), dtype=np.float32)
        
        skyline_indices = []

        # 2. Iterate through grids for calculation
        for r in range(n_grid_rows):
            for c in range(n_grid_cols):
                # Slice to extract data of the current grid
                r_start = r * grid_size_px
                r_end = (r + 1) * grid_size_px
                c_start = c * grid_size_px
                c_end = (c + 1) * grid_size_px
                
                window = height_data[r_start:r_end, c_start:c_end]
                
                # Extract valid values (exclude NoData)
                valid_pixels = window.compressed() # compressed() used for masked array
                
                si_value = 0
                mean_h = 0
                
                if valid_pixels.size > 0:
                    # --- Core formula implementation ---
                    # Figure 1 formula: Standard Deviation
                    # std = sqrt(mean(abs(x - x.mean())**2))
                    si_value = np.std(valid_pixels)
                    mean_h = np.mean(valid_pixels)
                
                si_matrix[r, c] = si_value
                
                # Record data for CSV
                # Calculate center coordinates of the grid (approximate location)
                center_x, center_y = transform * (c_start + grid_size_px/2, r_start + grid_size_px/2)
                
                skyline_indices.append({
                    'grid_id': f"{r}_{c}",
                    'center_x': center_x,
                    'center_y': center_y,
                    'std_dev_height': si_value, # Original Standard Deviation
                    'mean_height': mean_h
                })

        # 3. Normalization (Optional, but recommended to normalize to 0-1 to achieve the visual effect of Figure 3)
        # Thus, in the legend, 1 represents the area with the most drastic change in the city, and 0 represents the flattest area
        max_si = np.max(si_matrix)
        min_si = np.min(si_matrix)
        
        # Avoid division by zero
        if max_si > min_si:
            normalized_matrix = (si_matrix - min_si) / (max_si - min_si)
        else:
            normalized_matrix = si_matrix # All values are the same
            
        # Update normalized values in CSV
        if output_csv_path:
            df = pd.DataFrame(skyline_indices)
            # Add a normalized column to the CSV as well
            if max_si > min_si:
                df['skyline_index_norm'] = (df['std_dev_height'] - min_si) / (max_si - min_si)
            else:
                df['skyline_index_norm'] = 0
            df.to_csv(output_csv_path, index=False)
            print(f"Skyline Index CSV result saved to: {output_csv_path}")

        # 4. Resample back to original image size (generate mosaic effect)
        # Expand
        expanded_si = np.kron(normalized_matrix, np.ones((grid_size_px, grid_size_px)))
        
        # Create final output array, size consistent with original tif, initialized to 0
        final_output = np.zeros((rows, cols), dtype=np.float32)
        
        # Fill in the expanded data; parts at the edges smaller than one grid naturally remain 0
        actual_r, actual_c = expanded_si.shape
        final_output[:actual_r, :actual_c] = expanded_si

        # Write to file
        with rasterio.open(
            output_tif_path, 'w', driver='GTiff',
            height=rows, width=cols,
            count=1, dtype=np.float32,
            crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(final_output, 1)

        print(f"Skyline Index raster result saved to: {output_tif_path}")
        return True
    except Exception as e:
        print(f"Error occurred while calculating Skyline Index: {e}")
        import traceback
        traceback.print_exc()
        return False