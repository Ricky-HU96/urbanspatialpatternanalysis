# core/building_coverage_rate.py
import os
import rasterio
import numpy as np
import pandas as pd

def calculate_building_coverage_rate(input_build_tif, output_tif_path, grid_size=256, output_csv_path=None):
    try:
        print(f"Start processing Building Coverage Rate: {input_build_tif}")
        with rasterio.open(input_build_tif) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            original_shape = data.shape
            nodata_val = src.nodata

        # --- 1. Calculate Global BCR (Revised Logic) ---
        
        # Count building pixels (assuming 1-6 are buildings)
        total_building_pixels_global = np.sum((data >= 1) & (data <= 6))
        
        if nodata_val is not None and nodata_val != 0:
            # If NoData is a special value (e.g., -9999), exclude it
            valid_mask = (data != nodata_val)
            total_valid_pixels = np.sum(valid_mask)
        else:
            total_valid_pixels = data.size
            
        global_bcr = 0.0
        if total_valid_pixels > 0:
            global_bcr = total_building_pixels_global / total_valid_pixels

        print(f"Global Building Coverage Rate (Global BCR): {global_bcr:.4f}")

        # --- 2. Grid Calculation ---
        rows, cols = original_shape
        if rows < grid_size or cols < grid_size:
            grid_rows, grid_cols = 1, 1
            grids = [data]
            # In this case there is only one grid, the coordinate is the center of the image
            # transform * (col, row) -> (x, y)
            geo_x, geo_y = transform * (cols / 2.0, rows / 2.0)
            grid_info_list = [{
                'row_idx': 0, 'col_idx': 0, 
                'center_x': geo_x, 'center_y': geo_y
            }]
        else:
            grid_rows = rows // grid_size
            grid_cols = cols // grid_size
            
            grids = []
            grid_info_list = []
            
            for i in range(grid_rows):
                for j in range(grid_cols):
                    # Slice
                    chunk = data[i * grid_size:(i + 1) * grid_size, j * grid_size:(j + 1) * grid_size]
                    grids.append(chunk)
                    
                    # Calculate center coordinates
                    # Pixel coordinate center
                    center_row_px = i * grid_size + grid_size / 2.0
                    center_col_px = j * grid_size + grid_size / 2.0
                    
                    # Convert to geographic coordinates
                    geo_x, geo_y = transform * (center_col_px, center_row_px)
                    
                    grid_info_list.append({
                        'row_idx': i,
                        'col_idx': j,
                        'center_x': geo_x,
                        'center_y': geo_y
                    })

        density_indices = []
        for idx, (grid, info) in enumerate(zip(grids, grid_info_list)):
            building_pixels = np.sum((grid >= 1) & (grid <= 6))
            
            # Local grid denominator:
            # Same logic, local grids usually do not consider NoData, use grid size directly
            total_pixels_local = grid.size
            
            density = building_pixels / total_pixels_local if total_pixels_local > 0 else 0
            
            density_indices.append({
                'grid_id': idx,
                'row_index': info['row_idx'],
                'col_index': info['col_idx'],
                'center_x': info['center_x'],
                'center_y': info['center_y'],
                'building_pixels': building_pixels,
                'total_pixels': total_pixels_local,
                'building_density': density
            })
        
        if output_csv_path:
            pd.DataFrame(density_indices).to_csv(output_csv_path, index=False)
            print(f"Building coverage rate CSV result saved to: {output_csv_path}")

        # --- 3. Output TIF ---
        density_values = [item['building_density'] for item in density_indices]
        density_matrix = np.array(density_values).reshape((grid_rows, grid_cols))

        output_array = np.kron(density_matrix, np.ones((grid_size, grid_size)))
        output_array = output_array[:original_shape[0], :original_shape[1]].astype(np.float32)

        with rasterio.open(
            output_tif_path, 'w', driver='GTiff',
            height=original_shape[0], width=original_shape[1],
            count=1, dtype=output_array.dtype,
            crs=crs, transform=transform, compress='lzw') as dst:
            dst.write(output_array, 1)

        print(f"Building coverage rate raster result saved to: {output_tif_path}")
        
        # Return success status and global BCR value
        return True, global_bcr

    except Exception as e:
        print(f"Error occurred while calculating building coverage rate: {e}")
        import traceback
        traceback.print_exc()
        return False, None