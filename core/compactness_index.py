# core/compactness_index.py
import rasterio
from rasterio.windows import Window
from rasterio import features
import numpy as np
import pandas as pd
from shapely.geometry import shape, MultiPolygon

def calculate_compactness(input_build_tif, output_tif_path, grid_size=256, output_csv_path=None):
    try:
        print(f"Start processing Compactness Index (CI): {input_build_tif}")
        
        with rasterio.open(input_build_tif) as src:
            # Get basic information
            meta = src.meta.copy()
            width = src.width
            height = src.height
            transform = src.transform
            
            # Calculate the number of grids horizontally and vertically
            n_cols = (width + grid_size - 1) // grid_size
            n_rows = (height + grid_size - 1) // grid_size
            
            # Initialize result matrix (one value per grid)
            result_matrix = np.zeros((n_rows, n_cols), dtype=np.float32)
            
            compactness_stats = []

            # Iterate through each grid
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    # 1. Define the window range of the current grid
                    window_col_off = col_idx * grid_size
                    window_row_off = row_idx * grid_size
                    # Handle edge cases to prevent out of bounds
                    w = min(grid_size, width - window_col_off)
                    h = min(grid_size, height - window_row_off)
                    
                    window = Window(window_col_off, window_row_off, w, h)
                    
                    # 2. Read data within the window
                    data = src.read(1, window=window)
                    
                    # 3. Get the local geotransform parameters for the window (Key step: Convert pixels to geographic coordinates)
                    window_transform = src.window_transform(window)
                    
                    # 4. Generate binary mask (Assuming values > 0 are buildings)
                    # Can be adjusted based on specific data, e.g., mask = np.isin(data, [1,2,3...])
                    mask = (data > 0).astype(np.uint8)
                    
                    compactness = 0.0
                    area = 0.0
                    perimeter = 0.0
                    
                    if np.any(mask): # If there are buildings in the grid
                        # 5. Raster to Polygon - Get polygons with geographic coordinates
                        # rasterio.features.shapes returns a generator in GeoJSON style
                        shapes_gen = features.shapes(mask, transform=window_transform, mask=(mask==1))
                        
                        polygons = []
                        for geom, val in shapes_gen:
                            s = shape(geom) # Convert to Shapely object
                            if s.is_valid and not s.is_empty:
                                polygons.append(s)
                        
                        if polygons:
                            # Merge all buildings in the grid into a MultiPolygon
                            multi_poly = MultiPolygon(polygons)
                            
                            # 6. Calculate area and perimeter (Unit depends on projection, usually meters/square meters)
                            area = multi_poly.area
                            perimeter = multi_poly.length
                            
                            # 7. Calculate Compactness Index (Corresponding to Formula in PPT Figure 2)
                            # CI = 2 * sqrt(pi * A) / P
                            if perimeter > 0:
                                compactness = (2 * np.sqrt(np.pi * area)) / perimeter
                    
                    # Store in matrix
                    result_matrix[row_idx, col_idx] = compactness
                    
                    # Store in statistics list
                    compactness_stats.append({
                        'grid_id': f"{row_idx}_{col_idx}",
                        'row': row_idx,
                        'col': col_idx,
                        'compactness': compactness,
                        'area': area,
                        'perimeter': perimeter,
                        # Calculate grid center coordinates for display in GIS via CSV
                        'center_x': window_transform[2] + (w * window_transform[0] / 2),
                        'center_y': window_transform[5] + (h * window_transform[4] / 2)
                    })

        # 8. Save CSV (if needed)
        if output_csv_path:
            df = pd.DataFrame(compactness_stats)
            df.to_csv(output_csv_path, index=False)
            print(f"CSV statistics saved: {output_csv_path}")

        # 9. Generate visualization result raster
        # Upscale result matrix (n_rows x n_cols) back to original image size
        # Use Kronecker product for nearest neighbor upscaling
        full_res_matrix = np.kron(result_matrix, np.ones((grid_size, grid_size)))
        # Crop to original size
        full_res_matrix = full_res_matrix[:height, :width].astype(np.float32)

        # Write to output TIF
        meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': transform,
            'dtype': 'float32',
            'count': 1,
            'compress': 'lzw'
        })

        with rasterio.open(output_tif_path, 'w', **meta) as dst:
            dst.write(full_res_matrix, 1)

        print(f"Compactness raster saved: {output_tif_path}")
        return True, np.mean(result_matrix) # Return success status and global average

    except Exception as e:
        print(f"Error occurred while calculating Compactness Index: {e}")
        import traceback
        traceback.print_exc()
        return False