# core/road_density.py
import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_origin
import numpy as np

def calculate_road_length_in_cell(cell, road_network_sindex, road_network):
    # 1. Spatial index rough filtering
    possible_matches_index = list(road_network_sindex.intersection(cell.bounds))
    if not possible_matches_index:
        return 0.0
    
    intersecting_roads = road_network.iloc[possible_matches_index]
    
    # 2. Geometry clipping
    try:
        # intersects check is faster than intersection calculation, check first
        mask = intersecting_roads.intersects(cell)
        candidates = intersecting_roads[mask]
        
        if candidates.empty:
            return 0.0
            
        # 3. Precise length calculation
        # intersection might produce Point or MultiLineString, must handle rigorously
        clipped = candidates.intersection(cell)
        
        # The logic here is to exclude "points just touching the edge" or "invalid geometries"
        length = 0.0
        for geom in clipped:
            if not geom.is_empty and geom.geom_type in ['LineString', 'MultiLineString']:
                length += geom.length
                
        return length
    except Exception:
        return 0.0

def calculate_road_density(boundary_path, road_path, output_tif_path, grid_size=500, output_csv_path=None, missing_boundary_strategy="bbox"):
    try:
        print(f"--- Start processing Road Density: Grid {grid_size} meters ---")
        
        # 1. Read road network
        road_network = gpd.read_file(road_path)
        original_count = len(road_network)
        
        # [New] Geometry validity check and repair
        # Invalid geometries will cause intersection to return empty, resulting in "roads existing but appearing blank"
        road_network = road_network[road_network.is_valid] 
        # Filter out lines with length 0
        road_network = road_network[road_network.geometry.length > 0]
        
        if len(road_network) < original_count:
            print(f"Tip: Automatically removed {original_count - len(road_network)} invalid or zero-length line features.")

        # 2. Intelligent coordinate system handling
        if not road_network.crs.is_projected:
            print(f"Geographic coordinate system detected ({road_network.crs.name}), converting to projected coordinate system...")
            try:
                target_crs = road_network.estimate_utm_crs()
                road_network = road_network.to_crs(target_crs)
                print(f"Converted to: {target_crs.name} (EPSG:{target_crs.to_epsg()})")
            except:
                print("Auto-projection failed, forcing Web Mercator (EPSG:3857)")
                road_network = road_network.to_crs("EPSG:3857")

        # 3. Boundary processing (Generate Grid extent)
        boundary_gdf = None
        if boundary_path and os.path.exists(boundary_path):
            boundary_gdf = gpd.read_file(boundary_path)
            if boundary_gdf.crs != road_network.crs:
                boundary_gdf = boundary_gdf.to_crs(road_network.crs)
        else:
            all_roads_geom = road_network.geometry.unary_union
            if missing_boundary_strategy == "convex_hull":
                boundary_gdf = gpd.GeoDataFrame({'geometry': [all_roads_geom.convex_hull]}, crs=road_network.crs)
            elif missing_boundary_strategy == "buffer":
                boundary_gdf = gpd.GeoDataFrame({'geometry': [all_roads_geom.buffer(grid_size)]}, crs=road_network.crs)
            else: 
                minx, miny, maxx, maxy = road_network.total_bounds
                boundary_gdf = gpd.GeoDataFrame({'geometry': [box(minx, miny, maxx, maxy)]}, crs=road_network.crs)

        # 4. Grid generation (Key: Align with Rasterio Transform)
        xmin, ymin, xmax, ymax = boundary_gdf.total_bounds
        
        # Calculate rows and columns
        cols = int(np.ceil((xmax - xmin) / grid_size))
        rows = int(np.ceil((ymax - ymin) / grid_size))
        
        print(f"Grid matrix dimensions: {cols} cols x {rows} rows")
        
        # Pre-define Transform (Determines how pixels map to lat/lon)
        transform = from_origin(xmin, ymax, grid_size, grid_size)
        
        # Generate vector grid for spatial calculation
        cells_data = []
        for j in range(rows): # Row index (0 is top)
            y_top = ymax - j * grid_size
            y_bottom = ymax - (j + 1) * grid_size
            
            for i in range(cols): # Col index (0 is left)
                x_left = xmin + i * grid_size
                x_right = xmin + (i + 1) * grid_size
                
                poly = box(x_left, y_bottom, x_right, y_top)
                cells_data.append({
                    'geometry': poly,
                    'row': j,  # Record which row this cell belongs to
                    'col': i   # Record which column this cell belongs to
                })

        grid = gpd.GeoDataFrame(cells_data, crs=road_network.crs)
        
        # 5. Spatial filtering (Clip)
        grid['centroid'] = grid.geometry.centroid
        centers = grid.copy()
        centers.set_geometry('centroid', inplace=True)
        centers = centers.drop(columns=['geometry'])
        centers.rename_geometry('geometry', inplace=True)
        
        joined = gpd.sjoin(centers, boundary_gdf, how='inner', predicate='intersects')
        valid_indices = joined.index
        grid = grid.loc[valid_indices].copy()
        
        print(f"Valid grids to calculate: {len(grid)}")

        # 6. Calculate density (Time-consuming step)
        # Rebuild index to ensure query accuracy
        road_network_sindex = road_network.sindex 
        
        road_lengths = []
        total_cells = len(grid)
        log_step = max(1, total_cells // 10)
        
        count = 0
        for idx, row in grid.iterrows():
            length = calculate_road_length_in_cell(row.geometry, road_network_sindex, road_network)
            road_lengths.append(length)
            
            count += 1
            if count % log_step == 0:
                print(f"Progress: {int((count/total_cells)*100)}%")
        
        grid['road_length'] = road_lengths
        
        # Calculate density value (Unit: km/km2)
        cell_area_km2 = (grid_size / 1000) ** 2
        # Convert length from meters to kilometers
        grid['road_density'] = (grid['road_length'] / 1000) / cell_area_km2

        # 7. Fill raster matrix (Key correction step)
        # Initialize to -1.0, not 0
        density_array = np.full((rows, cols), -1.0, dtype=np.float32)
        
        for idx, row_data in grid.iterrows():
            r = row_data['row']
            c = row_data['col']
            val = row_data['road_density']
            
            if 0 <= r < rows and 0 <= c < cols:
                density_array[r, c] = val

        # Export CSV
        if output_csv_path:
            out_df = grid.drop(columns=['centroid', 'geometry'])
            out_df.to_csv(output_csv_path, index=False)

        # 8. Write GeoTIFF
        # Set nodata to -1.0
        with rasterio.open(output_tif_path, 'w', driver='GTiff', 
                           height=rows, width=cols, count=1,
                           dtype=density_array.dtype, crs=road_network.crs.to_wkt(), 
                           transform=transform, 
                           nodata=-1.0) as dst: 
            dst.write(density_array, 1)

        print(f"Success! Result saved to: {output_tif_path}")
        return True

    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()
        return False