# core/intersection_density.py
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box, LineString, MultiLineString
from shapely.ops import unary_union
from shapely.wkt import loads, dumps
import rasterio
from rasterio.transform import from_origin
import math

def clean_geometry_precision(geom, precision=6):
    """Helper function: Forcibly reduce geometry coordinate precision to achieve 'snapping' effect"""
    if geom is None or geom.is_empty:
        return geom
    return loads(dumps(geom, rounding_precision=precision))

def calculate_intersection_density(boundary_path, road_path, output_tif_path, grid_size=500, output_csv_path=None, missing_boundary_strategy=None):
    try:
        print(f"--- Start processing Intersection Density ---")
        print(f"Strategy: {missing_boundary_strategy}, Grid size: {grid_size} meters")
        
        # 1. Calculate unit area (km²) for density calculation
        # ID = Ni / A
        area_km2 = (grid_size / 1000.0) ** 2
        print(f"Calculation parameters: Single grid area = {area_km2:.6f} km²")
        
        # 2. Read road network
        road_network = gpd.read_file(road_path)
        
        # --- [Step 0] Coordinate Snapping ---
        # Solve the problem where road intersection ends are slightly disconnected due to errors
        print("Performing coordinate precision rounding (Snapping)...")
        road_network['geometry'] = road_network['geometry'].apply(lambda g: clean_geometry_precision(g, precision=6))
        
        # 3. Prepare boundary (Analysis Area)
        boundary_area = None
        if boundary_path and isinstance(boundary_path, str) and len(boundary_path) > 0:
            print("Using user-uploaded boundary file.")
            boundary_area = gpd.read_file(boundary_path)
            # Unify coordinate systems
            if boundary_area.crs and road_network.crs and boundary_area.crs != road_network.crs:
                road_network = road_network.to_crs(boundary_area.crs)
        else:
            print(f"No boundary provided, auto-generating based on [{missing_boundary_strategy}]...")
            # Ensure there is a base coordinate system; if the road network has no projection, default to 4326 to prevent errors
            target_crs = road_network.crs if road_network.crs else "EPSG:4326"
            
            generated_geom = None
            if missing_boundary_strategy == 'bbox':
                minx, miny, maxx, maxy = road_network.total_bounds
                generated_geom = box(minx, miny, maxx, maxy)
            elif missing_boundary_strategy == 'convex_hull':
                all_geom = road_network.geometry.unary_union
                generated_geom = all_geom.convex_hull
            elif missing_boundary_strategy == 'buffer':
                all_geom = road_network.geometry.unary_union
                # Simple estimation: if lat/lon, roughly convert buffer distance; if projected coordinates, use meters directly
                is_geo = target_crs.is_geographic if hasattr(target_crs, 'is_geographic') else True
                buffer_dist = grid_size if not is_geo else grid_size / 111320.0
                generated_geom = all_geom.buffer(buffer_dist)
            else:
                # Default bbox
                minx, miny, maxx, maxy = road_network.total_bounds
                generated_geom = box(minx, miny, maxx, maxy)
            
            boundary_area = gpd.GeoDataFrame({'geometry': [generated_geom]}, crs=target_crs)
            # Ensure road network coordinate system is consistent
            if road_network.crs != boundary_area.crs:
                road_network = road_network.to_crs(boundary_area.crs)

        # 4. Create Grid
        bounds = boundary_area.total_bounds
        xmin, ymin, xmax, ymax = bounds
        
        # --- [Lat/Lon Step Correction] ---
        # If WGS84 (Lat/Lon), need to calculate how many degrees 500m equals at current latitude
        center_lat = (ymin + ymax) / 2.0
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
        
        if boundary_area.crs.is_geographic and grid_size > 1:
            step_y = grid_size / meters_per_deg_lat
            step_x = grid_size / meters_per_deg_lon
            print(f"Grid correction (WGS84): {grid_size}m -> Lat step {step_y:.6f}°, Lon step {step_x:.6f}°")
        else:
            step_x = grid_size
            step_y = grid_size
        
        # Generate grid
        grid_cells = []
        x_range = np.arange(xmin, xmax, step_x)
        y_range = np.arange(ymin, ymax, step_y)
        
        for x in x_range:
            for y in y_range:
                grid_cells.append(box(x, y, x + step_x, y + step_y))
                
        grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=boundary_area.crs)

        # Only keep grids within the boundary (if not bbox strategy)
        if missing_boundary_strategy != 'bbox':
            grid = gpd.overlay(grid, boundary_area, how='intersection')

        # --- [Step 5] Topology Planarization ---
        # This is to ensure nodes exist at all intersections
        print("Calculating road network topological intersections (this may take some time)...")
        merged_geometry = unary_union(road_network.geometry)
        
        planar_lines = []
        if isinstance(merged_geometry, LineString):
            planar_lines = [merged_geometry]
        elif isinstance(merged_geometry, MultiLineString):
            planar_lines = list(merged_geometry.geoms)
        else:
            # Handle GeometryCollection cases
            for geom in getattr(merged_geometry, 'geoms', []):
                 if isinstance(geom, (LineString, MultiLineString)):
                    if isinstance(geom, MultiLineString):
                        planar_lines.extend(list(geom.geoms))
                    else:
                        planar_lines.append(geom)

        # --- [Step 6] Extract Intersections ---
        # Method: Count occurrences of all segment endpoints.
        # Points with count >= 3 are T-junctions or crossroads.
        # Points with count = 1 are dead ends.
        # Points with count = 2 are normal segment connection points (not intersections).
        all_coords = []
        for line in planar_lines:
            coords = list(line.coords)
            if len(coords) >= 2:
                all_coords.append(coords[0])  # Start point
                all_coords.append(coords[-1]) # End point

        df_coords = pd.DataFrame(all_coords, columns=['x', 'y'])
        # Round coordinates again to ensure matching
        df_coords['x'] = df_coords['x'].round(6)
        df_coords['y'] = df_coords['y'].round(6)

        # Count frequencies
        coord_counts = df_coords.groupby(['x', 'y']).size().reset_index(name='count')
        
        # Core criteria: Degree >= 3 indicates a physical intersection
        real_intersections_df = coord_counts[coord_counts['count'] >= 3]
        
        print(f"[Stats Result] Detected real intersections: {len(real_intersections_df)}")

        # Convert to GeoDataFrame
        geometry = [Point(xy) for xy in zip(real_intersections_df.x, real_intersections_df.y)]
        intersection_points = gpd.GeoDataFrame(real_intersections_df, geometry=geometry, crs=boundary_area.crs)
        
        # --- [Step 7] Spatial Join for Density Calculation ---
        grid['intersection_count'] = 0
        grid['density_km2'] = 0.0
        
        if len(intersection_points) > 0:
            print("Mapping intersections to grid...")
            # Count points within each grid
            intersection_gdf = gpd.sjoin(intersection_points, grid, how='inner', predicate='within')
            intersection_counts = intersection_gdf.groupby('index_right').size()
            
            # Assign values
            for idx, count in intersection_counts.items():
                if idx in grid.index:
                    grid.at[idx, 'intersection_count'] = count
                    # Core formula: Density = Count / Area (km²)
                    grid.at[idx, 'density_km2'] = count / area_km2

        # --- [Step 8] Output CSV (Includes more info) ---
        if output_csv_path:
            out_df = grid.copy()
            out_df['grid_id'] = out_df.index
            out_df['center_x'] = out_df.geometry.centroid.x
            out_df['center_y'] = out_df.geometry.centroid.y
            # Save two columns: Count and Density
            cols = [c for c in ['grid_id', 'intersection_count', 'density_km2', 'center_x', 'center_y'] if c in out_df.columns]
            out_df = out_df[cols]
            out_df.to_csv(output_csv_path, index=False)
            print(f"Detailed data (CSV) saved: {output_csv_path}")

        # --- [Step 9] Output Raster (TIF) ---
        # Note: We output Density values here, not Count values
        height = len(y_range)
        width = len(x_range)
        
        # Initialize raster matrix
        density_array = np.full((height, width), -9999.0, dtype=np.float32)
        
        for _, row in grid.iterrows():
            val = row['density_km2'] # <--- Changed to write density
            
            b = row.geometry.bounds
            col_idx = int((b[0] - xmin) / step_x)
            row_idx = int(((ymax - b[3])) / step_y)
            
            if 0 <= row_idx < height and 0 <= col_idx < width:
                density_array[row_idx, col_idx] = val

        transform = from_origin(xmin, ymax, step_x, step_y)
        
        with rasterio.open(output_tif_path, 'w', driver='GTiff',
                           height=height, width=width, count=1, dtype='float32',
                           nodata=-9999,
                           crs=boundary_area.crs.to_wkt(), transform=transform) as dst:
            dst.write(density_array, 1)

        print(f"Result Raster (TIF) saved: {output_tif_path}")
        return True, grid['density_km2'].mean() # Return average density for popup display

    except Exception as e:
        print(f"Intersection Density calculation error: {e}")
        import traceback
        traceback.print_exc()
        return False