# -*- coding: utf-8 -*-
import os
import pandas as pd
import tempfile
import matplotlib
# Ensure backend is set before importing pyplot
matplotlib.use('Agg')

from qgis.PyQt import QtWidgets
from qgis.PyQt.QtWidgets import QDialog, QMessageBox, QInputDialog
from qgis.core import (
    Qgis, QgsVectorLayer, QgsRasterLayer, QgsProject,
    QgsColorRampShader, QgsSingleBandPseudoColorRenderer, 
    QgsStyle, QgsRasterShader, QgsRasterBandStats
)
from qgis.gui import QgsFileWidget
from osgeo import gdal, ogr, osr

from .ui_urbanspatialpatternanalysis import Ui_UrbanSpatialPatternAnalysisDialogBase
from .core import (
    building_coverage_rate, compactness_index, average_building_height, 
    edge_density, intersection_density, patch_density, 
    road_density, shape_index, skyline_index
)

class UrbanSpatialPatternAnalysisDialog(QDialog, Ui_UrbanSpatialPatternAnalysisDialogBase):
    
    def __init__(self, iface, parent=None):
        super(UrbanSpatialPatternAnalysisDialog, self).__init__(parent)
        self.setupUi(self)
        self.iface = iface
        self.configure_widgets()
        self.populate_indicators()
        self.connect_signals()
        self.update_ui_for_indicator()
        
        # Temporary file list, used for cleanup when the plugin closes
        self.temp_files = []

    def configure_widgets(self):
        """Configure all file input/output widgets."""
        self.outputFileWidget.setStorageMode(QgsFileWidget.SaveFile)
        self.outputFileWidget.setFilter("GeoTIFF files (*.tif *.TIF)")
        self.outputFileWidget.setDefaultRoot(os.path.expanduser("~"))

        # Relax input filtering to support all files (Raster/Vector)
        all_files_filter = "All Files (*)"
        self.buildRasterFileWidget.setFilter(all_files_filter)
        self.heightRasterFileWidget.setFilter(all_files_filter)
        self.roadNetworkFileWidget.setFilter(all_files_filter)
        self.analysisAreaFileWidget.setFilter(all_files_filter)

    def connect_signals(self):
        """Centralize management of all signal and slot connections."""
        self.indicatorComboBox.currentIndexChanged.connect(self.update_ui_for_indicator)
        self.button_box.accepted.connect(self.run_analysis)
        self.button_box.rejected.connect(self.reject)

    def populate_indicators(self):
        """Populate the indicator selection dropdown menu"""
        self.indicatorComboBox.clear()
        self.indicatorComboBox.addItem("1. Building Coverage Ratio (BCR)", "bcr")
        self.indicatorComboBox.addItem("2. Compactness Index (CI)", "ci")
        self.indicatorComboBox.addItem("3. Plaque Density (PD)", "pd")
        self.indicatorComboBox.addItem("4. Shape Index (SI)", "si")
        self.indicatorComboBox.addItem("5. Edge Density (ED)", "ed")
        self.indicatorComboBox.addItem("6. Road density (RD)", "road_density")
        self.indicatorComboBox.addItem("7. Intersection Density (ID)", "intersection_density")
        self.indicatorComboBox.addItem("8. Skyline Index (SLI)", "skyline")
        self.indicatorComboBox.addItem("9. Average building height (ABH)", "avg_height")

    def update_ui_for_indicator(self):
        """Show/hide corresponding input boxes and output options based on the selected indicator."""
        indicator_id = self.indicatorComboBox.currentData()
        self.buildRasterGroup.setVisible(False)
        self.heightRasterGroup.setVisible(False)
        self.vectorGroup.setVisible(False)
        
        self.pngOutputCheckBox.setVisible(indicator_id == "si")

        build_raster_indicators = ["bcr", "ci"]
        height_raster_indicators = ["pd", "si", "ed", "skyline", "avg_height"]
        vector_indicators = ["road_density", "intersection_density"]
        
        if indicator_id in build_raster_indicators:
            self.buildRasterGroup.setVisible(True)
            self.label_2.setText("Architectural Input (Raster or Vector):")
        elif indicator_id in height_raster_indicators:
            self.heightRasterGroup.setVisible(True)
            self.label_3.setText("Height Input (Raster or Vector):")
        elif indicator_id in vector_indicators:
            self.vectorGroup.setVisible(True)
            
            # [Modification Start] -----------------------------------------------
            # Show special prompt for Intersection Density, not affecting Road Density
            if indicator_id == "road_density":
                self.label_5.setText("Analysis Area (Optional - Auto-detect)")
            elif indicator_id == "intersection_density":
                # New logic: Intersection Density is also marked as optional
                self.label_5.setText("Analysis Area (Optional - Auto-detect)")
            else:
                self.label_5.setText("Analysis Area (analysisAreaFileWidget)")
            # [Modification End] -----------------------------------------------

    def convert_vector_to_raster(self, vector_path, burn_attribute=None, resolution=None):
        """
        Helper function: Convert vector file to temporary raster file.
        """
        try:
            self.iface.messageBar().pushMessage("Converting", f"Rasterizing vector: {os.path.basename(vector_path)}", level=Qgis.Info)
            
            vec_ds = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
            if vec_ds is None:
                raise Exception("Cannot open vector file")
            layer = vec_ds.GetLayer()
            spatial_ref = layer.GetSpatialRef()
            
            x_min, x_max, y_min, y_max = layer.GetExtent()
            
            if resolution is None:
                is_geographic = spatial_ref.IsGeographic() if spatial_ref else False
                if is_geographic:
                    resolution = 0.0001
                else:
                    resolution = 10.0

            cols = int((x_max - x_min) / resolution)
            rows = int((y_max - y_min) / resolution)
            
            MAX_PIXELS = 1600000000 
            
            if cols * rows > MAX_PIXELS: 
                QMessageBox.warning(self, "Huge Data Volume Warning", 
                                    f"The resolution you selected will result in an extremely huge image ({cols}x{rows} pixels).\nThe plugin will automatically reduce the resolution to prevent crashes.")
                while cols * rows > MAX_PIXELS:
                    resolution *= 2 
                    cols = int((x_max - x_min) / resolution)
                    rows = int((y_max - y_min) / resolution)
                print(f"Final adjusted safe resolution: {resolution}")

            temp_tif = tempfile.NamedTemporaryFile(suffix='.tif', delete=False).name
            self.temp_files.append(temp_tif)

            options = gdal.RasterizeOptions(
                xRes=resolution, 
                yRes=resolution, 
                outputBounds=[x_min, y_min, x_max, y_max],
                noData=0,
                initValues=0,
                outputType=gdal.GDT_Float32,
                attribute=burn_attribute if burn_attribute else None,
                burnValues=[1] if not burn_attribute else None
            )
            
            gdal.Rasterize(temp_tif, vector_path, options=options)
            
            return temp_tif, resolution
        except Exception as e:
            print("Rasterization Error Traceback:")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Conversion Failed", f"Vector to raster conversion failed: {str(e)}")
            return None, 0

    def find_height_field(self, vector_path):
        """Attempt to automatically find the height field; if not found, ask the user to input."""
        layer = QgsVectorLayer(vector_path, "temp", "ogr")
        if not layer.isValid():
            return None
        
        fields = [f.name() for f in layer.fields()]
        candidates = ['height', 'Height', 'HEIGHT', 'z', 'Z', 'elevation', 'Elev', 'floor', 'Floor']
        
        for cand in candidates:
            if cand in fields:
                return cand
        
        item, ok = QInputDialog.getItem(self, "Select Height Field", 
                                        f"Common height fields were not detected in file {os.path.basename(vector_path)}.\nPlease select the field representing building height:", 
                                        fields, 0, False)
        if ok and item:
            return item
        return None

    def preprocess_input(self, file_path, input_type='binary'):
        """
        Preprocess input files.
        """
        if not file_path or not os.path.exists(file_path):
            return file_path, 0

        is_vector = False
        try:
            ds = gdal.OpenEx(file_path, gdal.OF_VECTOR)
            if ds is not None and ds.GetLayerCount() > 0:
                is_vector = True
            
            if file_path.lower().endswith(('.shp', '.geojson', '.kml', '.gpkg')):
                is_vector = True
            elif file_path.lower().endswith(('.tif', '.tiff', '.img', '.dat')):
                is_vector = False
        except:
            pass

        if is_vector:
            burn_field = None
            if input_type == 'height':
                burn_field = self.find_height_field(file_path)
                if not burn_field:
                    QMessageBox.warning(self, "Warning", "No height field selected. Default height will be used for calculation.")
            
            layer = QgsVectorLayer(file_path, "temp", "ogr")
            crs = layer.crs()
            is_geo = crs.isGeographic()
            
            if is_geo:
                default_res = 0.0001
                unit_label = "Degrees"
                decimals = 6 
            else:
                default_res = 10.0
                unit_label = "Meters"
                decimals = 2

            user_res, ok = QInputDialog.getDouble(
                self, 
                "Set Rasterization Resolution", 
                f"Vector data detected.\nTo ensure accuracy of shape index analysis, please set raster size:\n(Unit: {unit_label})\n\nSuggestion: 5-10 meters for urban building analysis", 
                default_res, 
                0.0000001, 
                10000.0,   
                decimals
            )
            
            if not ok:
                user_res = default_res
                
            raster_path, res = self.convert_vector_to_raster(file_path, burn_attribute=burn_field, resolution=user_res)
            
            if raster_path:
                return raster_path, res
            else:
                return file_path, 0
        else:
            current_res = 0
            try:
                ds = gdal.Open(file_path)
                if ds:
                    gt = ds.GetGeoTransform()
                    current_res = abs(gt[1]) 
            except:
                pass
            return file_path, current_res

    def run_analysis(self):
        """Main logic executed after clicking the 'Run' button."""
        indicator_id = self.indicatorComboBox.currentData()
    
        raw_build_path = self.buildRasterFileWidget.filePath() 
        raw_height_path = self.heightRasterFileWidget.filePath()
        road_shp_path = self.roadNetworkFileWidget.filePath()
        boundary_shp_path = self.analysisAreaFileWidget.filePath() 
        
        output_tif_path = self.outputFileWidget.filePath()
        should_output_csv = self.csvOutputCheckBox.isChecked()

        # [Fix] For Shape Index, do not rely on isVisible() (may be False due to parent/layout visibility)
        should_output_png = self.pngOutputCheckBox.isChecked() if indicator_id == "si" else (self.pngOutputCheckBox.isChecked() and self.pngOutputCheckBox.isVisible())

        if not output_tif_path:
            QMessageBox.critical(self, "Error", "You must specify an output raster (.tif) file path!")
            return
        
        if not output_tif_path.lower().endswith(('.tif', '.tiff')):
            output_tif_path += '.tif'

        base_path, _ = os.path.splitext(output_tif_path)
        csv_output_path = base_path + ".csv" if should_output_csv else None
        png_output_path = base_path + ".png" if should_output_png else None
        
        # --- [Road Density Logic: Keep as is, Unmodified] ---
        missing_boundary_strategy = "bbox" # Default value
        
        if indicator_id == "road_density":
            if not road_shp_path:
                 QMessageBox.warning(self, "Missing Input", "Please select the road network file!")
                 return
            
            # If no boundary file is selected, pop up to ask
            if not boundary_shp_path or not os.path.exists(boundary_shp_path):
                strategies = [
                    "Scheme 1: Rectangular Box (Fastest)", 
                    "Scheme 2: Convex Hull (Tight Shape)",
                    "Scheme 3: Buffer (Roads + Margin)"
                ]
                item, ok = QInputDialog.getItem(
                    self, 
                    "Missing Analysis Area", 
                    "No boundary file detected. Please select a data processing strategy:\n(No boundary file detected. How should the area be defined?)", 
                    strategies, 
                    0, 
                    False
                )
                if ok and item:
                    if "Rectangular" in item: missing_boundary_strategy = "bbox"
                    elif "Convex" in item: missing_boundary_strategy = "convex_hull"
                    elif "Buffer" in item: missing_boundary_strategy = "buffer"
                else:
                    return # User cancelled, stop execution
            else:
                missing_boundary_strategy = None # User provided file, no strategy needed

        # --- [Modification Start] Intersection Density Logic (Newly added, does not interfere with above) ---
        if indicator_id == "intersection_density":
            if not road_shp_path:
                 QMessageBox.warning(self, "Missing Input", "Please select the road network file!")
                 return
            
            # If no boundary file is selected, pop up to ask
            if not boundary_shp_path or not os.path.exists(boundary_shp_path):
                strategies = [
                    "Scheme 1: Rectangular Box (Fastest)", 
                    "Scheme 2: Convex Hull (Tight Shape)",
                    "Scheme 3: Buffer (Roads + Margin)"
                ]
                item, ok = QInputDialog.getItem(
                    self, 
                    "Missing Analysis Area (Intersection Density)", 
                    "No boundary file detected. Please select a data processing strategy:", 
                    strategies, 
                    0, 
                    False
                )
                if ok and item:
                    if "Rectangular" in item: missing_boundary_strategy = "bbox"
                    elif "Convex" in item: missing_boundary_strategy = "convex_hull"
                    elif "Buffer" in item: missing_boundary_strategy = "buffer"
                else:
                    return # User cancelled
            else:
                missing_boundary_strategy = None 
        # --- [Modification End] ---

        # --- Grid Size Parameter Input (Unified to METERS) ---
        DEFAULT_GRID_SIZE = 500  
        target_grid_size = DEFAULT_GRID_SIZE
        
        # [Modify]: Add bcr, ci, pd to the list requiring grid parameters
        indicators_requiring_grid = ["bcr", "ci", "pd", "ed", "road_density", "skyline", "avg_height", "intersection_density"]
        
        if indicator_id in indicators_requiring_grid:
            # [关键修改] 统一提示用户输入“米”，无论底层数据是什么单位
            # 这样你只需要输入 500，代码会在后面帮你算成像素
            prompt_label = "Set Analysis Grid Size (Meters):"
            
            input_grid, ok = QInputDialog.getInt(
                self, 
                "Grid Size Setting", 
                f"{prompt_label}\n(This determines the resolution of the result.\nWe will auto-convert to pixels if needed.)", 
                value=500,  
                min=10,    
                max=10000, 
                step=100
            )
            if ok:
                target_grid_size = input_grid
            else:
                return

        kwargs = {}
        if csv_output_path:
            kwargs['output_csv_path'] = csv_output_path
        if png_output_path:
            kwargs['output_png_path'] = png_output_path

        try:
            self.iface.messageBar().pushMessage("Processing...", f"Calculating {self.indicatorComboBox.currentText()}", level=Qgis.Info, duration=5)

            real_build_path = raw_build_path
            real_height_path = raw_height_path
            
            used_resolution = 0
            is_geographic = False # [新增] 标记是否为地理坐标系(度)

            build_raster_indicators = ["bcr", "ci"]
            height_raster_indicators = ["pd", "si", "ed", "skyline", "avg_height"]

            # --- Preprocessing & Resolution Detection ---
            if indicator_id in build_raster_indicators and raw_build_path:
                real_build_path, res = self.preprocess_input(raw_build_path, input_type='binary')
                if res > 0: used_resolution = res
                # [新增] 检查坐标系
                try:
                    ds = gdal.Open(real_build_path)
                    if ds:
                        prj = ds.GetProjection()
                        srs = osr.SpatialReference(wkt=prj)
                        if srs.IsGeographic(): is_geographic = True
                except: pass
            
            if indicator_id in height_raster_indicators and raw_height_path:
                real_height_path, res = self.preprocess_input(raw_height_path, input_type='height')
                if res > 0: used_resolution = res
                # [新增] 检查坐标系
                try:
                    ds = gdal.Open(real_height_path)
                    if ds:
                        prj = ds.GetProjection()
                        srs = osr.SpatialReference(wkt=prj)
                        if srs.IsGeographic(): is_geographic = True
                except: pass

            # --- [关键修改] 智能计算像素数 (Calculated Pixels) ---
            # 目标：将用户输入的 target_grid_size (米) 转换为 核心算法需要的 grid_size (像素)
            calculated_pixels = 256 # 默认备用值
            
            if used_resolution > 0:
                if is_geographic:
                    # 如果是度：1度 ≈ 111320米 (赤道估算)
                    # 算法：(目标米数 / 111320) / 分辨率度数
                    degree_equivalent = target_grid_size / 111320.0
                    calculated_pixels = int(degree_equivalent / used_resolution)
                else:
                    # 如果是米：直接除
                    calculated_pixels = int(target_grid_size / used_resolution)
                
                # 最小保护
                if calculated_pixels < 1: calculated_pixels = 1
            else:
                # 如果没读取到分辨率，就假设用户输入的就是像素
                calculated_pixels = target_grid_size
            
            # Avg Height 之前有自己单独的逻辑，现在统一用 calculated_pixels
            avg_height_grid_pixels = calculated_pixels

            success = False
            global_result_msg = "" 
            
            # [Modification] Updated analysis_map to use calculated_pixels where appropriate
            # 规则：
            # 1. BCR, CI, PD, ED, AvgHeight 核心代码接收像素 -> 传 calculated_pixels
            # 2. Skyline, Road, Intersection 核心代码接收米 -> 传 target_grid_size
            
            analysis_map = {
                "bcr": (building_coverage_rate.calculate_building_coverage_rate, {
                    'input_build_tif': real_build_path,
                    'grid_size': calculated_pixels # <--- 传计算后的像素
                }),
                "ci": (compactness_index.calculate_compactness, {
                    'input_build_tif': real_build_path,
                    'grid_size': calculated_pixels # <--- 传计算后的像素
                }),
                "pd": (patch_density.calculate_patch_density, {
                    'input_height_tif': real_height_path, 
                    'grid_size': calculated_pixels, # <--- 传计算后的像素
                    'threshold_min': 2
                }),
                "si": (shape_index.calculate_shape_index, {
                    'input_height_tif': real_height_path,
                    'threshold_min': 1
                }),
                "ed": (edge_density.calculate_edge_density, {
                    'input_height_tif': real_height_path,
                    'threshold_min': 2, 
                    'grid_size': calculated_pixels # <--- 传计算后的像素
                }),
                # [关键修改] Skyline 现在使用 calculated_pixels (传入参数名为 grid_size_px)
                # 这解决了 WGS84 坐标系下用“500”除以“度”导致的内存崩溃问题
                "skyline": (skyline_index.calculate_skyline_index, {
                    'input_height_tif': real_height_path,
                    'grid_size_px': calculated_pixels 
                }),
                "avg_height": (average_building_height.calculate_average_building_height, {
                    'input_height_tif': real_height_path,
                    'grid_size': calculated_pixels, # <--- 传计算后的像素
                    'threshold_min': 0.1 
                }),
                
                # road_density Keep as is
                "road_density": (road_density.calculate_road_density, {
                    'boundary_path': boundary_shp_path if boundary_shp_path else None, 
                    'road_path': road_shp_path, 
                    'grid_size': target_grid_size, # Road 接收米，保持不变
                    'missing_boundary_strategy': missing_boundary_strategy
                }),
                
                # intersection_density Keep as is
                "intersection_density": (intersection_density.calculate_intersection_density, {
                    'boundary_path': boundary_shp_path if boundary_shp_path else None, 
                    'road_path': road_shp_path, 
                    'output_tif_path': output_tif_path,
                    'grid_size': target_grid_size, # Intersection 接收米，保持不变
                    'missing_boundary_strategy': missing_boundary_strategy
                })
            }

            if indicator_id in analysis_map:
                func, params = analysis_map[indicator_id]
                
                # --- [Modification Start] ---------------------------------
                if indicator_id == "road_density":
                    # Road Density logic remains unchanged
                    if not params.get('road_path'):
                        QMessageBox.warning(self, "Input Error", "Road Density must provide a road network file!")
                        return
                
                elif indicator_id == "intersection_density":
                    # New Intersection Density specific check: allow boundary to be None
                    if not params.get('road_path'):
                         QMessageBox.warning(self, "Input Error", "Intersection Density must provide a road network file!")
                         return
                
                else:
                    # All other functions: keep original strict check
                    if any(not val for val in params.values()):
                        QMessageBox.warning(self, "Input Error", "Please provide all required input files for the current function!")
                        return
                # --- [Modification End] ---------------------------------
                
                # Merge parameters
                all_params = {**params}
                if 'output_tif_path' not in all_params:
                    all_params['output_tif_path'] = output_tif_path
                all_params.update(kwargs)
                
                result = func(**all_params)
                
                if isinstance(result, tuple):
                    success = result[0]
                    if len(result) > 1 and result[1] is not None:
                        val = result[1]
                        if isinstance(val, float):
                            global_result_msg = f"\n\n[Global Statistical Result]\nGlobal Value: {val:.4f}"
                        else:
                            global_result_msg = f"\n\n[Global Statistical Result]\n{val}"
                else:
                    success = result
            
            if success:
                message = f'Result successfully saved to:\n{output_tif_path}'
                
                if used_resolution > 0:
                    unit_label = "Degrees" if is_geographic else "Meters"
                    
                    # [修改] 结果信息展示逻辑，显示真实的物理距离
                    # 既然现在 target_grid_size 就是用户输入的“米”，我们直接展示它
                    actual_meters_approx = target_grid_size 
                    
                    if indicator_id in ["bcr", "ci", "pd", "ed", "avg_height", "skyline"]:
                        # 这些指标使用了 calculated_pixels，展示一下换算结果
                        real_grid_str = f"{actual_meters_approx} Meters (converted to ~{calculated_pixels} pixels)"
                    else:
                        real_grid_str = f"{actual_meters_approx} Meters"

                    grid_info = (
                        f"\n\n----------------------------\n"
                        f"[Spatial Parameter Information]\n"
                        f"1. Input Unit: {unit_label}\n"
                        f"2. Original Resolution: {used_resolution:.6f}\n"
                        f"3. Analysis Grid: {real_grid_str}\n"
                        f"----------------------------"
                    )
                    message += grid_info
                elif indicator_id == "road_density" or indicator_id == "intersection_density":
                    message += (
                        f"\n\n----------------------------\n"
                        f"[Spatial Parameter Information]\n"
                        f"Analysis Grid Size: {target_grid_size} (Unit depends on projection)\n"
                        f"----------------------------"
                    )

                if csv_output_path:
                    message += f'\nDetailed data saved to:\n{csv_output_path}'
                
                message += global_result_msg
                
                reply = QMessageBox.question(self, 'Calculation Complete', message + '\n\nLoad layer?', 
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    rlayer = self.iface.addRasterLayer(output_tif_path, os.path.basename(output_tif_path))
                    
                    if (indicator_id == "si" or indicator_id == "skyline") and rlayer and rlayer.isValid():
                        self.apply_shape_index_style(rlayer)
                    
                    if csv_output_path:
                         try:
                             df = pd.read_csv(csv_output_path)
                             if "center_x" in df.columns or "grid_id" in df.columns:
                                 self.iface.messageBar().pushMessage("Tip", "Generated CSV contains data, can be imported via 'Add Delimited Text Layer' to view in QGIS.", level=Qgis.Info)
                         except:
                             pass

                self.accept()
            else:
                QMessageBox.critical(self, "Calculation Failed", "An error occurred during calculation. Please check QGIS log message panel for details.")

        except Exception as e:
            QMessageBox.critical(self, "Critical Error", f"Plugin execution exception: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_shape_index_style(self, layer):
        """
        Apply pseudo-color rendering for Shape Index and Skyline Index results.
        """
        try:
            provider = layer.dataProvider()
            stats = provider.bandStatistics(1, QgsRasterBandStats.All)
            min_val = stats.minimumValue
            max_val = stats.maximumValue
            
            fcn = QgsColorRampShader()
            fcn.setColorRampType(QgsColorRampShader.Interpolated)
            
            style = QgsStyle.defaultStyle()
            ramp = style.colorRamp('Spectral')
            if ramp:
                ramp.invert() 
                
                lst = []
                item_count = 5
                if max_val <= min_val:
                    max_val = min_val + 0.1
                
                for i in range(item_count):
                    val = min_val + (max_val - min_val) * i / (item_count - 1)
                    col = ramp.color(i / (item_count - 1))
                    lst.append(QgsColorRampShader.ColorRampItem(val, col, f'{val:.2f}'))
                    
                fcn.setColorRampItemList(lst)
                
                shader = QgsRasterShader()
                shader.setRasterShaderFunction(fcn)
                renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
                layer.setRenderer(renderer)
                layer.triggerRepaint()
        except Exception as e:
            print(f"Failed to apply style: {e}")

    def reject(self):
        """Override close/cancel event, clean up temporary files"""
        for f in self.temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        super(UrbanSpatialPatternAnalysisDialog, self).reject()