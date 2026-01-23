# core/shape_index.py
import rasterio
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import traceback
import os

# [修复] 引入 Figure 和 Backend，完全脱离 pyplot
# 这种方式是 QGIS/ArcGIS 等嵌入式 Python 环境中生成图表的唯一标准做法
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

def calculate_shape_index(input_height_tif, output_tif_path, threshold_min=1, threshold_max=None, output_csv_path=None, output_png_path=None):
    try:
        print(f"Start processing Shape Index: {input_height_tif}")
        with rasterio.open(input_height_tif) as src:
            # Read all data
            data = src.read(1)
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform

            # 1. Threshold processing
            if threshold_max is None:
                threshold_max = np.max(data)
            
            # Generate binary mask
            mask = (data >= threshold_min) & (data <= threshold_max)

            # 2. Connected component labeling
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)
            
            print(f"Detected {num_labels} building patches")

            # Initialize output array
            shape_index_array = np.zeros((height, width), dtype=np.float32)
            patch_metrics = []

            # 3. Calculate shape index for each patch
            if num_labels > 0:
                props = regionprops(labeled_mask)
                for prop in props:
                    if prop.area <= 0: continue

                    area = prop.area
                    perimeter = prop.perimeter

                    # Core formula: SI = P / (2 * sqrt(pi * A))
                    shape_index = perimeter / (2 * np.sqrt(np.pi * area))

                    coords = prop.coords 
                    shape_index_array[coords[:, 0], coords[:, 1]] = shape_index

                    patch_metrics.append({
                        'Patch_ID': prop.label,
                        'Patch_Area_pixels': area,
                        'Perimeter': perimeter,
                        'Shape_Index': shape_index
                    })
        
        print(f"Metrics calculation complete. Total patches: {len(patch_metrics)}")

        # Output CSV
        if output_csv_path and patch_metrics:
            try:
                pd.DataFrame(patch_metrics).to_csv(output_csv_path, index=False)
                print(f"Shape Index CSV result saved to: {output_csv_path}")
            except Exception as e:
                print(f"Warning: Failed to save CSV: {e}")

        # [深度修复] 强制绘图逻辑
        if output_png_path:
            if not patch_metrics:
                print("Warning: No patches detected (patch_metrics is empty), skipping histogram generation.")
            else:
                print(f"Generating histogram to: {output_png_path}")
                try:
                    # 1. 准备数据
                    df = pd.DataFrame(patch_metrics)
                    
                    # 2. 创建纯后端 Figure 对象 (不涉及 GUI)
                    fig = Figure(figsize=(10, 6), dpi=300)
                    canvas = FigureCanvasAgg(fig) # 将 Figure 绑定到 Agg 画布
                    ax = fig.add_subplot(111)
                    
                    # 3. 绘制直方图
                    if not df.empty and 'Shape_Index' in df.columns:
                        ax.hist(df['Shape_Index'], bins=50, color='#d62728', alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Shape Index (1.0 = Circle)')
                        ax.set_ylabel('Frequency')
                        ax.set_title(f'Distribution of Patch Shape Index (Total: {len(df)})')
                        ax.grid(axis='y', alpha=0.5, linestyle='--')
                        
                        # 4. [Fix] Ensure output directory exists and save by passing file path (more compatible in QGIS)
                        out_dir = os.path.dirname(output_png_path)
                        if out_dir and (not os.path.exists(out_dir)):
                            os.makedirs(out_dir, exist_ok=True)

                        values = df['Shape_Index'].replace([np.inf, -np.inf], np.nan).dropna()
                        if values.empty:
                            print("Warning: Shape_Index values are empty after filtering NaN/Inf, skipping plot.")
                        else:
                            canvas.draw()
                            try:
                                # Prefer saving by file path (string) for best compatibility
                                fig.savefig(output_png_path, format='png', bbox_inches='tight')
                            except Exception as e:
                                print(f"Warning: fig.savefig failed: {e}")
                            
                            # If matplotlib save failed silently, try a Qt fallback renderer
                            if (not os.path.exists(output_png_path)) or (os.path.getsize(output_png_path) <= 0):
                                try:
                                    from qgis.PyQt.QtGui import QImage, QPainter, QPen, QColor, QFont
                                    from qgis.PyQt.QtCore import Qt
                                    counts, edges = np.histogram(values.to_numpy(dtype=float), bins=50)
                                    if counts.size and int(counts.max()) > 0:
                                        W, H = 1200, 800
                                        margin_l, margin_r, margin_t, margin_b = 120, 60, 80, 120
                                        plot_w = W - margin_l - margin_r
                                        plot_h = H - margin_t - margin_b
                                        img = QImage(W, H, QImage.Format_ARGB32)
                                        img.fill(QColor(255, 255, 255))
                                        p = QPainter(img)
                                        p.setRenderHint(QPainter.Antialiasing, True)

                                        p.setPen(QPen(QColor(0, 0, 0)))
                                        p.setFont(QFont("Arial", 16))
                                        p.drawText(0, 0, W, margin_t, Qt.AlignCenter,
                                                   f"Distribution of Patch Shape Index (Total: {len(values)})")

                                        axis_pen = QPen(QColor(0, 0, 0))
                                        axis_pen.setWidth(2)
                                        p.setPen(axis_pen)
                                        x0, y0 = margin_l, H - margin_b
                                        x1, y1 = W - margin_r, margin_t
                                        p.drawLine(x0, y0, x1, y0)
                                        p.drawLine(x0, y0, x0, y1)

                                        bar_pen = QPen(QColor(0, 0, 0))
                                        bar_pen.setWidth(1)
                                        p.setPen(bar_pen)
                                        bar_color = QColor(214, 39, 40)
                                        max_count = float(counts.max())
                                        bins = len(counts)
                                        bar_w = max(1, int(plot_w / max(1, bins)))

                                        for i, c in enumerate(counts):
                                            if c <= 0:
                                                continue
                                            h_px = int((float(c) / max_count) * (plot_h - 10))
                                            x = x0 + i * bar_w
                                            y = y0 - h_px
                                            p.fillRect(x, y, bar_w - 1, h_px, bar_color)
                                            p.drawRect(x, y, bar_w - 1, h_px)

                                        p.setPen(QPen(QColor(0, 0, 0)))
                                        p.setFont(QFont("Arial", 12))
                                        p.drawText(margin_l, H - margin_b + 50, plot_w, 30, Qt.AlignCenter,
                                                   "Shape Index (1.0 = Circle)")
                                        p.drawText(10, margin_t + plot_h // 2 - 15, 100, 30, Qt.AlignCenter,
                                                   "Frequency")
                                        p.end()

                                        img.save(output_png_path, "PNG")
                                except Exception as e:
                                    print(f"Warning: Qt fallback failed: {e}")
                        
                        if os.path.exists(output_png_path):
                            print(f"Histogram generated successfully: {output_png_path}")
                        else:
                            print("Error: Histogram write executed but file not found on disk.")
                    else:
                        print("Warning: DataFrame is empty or missing 'Shape_Index', skipping plot.")

                except Exception as plot_error:
                    print(f"Warning: Failed to generate histogram: {plot_error}")
                    import traceback
                    traceback.print_exc()

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
