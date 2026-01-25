# Urban Spatial Pattern Analysis (QGIS Plugin)

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Urban Spatial Pattern Analysis** is a QGIS 3 plugin for analyzing urban spatial patterns. It provides **9 indicators** commonly used in urban morphology and spatial-structure studies. The plugin supports raster and vector inputs, outputs GeoTIFF results (optional CSV summaries; optional histogram PNG for Shape Index), and is suitable for grid-based urban built-environment analysis and visualization.
QGIS Repository: https://plugins.qgis.org/plugins/urbanspatialpatternanalysis/

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.  
Please include a `LICENSE` file in the repository root (GPL-3.0 text), and ensure any redistribution/modification complies with GPL-3.0.

---

## Key Features (9 Indicators)

The plugin UI provides the following indicators:

1. **Building Coverage Ratio (BCR)**
2. **Compactness Index (CI)**
3. **Patch Density (PD)**
4. **Shape Index (SI)** *(supports optional histogram PNG export)*
5. **Edge Density (ED)**
6. **Road Density (RD)**
7. **Intersection Density (ID)**
8. **Skyline Index (SLI)**
9. **Average Building Height (ABH)**

---

## Installation (Available in the Official QGIS Plugin Repository)

Since the plugin is published in the official QGIS plugin repository, the recommended installation steps are:

1. Open QGIS (QGIS 3.x is recommended; LTR versions are preferred).
2. Go to **Plugins → Manage and Install Plugins…**
3. Search for the plugin name (e.g., **Urban Spatial Pattern Analysis**) under **All** or **Not Installed**.
4. Click **Install Plugin**.
5. After installation, launch the plugin from the plugin menu/toolbar.

If you cannot find the plugin, it is usually due to networking issues or the plugin list not being refreshed. Try restarting QGIS or refreshing the plugin list in the plugin manager.

---

## Environment & Dependency Setup (Recommended Check on First Use)

Core computations rely on several third‑party Python libraries in the Python environment used by QGIS. Many QGIS distributions include some of these packages, but availability varies by OS and distribution. If you see `ModuleNotFoundError`, install the missing dependencies.

### Core Dependencies
- `numpy`
- `pandas`
- `rasterio`
- `scikit-image`
- `shapely`
- `geopandas`
- `matplotlib` *(only for Shape Index histogram export)*

> Note: The plugin does **not** require a GUI backend for matplotlib. Histogram export is generated using a non-interactive approach (PNG output).

### Windows (OSGeo4W / QGIS LTR) — Common Approach
1. Ensure you are using the **QGIS Python environment**, not the system Python.
2. Open **OSGeo4W Shell** (or otherwise ensure `python`/`pip` point to QGIS Python).
3. Install only what is missing (example):
   ```bash
   python -m pip install numpy pandas scikit-image shapely geopandas matplotlib rasterio
   ```

### macOS / Linux
You can typically install dependencies via system packages, conda, or pip depending on how QGIS is installed. The key requirement is: install packages into **the same Python environment that QGIS uses**. If imports fail, first verify that `python` and `pip` you’re calling match QGIS’s interpreter.

---

## How to Use (End-to-End Workflow)

### 1) Launch the Plugin
Open the plugin dialog in QGIS, then select an indicator from the dropdown (BCR/CI/PD/SI/ED/RD/ID/SLI/ABH).

### 2) Select Input Data
Inputs depend on the indicator:

- **BCR / CI**: Building input (raster or vector)
- **PD / SI / ED / SLI / ABH**: Height input (raster or vector)
- **RD / ID**: Road network (vector line data); optional analysis boundary (polygon)

If you provide a vector as the “building/height input”, the plugin will prompt for rasterization resolution and create a temporary raster for computation.

### 3) Set Output Path
Choose the output GeoTIFF (`.tif`) path. The plugin writes additional files using the same base name:

- If **CSV Output** is enabled, it also saves `same_name.csv`
- For **Shape Index (SI)**, if **PNG Output** is enabled, it also saves `same_name.png` (histogram)

### 4) Configure Parameters (Prompted by Dialogs)
- Most indicators prompt for **Grid Size**, shown in **meters** in the UI. The plugin will internally convert units when required (e.g., for raster-window/pixel-based calculations).
- **RD / ID**: If you don’t provide a boundary polygon, you will be prompted to choose an auto-boundary strategy (e.g., bbox/convex hull/buffer).
- For vector inputs, you will be asked to confirm rasterization resolution. Choose an appropriate value for your study scale (e.g., ~5–10 m for typical urban building analysis, depending on data).

### 5) Run & Load Results
Click **Run** to start computation. After completion, the plugin can load the output raster into the current QGIS project for immediate visualization.

---

## Indicator Output Interpretation (Brief)

- **BCR**: Building pixel proportion per grid cell (coverage heatmap)
- **CI**: Compactness of building shapes per grid (higher often indicates more compact shapes)
- **PD**: Number/density of building patches per grid
- **SI**: Shape complexity index of connected patches; optional histogram of patch SI distribution
- **ED**: Boundary/contour length density per grid
- **RD**: Road length per area (commonly expressed as km/km² or equivalent)
- **ID**: Intersection count per area (count/km²)
- **SLI**: Height variability per grid (e.g., standard deviation) normalized for visualization
- **ABH**: Average building height per grid

---

## Troubleshooting

### 1) `ModuleNotFoundError`
Your QGIS Python environment is missing a dependency. Install the missing module(s) following the setup section above.

### 2) Output is all zeros or very sparse
Often caused by thresholds, value ranges, CRS, or resolution. Check:
- Whether your height raster has valid (non-NoData) values and reasonable units
- Whether rasterization resolution is too coarse (for vector inputs)
- Whether grid size is too small/large, causing unstable grid statistics

### 3) RD/ID results look wrong when no boundary is provided
When no boundary is supplied, the plugin derives an analysis area from the road network. Try a different strategy (bbox/convex hull/buffer), and ensure the road data uses an appropriate projected CRS (meters). Data quality strongly affects results.

### 4) Shape Index histogram PNG is not generated
Most commonly due to an unwritable output folder or missing dependencies. Try saving into a writable directory (e.g., Desktop/Documents) and confirm that matplotlib/Qt dependencies work in your QGIS environment.

---

## Contributing (Optional)

Issues and pull requests are welcome. For reproducible bug reports, please include:
- QGIS version and platform (Windows/macOS/Linux)
- Input data type (raster/vector) and CRS information
- Error traceback or **Log Messages** panel output
- A minimal reproducible dataset (if it can be shared publicly)

---

## Citation (Optional)

If used in academic work, consider citing the plugin in your paper/report, and include the QGIS version and dataset sources.
