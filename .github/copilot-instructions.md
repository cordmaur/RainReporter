# RainReporter — Project Guidelines

## Project Overview

RainReporter generates **monthly and daily rainfall PDF reports** for hydrological basins in Brazil. It downloads precipitation data (MERGE/GPM satellite) from **INPE servers** via the `mergedownloader` package, renders maps with `geopandas`/`contextily`/`rasterio`, and produces anomaly analysis against long-term climatology.

## Architecture

```
rainreporter/
  reporter.py         # Orchestrator: Reporter class, batch-processes .json5 configs
  abstract_report.py  # ABC: AbstractReport — base for all report types
  monthly_report.py   # MonthlyReport ("Mensal"): anomaly maps, hydro-climatic plots
  daily_report.py     # DailyReport ("Diario"): 30-day anomaly, WRF forecast, Parquet export
  mapper.py           # Mapper class: cartographic engine (shapefiles, rasters, basemaps)
  utils.py            # open_json_file() — JSON5 config parser
```

- Report type is selected by `"tipo": "Mensal"` or `"tipo": "Diario"` in config.
- `Reporter.generate_pdf(json_file, output_folder)` is the top-level entry point for a single config.
- `Mapper` caches shapefile GeoDataFrames with LRU cache — avoid re-constructing `Mapper` in hot loops.

## Build & Install

```bash
pip install -e .
# Requirements are in .devcontainer/requirements.txt
```

Key dependencies: `geopandas`, `xarray`, `rasterio`, `rioxarray`, `contextily`, `pyjson5`, `mergedownloader`, `adjustText`, `cfgrib`, `ecCodes`.

## Running Reports

```bash
# Explicit folders
python run_reporter.py \
  --configs configs/ --bases bases/ --downloads downloads/ --output output/

# Master folder shorthand
python run_reporter.py --master_folder /path/to/master/

# Hot mode: moves processed configs to configs/hot_processed/ after completion
python run_reporter.py --master_folder /path/to/master/ --hot

# Container
bash run_container.sh
```

**Required master_folder layout:**
```
master_folder/
  bases/       # shapefiles referenced in configs (basin .shp files)
  configs/     # .json5 report specifications
  downloads/   # INPE data download cache
  output/      # generated PDF/XLSX/PNG/Parquet reports
```

## Config Convention (JSON5)

Config files use **JSON5** (comments, trailing commas, unquoted keys allowed). Two levels:

**Global style config** (`reporter.json5` at project root):
- Defines `shape_style` (basin boundary appearance) and `context_shapes` (cities, states, rivers, dams background layers with z-ordering).
- Shapefile paths in the global config are relative to the project root (e.g., `"../data/cities/cidades.shp"`).

**Report spec** (files in `configs/`):
```json5
{
  "arquivo": "OutputFilePrefix",
  "data": false,            // false = today; or "YYYY-MM" for monthly, "YYYY-MM-DD" for daily
  "relatorios": [
    {
      "nome": "Basin Name",
      "tipo": "Mensal",     // or "Diario"
      "shp": "/abs/path/to/basin.shp",
      "inicio_periodo_chuvoso": 10,  // month rainy season starts (1–12)
      "total_meses": 24              // lookback months (Mensal only, optional)
    }
  ]
}
```

## Static Data (data/)

| Folder | Contents | Use |
|--------|----------|-----|
| `data/cities/` | `cidades.shp` | City labels and population |
| `data/states/` | `BR_UF_2022.shp` | State boundaries |
| `data/rivers/` | `main_rivers.shp` | Major Brazilian rivers |
| `data/dams/` | `UHE_Base_Existente.shp` | Hydroelectric dams |

These are the standard background layers rendered by `Mapper.plot_context_layers()`.

## Conventions

- **Language**: All user-facing labels, config keys, and report content are in **Portuguese** (e.g., `"tipo": "Mensal"`, `"arquivo"`, `"nome"`).
- **Notebooks** in `nbs/` are used for development and prototyping — numbered sequentially. Production logic lives in `rainreporter/`.
- **No test suite** currently. Validate changes through notebooks (e.g., `nbs/68-ReportGenerator_development.ipynb`).
- **Parquet + PNG exports**: `DailyReport.export_report_data()` produces Parquet time series and PNG map assets alongside PDFs.
- Config shapefile paths for basin boundaries should be absolute or relative to the working directory at runtime, not the repo root.
