# Satellite Pass Computation Tool

This Python module provides tools to compute past and future satellite passes over a given area. It includes algorithms developed by C. Chupin under the PyGoat toolbox (*PhD CC/YTT*).

(YTT, 01/2024) The code was adapted for SWOT altimetry with new ephemerides and extracted from the PyGoat toolbox as an independent module with very few library dependencies.

**Warning:** This code is still under development, and significant changes may take place in future versions.

## Requirements

- Python 3.9 or newer
- Required Python libraries: `numpy`, `pandas`
- Libraries used by the example notebooks: `matplotlib`, `jupyter`
- `pathlib` is part of the Python standard library

## Installation

Clone the repository into `my_directory`, then install it in editable mode:

```bash
cd my_directory/satpass
python -m pip install -e .
```

The module can then be imported in Python:

```python
import satpass
```

Alternatively, without installing the package, add its `src` directory to the Python path:

```python
import sys
sys.path.append("my_directory/satpass/src")

import satpass
```

## Usage

The notebooks in `examples/` demonstrate the main workflows for computing satellite passes over specific areas and colocating fixed or moving observations with SWOT.

```python
import pandas as pd
from satpass import sat_pass

# Define the mission, time interval, and geographic area.
mission = "sw"
date = [pd.Timestamp("2023-11-11"), pd.Timestamp("2023-11-17")]
area = (145, 160, -59, -51)

# Compute satellite passes.
passes_df = sat_pass(mission, date, area)

# Display the results.
print(passes_df)
```

The area format is:

```text
(longitude_min, longitude_max, latitude_min, latitude_max)
```

## SWOT colocation

For a fixed mooring, `colocate_xy` returns every nominal SWOT passage during a given interval:

```python
from satpass import colocate_xy

passages = colocate_xy(
    x=145.0,
    y=-55.0,
    date_range=["2026-08-01", "2026-08-31"],
    max_distance_km=65.0,
)
```

For a moving platform or observations containing longitude, latitude, and time, use `colocate_xyt`:

```python
from satpass import colocate_xyt

matches = colocate_xyt(
    x=observations["longitude"],
    y=observations["latitude"],
    t=observations["time"],
    n_passes=1,
)
```

These routines use nominal ground tracks for planning and first-stage colocation. Science-grade colocation should verify the actual satellite product, pixel geolocation, swath geometry, quality flags, and data availability.

## Example notebooks

- `plot_nominal_tracks.ipynb`: plot nominal tracks over a region.
- `swot_passes_time_region.ipynb`: compute SWOT passages in a region and time interval.
- `colocate_mooring_with_swot.ipynb`: list passages over a fixed mooring.
- `colocate_synthetic_glider_with_swot.ipynb`: colocate a moving synthetic glider.