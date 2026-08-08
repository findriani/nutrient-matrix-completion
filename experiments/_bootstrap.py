"""
_bootstrap.py — make the project root importable from inside experiments/.

The scripts in this folder live in <project>/experiments/ but depend on config.py,
utils.py and the raw data files (TKPIv2.csv, usda_sr_legacy_2018_wide.csv,
FoodData_Central_...) which live at the project root. Importing this module
puts the project root on sys.path and exposes PROJECT_ROOT for data paths.

Usage (at the very top of a script, before `from config import ...`):
    from _bootstrap import PROJECT_ROOT
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
