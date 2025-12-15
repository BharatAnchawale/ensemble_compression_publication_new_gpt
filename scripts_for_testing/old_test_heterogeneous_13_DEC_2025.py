"""
Test Ensemble on Heterogeneous Data
Purpose: Demonstrate ensemble wins on mixed-content files
"""

import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from compression.ensemble_13_DEC_2025 import test_ensemble_on_files

# Get heterogeneous files
hetero_dir = Path("data/academic_corpora/heterogeneous")
files = [str(f) for f in hetero_dir.glob("*.dat") if f.is_file()]

if not files:
    print("❌ No heterogeneous files found!")
    print("Please run: python create_heterogeneous_data_13_DEC_2025.py first")
    sys.exit(1)

print(f"Found {len(files)} heterogeneous test files")
print("Files:", [os.path.basename(f) for f in files])

# Test with original ensemble (64KB chunks)
print("\n" + "="*80)
print("TESTING ENSEMBLE ON HETEROGENEOUS DATA")
print("="*80)

test_ensemble_on_files(
    files,
    output_csv="results/experiments/ensemble_heterogeneous_13_DEC_2025.csv",
    chunk_size=64*1024
)