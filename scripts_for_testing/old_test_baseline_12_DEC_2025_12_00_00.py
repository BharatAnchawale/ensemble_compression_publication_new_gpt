"""
Test baseline compression on Canterbury corpus
"""

import sys
import os
from pathlib import Path

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from compression.baseline_12_DEC_2025_12_00_00 import test_on_files

# Get all files
all_files = []

corpora = ['artificial', 'calgary', 'canterbury', 'heterogeneous', 'large', 'miscellaneous']

for corpus in corpora:
    corpus_path = Path(f"data/academic_corpora/{corpus}")
    if corpus_path.exists():
        files = [str(f) for f in corpus_path.glob("*") 
                if f.is_file() and not f.name.endswith(('.gz', '.tar', '.zip'))]
        all_files.extend(files)
        print(f"  {corpus}: {len(files)} files")

print(f"\nTotal files: {len(all_files)}")

# Run tests
test_on_files(
    all_files,
    output_csv="results/experiments/baseline_results_12_DEC_2025_12_00_00.csv"
)