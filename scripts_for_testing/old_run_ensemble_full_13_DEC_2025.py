"""
Full Ensemble Compression Experiment
Run ensemble on all Canterbury corpus files
"""

import sys
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from compression.ensemble_13_DEC_2025 import test_ensemble_on_files

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

# Run ensemble compression on all files
test_ensemble_on_files(
    all_files,
    output_csv="results/experiments/ensemble_results_13_DEC_2025.csv",
    chunk_size=64*1024
)