"""
Run Multi-Objective Experiments on ALL Files
"""
import sys
import os
from pathlib import Path

# Add src to path - absolute path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Now import
from analysis.multi_objective_14_DEC_2025 import run_multi_objective_experiments

# Get ALL files from all corpora
all_files = []
corpora = ['artificial', 'calgary', 'canterbury', 'heterogeneous', 
           'heterogeneous_additional', 'large', 'miscellaneous', 'required_corpus']

for corpus in corpora:
    corpus_path = Path(f"data/academic_corpora/{corpus}")
    if corpus_path.exists():
        files = [str(f) for f in corpus_path.glob("*") 
                if f.is_file() and not f.name.endswith(('.gz', '.tar', '.zip'))]
        all_files.extend(files)
        print(f"  {corpus}: {len(files)} files")

print(f"\nTotal files: {len(all_files)}")

# Run experiments
results = run_multi_objective_experiments(
    all_files,
    output_csv="results/experiments/multi_objective_all_300plusfiles_complete_15_DEC_2025.csv",
    runs_per_file=3
)