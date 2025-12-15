"""
Run Ensemble Compression on ALL 57 Files
Compare with baseline to see if >5% improvement
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compression.baseline_12_DEC_2025_12_00_00 import BaselineCompressor
from compression.ensemble_13_DEC_2025 import EnsembleCompressor
import pandas as pd

print("="*80)
print("ENSEMBLE COMPRESSION TEST - ALL FILES")
print("="*80)

# Get ALL files
all_files = []
corpora = ['artificial', 'calgary', 'canterbury', 'heterogeneous', 
           'heterogeneous_additional', 'large', 'miscellaneous']

for corpus in corpora:
    corpus_path = Path(f"data/academic_corpora/{corpus}")
    if corpus_path.exists():
        files = [str(f) for f in corpus_path.glob("*") 
                if f.is_file() and not f.name.endswith(('.gz', '.tar', '.zip'))]
        all_files.extend(files)
        print(f"  {corpus}: {len(files)} files")

print(f"\nTotal files: {len(all_files)}")

# Run ensemble and baseline
ensemble_compressor = EnsembleCompressor(chunk_size=64*1024)
baseline_compressor = BaselineCompressor()

results = []

for i, filepath in enumerate(all_files, 1):
    print(f"\n[{i}/{len(all_files)}] Processing: {Path(filepath).name}")
    
    # Ensemble
    try:
        ens_result = ensemble_compressor.compress_file(filepath)
        ens_ratio = ens_result['compression_ratio']
    except Exception as e:
        print(f"  ⚠️ Ensemble failed: {e}")
        ens_ratio = 0
    
    # Best baseline (we'll use lzma as it had best ratio)
    try:
        baseline_result = baseline_compressor.compress_file(filepath, 'lzma')
        baseline_ratio = baseline_result['compression_ratio']
    except Exception as e:
        print(f"  ⚠️ Baseline failed: {e}")
        baseline_ratio = 0
    
    # Calculate improvement
    if baseline_ratio > 0:
        improvement = ((ens_ratio - baseline_ratio) / baseline_ratio) * 100
    else:
        improvement = 0
    
    print(f"  Ensemble: {ens_ratio:.3f}x | Baseline: {baseline_ratio:.3f}x | Improvement: {improvement:+.2f}%")
    
    results.append({
        'filename': Path(filepath).name,
        'filepath': filepath,
        'ensemble_ratio': ens_ratio,
        'baseline_ratio': baseline_ratio,
        'improvement_pct': improvement
    })

# Save results
df = pd.DataFrame(results)
df.to_csv('results/experiments/ensemble_all_57files_15_DEC_2025.csv', index=False)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nTotal files: {len(df)}")
print(f"Average ensemble ratio: {df['ensemble_ratio'].mean():.3f}x")
print(f"Average baseline ratio: {df['baseline_ratio'].mean():.3f}x")
print(f"Average improvement: {df['improvement_pct'].mean():+.2f}%")
print(f"\nFiles where ensemble wins: {len(df[df['improvement_pct'] > 0])}/{len(df)} ({len(df[df['improvement_pct'] > 0])/len(df)*100:.1f}%)")
print(f"Files where ensemble loses: {len(df[df['improvement_pct'] < 0])}/{len(df)}")

# Check if >5% improvement
if df['improvement_pct'].mean() > 5:
    print(f"\n✅ SUCCESS! Average improvement {df['improvement_pct'].mean():.2f}% > 5%")
    print("   → Ensemble CAN be a main contribution!")
else:
    print(f"\n⚠️ Average improvement {df['improvement_pct'].mean():.2f}% < 5%")
    print("   → Ensemble remains minor finding, multi-objective is main story")

print(f"\n✅ Results saved to: results/experiments/ensemble_all_57files_15_DEC_2025.csv")