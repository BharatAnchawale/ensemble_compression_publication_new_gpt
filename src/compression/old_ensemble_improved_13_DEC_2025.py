"""
Improved Ensemble Compression Module
Created: December 13, 2025
Purpose: Enhanced chunk-level ensemble with adaptive chunk sizes and overhead reduction
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.baseline_12_DEC_2025_12_00_00 import BaselineCompressor


class ImprovedEnsembleCompressor:
    """
    Improved ensemble with:
    1. Adaptive chunk sizing
    2. Smart algorithm pre-selection (skip poor performers)
    3. Overhead-aware selection
    """
    
    def __init__(self, chunk_size=64*1024, adaptive=True):
        self.chunk_size = chunk_size
        self.adaptive = adaptive
        self.compressor = BaselineCompressor()
        self.algorithms = list(self.compressor.algorithms.keys())
        
        print(f"Improved Ensemble Compressor initialized:")
        print(f"  Base chunk size: {chunk_size/1024:.0f} KB")
        print(f"  Adaptive mode: {adaptive}")
        print(f"  Algorithms: {', '.join(self.algorithms)}")
    
    def _get_adaptive_chunk_size(self, file_size):
        """Determine optimal chunk size based on file size"""
        if not self.adaptive:
            return self.chunk_size
        
        # Larger files can use larger chunks
        if file_size < 100 * 1024:  # < 100 KB
            return 32 * 1024  # 32 KB chunks
        elif file_size < 1024 * 1024:  # < 1 MB
            return 64 * 1024  # 64 KB chunks
        else:  # >= 1 MB
            return 128 * 1024  # 128 KB chunks
    
    def compress_file(self, filepath, verbose=False):
        """Compress file with improved ensemble strategy"""
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        original_size = len(data)
        if original_size == 0:
            return None
        
        # Adaptive chunk size
        chunk_size = self._get_adaptive_chunk_size(original_size)
        
        # For very small files, just use best single algorithm
        if original_size < chunk_size * 0.8:  # Less than one full chunk
            if verbose:
                print(f"Small file ({original_size/1024:.1f} KB) - using best single algorithm")
            
            best_algo = None
            best_result = None
            best_ratio = 0
            
            for algo in self.algorithms:
                result = self.compressor.compress_file(filepath, algo)
                if result and result['compression_ratio'] > best_ratio:
                    best_ratio = result['compression_ratio']
                    best_result = result
                    best_algo = algo
            
            if best_result:
                best_result['algorithm'] = f'ensemble({best_algo})'
                best_result['num_chunks'] = 1
                best_result['chunk_size'] = chunk_size
            
            return best_result
        
        # Split into chunks
        num_chunks = (original_size + chunk_size - 1) // chunk_size
        chunks = [data[i:i+chunk_size] 
                 for i in range(0, original_size, chunk_size)]
        
        if verbose:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            print(f"  Original size: {original_size/1024:.2f} KB")
            print(f"  Chunk size: {chunk_size/1024:.0f} KB")
            print(f"  Chunks: {num_chunks}")
        
        # Quick pre-test on first chunk to eliminate poor performers
        first_chunk = chunks[0]
        algorithm_performance = {}
        
        for algo in self.algorithms:
            try:
                compressed = self.compressor.algorithms[algo](first_chunk)
                ratio = len(first_chunk) / len(compressed)
                speed_start = time.time()
                _ = self.compressor.algorithms[algo](first_chunk)
                speed = len(first_chunk) / (time.time() - speed_start)
                
                # Combined score: ratio is more important but speed matters
                score = ratio * 0.7 + (speed / 1000000) * 0.3  # Normalize speed
                algorithm_performance[algo] = {
                    'ratio': ratio,
                    'speed': speed,
                    'score': score
                }
            except:
                continue
        
        # Select top 3 algorithms for detailed testing
        top_algos = sorted(algorithm_performance.keys(), 
                          key=lambda x: algorithm_performance[x]['score'],
                          reverse=True)[:3]
        
        if verbose:
            print(f"  Pre-selected top algorithms: {', '.join(top_algos)}")
        
        # Process each chunk with top algorithms only
        chunk_results = []
        total_compressed = 0
        algorithm_selections = {algo: 0 for algo in self.algorithms}
        
        start_time = time.time()
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_original_size = len(chunk)
            
            best_algo = None
            best_compressed = None
            best_ratio = 0
            
            # Test only top algorithms
            for algo in top_algos:
                try:
                    compressed = self.compressor.algorithms[algo](chunk)
                    compressed_size = len(compressed)
                    
                    # Account for metadata overhead (1 byte per chunk for algo ID)
                    adjusted_size = compressed_size + 1
                    ratio = chunk_original_size / adjusted_size
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_compressed = compressed
                        best_algo = algo
                except:
                    continue
            
            if best_compressed is None:
                # Fallback
                best_compressed = chunk
                best_algo = 'none'
            
            # Add 1 byte metadata overhead per chunk
            compressed_size = len(best_compressed) + 1
            total_compressed += compressed_size
            algorithm_selections[best_algo] += 1
            
            chunk_results.append({
                'chunk_index': chunk_idx,
                'algorithm': best_algo,
                'original_size': chunk_original_size,
                'compressed_size': compressed_size,
                'ratio': chunk_original_size / compressed_size
            })
            
            if verbose and chunk_idx < 5:
                print(f"    Chunk {chunk_idx}: {best_algo} -> {best_ratio:.2f}x")
        
        compression_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_ratio = original_size / total_compressed if total_compressed > 0 else 0
        speed_mbps = (original_size / (1024*1024)) / compression_time if compression_time > 0 else 0
        
        algo_distribution = {algo: (count / num_chunks * 100) 
                            for algo, count in algorithm_selections.items() 
                            if count > 0}
        
        if verbose:
            print(f"  Ensemble ratio: {overall_ratio:.3f}x")
            print(f"  Algorithm usage:")
            for algo, pct in sorted(algo_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"    {algo}: {pct:.1f}%")
        
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'algorithm': 'ensemble_improved',
            'original_size': original_size,
            'compressed_size': total_compressed,
            'compression_ratio': overall_ratio,
            'compression_time': compression_time,
            'speed_mbps': speed_mbps,
            'num_chunks': num_chunks,
            'chunk_size': chunk_size,
            'algorithm_distribution': algo_distribution,
            'chunk_details': chunk_results
        }


def test_improved_ensemble(file_list, output_csv=None):
    """Test improved ensemble on files"""
    import pandas as pd
    
    comp_basic = BaselineCompressor()
    comp_improved = ImprovedEnsembleCompressor(adaptive=True)
    
    all_results = []
    comparison = []
    
    print(f"\n{'='*80}")
    print(f"TESTING IMPROVED ENSEMBLE ON {len(file_list)} FILES")
    print(f"{'='*80}\n")
    
    for i, filepath in enumerate(file_list, 1):
        print(f"\n[{i}/{len(file_list)}] {'='*70}")
        
        if not os.path.exists(filepath):
            continue
        
        # Improved ensemble
        improved_result = comp_improved.compress_file(filepath, verbose=True)
        if improved_result:
            all_results.append(improved_result)
        
        # Baseline for comparison
        print(f"\n  Baseline algorithms:")
        best_baseline = 0
        best_algo = None
        
        for algo in comp_basic.algorithms.keys():
            result = comp_basic.compress_file(filepath, algo)
            if result:
                all_results.append(result)
                ratio = result['compression_ratio']
                print(f"    {algo:8s}: {ratio:.3f}x")
                
                if ratio > best_baseline:
                    best_baseline = ratio
                    best_algo = algo
        
        # Compare
        if improved_result and best_baseline > 0:
            improvement = ((improved_result['compression_ratio'] - best_baseline) 
                          / best_baseline * 100)
            
            comparison.append({
                'filename': improved_result['filename'],
                'improved_ratio': improved_result['compression_ratio'],
                'best_baseline_ratio': best_baseline,
                'best_baseline_algo': best_algo,
                'improvement_pct': improvement,
                'file_size_kb': improved_result['original_size'] / 1024
            })
            
            print(f"\n  📊 Improved: {improved_result['compression_ratio']:.3f}x  |  "
                  f"Best baseline ({best_algo}): {best_baseline:.3f}x  |  "
                  f"Improvement: {improvement:+.2f}%")
    
    # Save results
    if all_results and output_csv:
        df = pd.DataFrame(all_results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
        
        if comparison:
            comp_csv = output_csv.replace('.csv', '_comparison.csv')
            df_comp = pd.DataFrame(comparison)
            df_comp.to_csv(comp_csv, index=False)
            print(f"✅ Comparison saved to: {comp_csv}")
    
    # Summary
    print(f"\n{'='*80}")
    print("IMPROVED ENSEMBLE SUMMARY")
    print(f"{'='*80}\n")
    
    if comparison:
        df_comp = pd.DataFrame(comparison)
        avg_improvement = df_comp['improvement_pct'].mean()
        wins = len(df_comp[df_comp['improvement_pct'] > 0])
        total = len(df_comp)
        ties = len(df_comp[df_comp['improvement_pct'] == 0])
        
        print(f"🎯 Performance vs Best Baseline:")
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        print(f"  Win rate: {wins}/{total} ({wins/total*100:.1f}%)")
        print(f"  Ties: {ties}/{total}")
        print(f"  Best improvement: {df_comp['improvement_pct'].max():+.2f}%")
        print(f"  Worst case: {df_comp['improvement_pct'].min():+.2f}%")
    
    return all_results, comparison


if __name__ == "__main__":
    test_files = [
        "data/academic_corpora/canterbury/alice29.txt",
        "data/academic_corpora/canterbury/kennedy.xls",
        "data/academic_corpora/canterbury/lcet10.txt",
    ]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        test_improved_ensemble(
            existing_files,
            output_csv="results/experiments/ensemble_improved_test_13_DEC_2025.csv"
        )