"""
Ensemble Compression Module
Created: December 13, 2025
Purpose: Chunk-level adaptive algorithm selection for optimal compression
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Import baseline compressor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.baseline_12_DEC_2025_12_00_00 import BaselineCompressor


class EnsembleCompressor:
    """
    Ensemble compressor that splits files into chunks and selects
    the best algorithm for each chunk independently
    """
    
    def __init__(self, chunk_size=64*1024):
        """
        Initialize ensemble compressor
        
        Args:
            chunk_size: Size of chunks in bytes (default: 64KB)
        """
        self.chunk_size = chunk_size
        self.compressor = BaselineCompressor()
        self.algorithms = list(self.compressor.algorithms.keys())
        
        print(f"Ensemble Compressor initialized:")
        print(f"  Chunk size: {chunk_size/1024:.0f} KB")
        print(f"  Algorithms: {', '.join(self.algorithms)}")
    
    def compress_file(self, filepath, verbose=False):
        """
        Compress a file using ensemble approach
        
        Args:
            filepath: Path to file to compress
            verbose: Print detailed chunk-level information
            
        Returns:
            dict: Compression results with chunk details
        """
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        original_size = len(data)
        
        if original_size == 0:
            print(f"Warning: Empty file {filepath}")
            return None
        
        # Split into chunks
        num_chunks = (original_size + self.chunk_size - 1) // self.chunk_size
        chunks = [data[i:i+self.chunk_size] 
                 for i in range(0, original_size, self.chunk_size)]
        
        if verbose:
            print(f"\nProcessing: {os.path.basename(filepath)}")
            print(f"  Original size: {original_size/1024:.2f} KB")
            print(f"  Chunks: {num_chunks}")
        
        # Process each chunk
        chunk_results = []
        total_compressed = 0
        algorithm_selections = {algo: 0 for algo in self.algorithms}
        
        start_time = time.time()
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_original_size = len(chunk)
            
            # Test all algorithms on this chunk
            best_algo = None
            best_compressed = None
            best_ratio = 0
            
            for algo in self.algorithms:
                try:
                    compressed = self.compressor.algorithms[algo](chunk)
                    ratio = chunk_original_size / len(compressed)
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_compressed = compressed
                        best_algo = algo
                except Exception as e:
                    if verbose:
                        print(f"    Chunk {chunk_idx}: {algo} failed - {e}")
                    continue
            
            if best_compressed is None:
                print(f"Warning: All algorithms failed for chunk {chunk_idx}")
                best_compressed = chunk  # Fallback: use uncompressed
                best_algo = 'none'
                best_ratio = 1.0
            
            compressed_size = len(best_compressed)
            total_compressed += compressed_size
            algorithm_selections[best_algo] += 1
            
            chunk_results.append({
                'chunk_index': chunk_idx,
                'algorithm': best_algo,
                'original_size': chunk_original_size,
                'compressed_size': compressed_size,
                'ratio': best_ratio
            })
            
            if verbose and chunk_idx < 5:  # Show first 5 chunks
                print(f"    Chunk {chunk_idx}: {best_algo} -> {best_ratio:.2f}x "
                      f"({chunk_original_size/1024:.1f} KB → {compressed_size/1024:.1f} KB)")
        
        compression_time = time.time() - start_time
        
        # Calculate overall metrics
        overall_ratio = original_size / total_compressed if total_compressed > 0 else 0
        speed_mbps = (original_size / (1024*1024)) / compression_time if compression_time > 0 else 0
        
        # Calculate algorithm distribution
        algo_distribution = {algo: (count / num_chunks * 100) 
                            for algo, count in algorithm_selections.items() 
                            if count > 0}
        
        if verbose:
            print(f"  Ensemble ratio: {overall_ratio:.3f}x")
            print(f"  Compression time: {compression_time:.3f}s")
            print(f"  Speed: {speed_mbps:.2f} MB/s")
            print(f"  Algorithm usage:")
            for algo, pct in sorted(algo_distribution.items(), key=lambda x: x[1], reverse=True):
                print(f"    {algo}: {pct:.1f}%")
        
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'algorithm': 'ensemble',
            'original_size': original_size,
            'compressed_size': total_compressed,
            'compression_ratio': overall_ratio,
            'compression_time': compression_time,
            'speed_mbps': speed_mbps,
            'num_chunks': num_chunks,
            'chunk_size': self.chunk_size,
            'algorithm_distribution': algo_distribution,
            'chunk_details': chunk_results
        }
    
    def compare_with_baseline(self, filepath):
        """
        Compare ensemble compression with all baseline algorithms
        
        Args:
            filepath: Path to file to compress
            
        Returns:
            dict: Comparison results
        """
        print(f"\n{'='*80}")
        print(f"COMPARING: {os.path.basename(filepath)}")
        print(f"{'='*80}")
        
        # Get ensemble result
        ensemble_result = self.compress_file(filepath, verbose=True)
        
        if ensemble_result is None:
            return None
        
        # Get baseline results
        print(f"\n{'='*80}")
        print("BASELINE ALGORITHMS:")
        print(f"{'='*80}")
        
        baseline_results = {}
        for algo in self.algorithms:
            result = self.compressor.compress_file(filepath, algo)
            if result:
                baseline_results[algo] = result
                print(f"  {algo:8s}: {result['compression_ratio']:.3f}x  "
                      f"({result['compressed_size']/1024:.2f} KB, {result['compression_time']:.3f}s)")
        
        # Calculate improvements
        print(f"\n{'='*80}")
        print("ENSEMBLE VS BASELINE:")
        print(f"{'='*80}")
        
        best_baseline_algo = max(baseline_results.keys(), 
                                key=lambda x: baseline_results[x]['compression_ratio'])
        best_baseline_ratio = baseline_results[best_baseline_algo]['compression_ratio']
        
        improvement_pct = ((ensemble_result['compression_ratio'] - best_baseline_ratio) 
                          / best_baseline_ratio * 100)
        
        print(f"  Ensemble:        {ensemble_result['compression_ratio']:.3f}x")
        print(f"  Best baseline:   {best_baseline_ratio:.3f}x ({best_baseline_algo})")
        print(f"  Improvement:     {improvement_pct:+.2f}%")
        
        if improvement_pct > 0:
            print(f"  ✅ Ensemble WINS by {improvement_pct:.2f}%")
        else:
            print(f"  ⚠️  Baseline wins by {abs(improvement_pct):.2f}%")
        
        return {
            'filename': os.path.basename(filepath),
            'ensemble': ensemble_result,
            'baseline': baseline_results,
            'best_baseline': best_baseline_algo,
            'improvement_pct': improvement_pct
        }


def test_ensemble_on_files(file_list, output_csv=None, chunk_size=64*1024):
    """
    Test ensemble compression on multiple files
    
    Args:
        file_list: List of file paths
        output_csv: Path to save results CSV
        chunk_size: Chunk size in bytes
        
    Returns:
        tuple: (ensemble_results, baseline_results, comparison_summary)
    """
    import pandas as pd
    
    ensemble_comp = EnsembleCompressor(chunk_size=chunk_size)
    
    ensemble_results = []
    baseline_results = []
    comparison_summary = []
    
    print(f"\n{'='*80}")
    print(f"TESTING {len(file_list)} FILES WITH ENSEMBLE COMPRESSION")
    print(f"{'='*80}\n")
    
    for i, filepath in enumerate(file_list, 1):
        print(f"\n[{i}/{len(file_list)}] " + "="*70)
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            continue
        
        # Ensemble compression
        ensemble_result = ensemble_comp.compress_file(filepath, verbose=True)
        if ensemble_result:
            ensemble_results.append({
                'filename': ensemble_result['filename'],
                'filepath': ensemble_result['filepath'],
                'algorithm': 'ensemble',
                'original_size': ensemble_result['original_size'],
                'compressed_size': ensemble_result['compressed_size'],
                'compression_ratio': ensemble_result['compression_ratio'],
                'compression_time': ensemble_result['compression_time'],
                'speed_mbps': ensemble_result['speed_mbps'],
                'num_chunks': ensemble_result['num_chunks']
            })
        
        # Baseline algorithms for comparison
        print(f"\n  Baseline comparison:")
        best_baseline_ratio = 0
        best_baseline_algo = None
        
        for algo in ensemble_comp.algorithms:
            result = ensemble_comp.compressor.compress_file(filepath, algo)
            if result:
                baseline_results.append(result)
                print(f"    {algo:8s}: {result['compression_ratio']:.3f}x")
                
                if result['compression_ratio'] > best_baseline_ratio:
                    best_baseline_ratio = result['compression_ratio']
                    best_baseline_algo = algo
        
        # Calculate improvement
        if ensemble_result and best_baseline_ratio > 0:
            improvement_pct = ((ensemble_result['compression_ratio'] - best_baseline_ratio) 
                              / best_baseline_ratio * 100)
            
            comparison_summary.append({
                'filename': ensemble_result['filename'],
                'ensemble_ratio': ensemble_result['compression_ratio'],
                'best_baseline_ratio': best_baseline_ratio,
                'best_baseline_algo': best_baseline_algo,
                'improvement_pct': improvement_pct,
                'file_size_kb': ensemble_result['original_size'] / 1024
            })
            
            print(f"\n  📊 Ensemble: {ensemble_result['compression_ratio']:.3f}x  |  "
                  f"Best baseline ({best_baseline_algo}): {best_baseline_ratio:.3f}x  |  "
                  f"Improvement: {improvement_pct:+.2f}%")
    
    # Save results to CSV
    if ensemble_results and output_csv:
        # Combine ensemble and baseline results
        all_results = ensemble_results + baseline_results
        df = pd.DataFrame(all_results)
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
        
        # Save comparison summary
        if comparison_summary:
            comparison_csv = output_csv.replace('.csv', '_comparison.csv')
            df_comp = pd.DataFrame(comparison_summary)
            df_comp.to_csv(comparison_csv, index=False)
            print(f"✅ Comparison saved to: {comparison_csv}")
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}\n")
    
    if ensemble_results:
        df_ensemble = pd.DataFrame(ensemble_results)
        print(f"Ensemble compression:")
        print(f"  Average ratio: {df_ensemble['compression_ratio'].mean():.3f}x ± {df_ensemble['compression_ratio'].std():.3f}")
        print(f"  Average speed: {df_ensemble['speed_mbps'].mean():.2f} MB/s")
    
    if baseline_results:
        df_baseline = pd.DataFrame(baseline_results)
        print(f"\nBaseline algorithms:")
        for algo in df_baseline['algorithm'].unique():
            algo_data = df_baseline[df_baseline['algorithm'] == algo]
            print(f"  {algo:8s}: {algo_data['compression_ratio'].mean():.3f}x ± {algo_data['compression_ratio'].std():.3f}")
    
    if comparison_summary:
        df_comp = pd.DataFrame(comparison_summary)
        avg_improvement = df_comp['improvement_pct'].mean()
        wins = len(df_comp[df_comp['improvement_pct'] > 0])
        total = len(df_comp)
        
        print(f"\n🎯 Ensemble Performance:")
        print(f"  Average improvement: {avg_improvement:+.2f}%")
        print(f"  Win rate: {wins}/{total} ({wins/total*100:.1f}%)")
        print(f"  Best improvement: {df_comp['improvement_pct'].max():+.2f}%")
        print(f"  Worst case: {df_comp['improvement_pct'].min():+.2f}%")
    
    return ensemble_results, baseline_results, comparison_summary


if __name__ == "__main__":
    # Test on a few Canterbury files
    test_files = [
        "data/academic_corpora/canterbury/alice29.txt",
        "data/academic_corpora/canterbury/asyoulik.txt",
        "data/academic_corpora/canterbury/lcet10.txt",
    ]
    
    # Filter to files that exist
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"Testing ensemble compression on {len(existing_files)} files...")
        test_ensemble_on_files(
            existing_files,
            output_csv="results/experiments/ensemble_test_13_DEC_2025.csv",
            chunk_size=64*1024
        )
    else:
        print("⚠️  No test files found!")