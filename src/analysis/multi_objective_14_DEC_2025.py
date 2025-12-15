"""
Multi-Objective Optimization Module
Created: December 14, 2025
Purpose: Measure compression ratio, speed, and memory usage
         Generate Pareto frontiers and preference-based selection
"""

import os
import sys
import time
import psutil
import numpy as np
import pandas as pd
from pathlib import Path

# Import baseline compressor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from compression.baseline_12_DEC_2025_12_00_00 import BaselineCompressor


class MultiObjectiveCompressor:
    """
    Compressor with multi-objective measurement:
    - Compression ratio (higher is better)
    - Speed in MB/s (higher is better)
    - Memory usage in MB (lower is better)
    """
    
    def __init__(self):
        self.compressor = BaselineCompressor()
        self.algorithms = list(self.compressor.algorithms.keys())
        print(f"Multi-Objective Compressor initialized with {len(self.algorithms)} algorithms")
    
    def measure_memory_usage(self, func, *args):
        """Measure peak memory usage during function execution"""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Execute function
        result = func(*args)
        
        # Get peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_used = peak_memory - baseline_memory
        
        # Ensure non-negative
        memory_used = max(0.1, memory_used)  # Minimum 0.1 MB
        
        return result, memory_used
    
    def compress_with_metrics(self, filepath, algorithm, runs=3):
        """
        Compress file and measure all three objectives
        
        Args:
            filepath: Path to file
            algorithm: Algorithm name
            runs: Number of runs for averaging (default: 3)
        
        Returns:
            dict: All metrics including ratio, speed, memory
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Read file once
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        original_size = len(data)
        
        if original_size == 0:
            return None
        
        # Run multiple times for accurate measurements
        ratios = []
        speeds = []
        memories = []
        compressed_sizes = []
        
        for run in range(runs):
            # Measure compression with memory tracking
            start_time = time.time()
            
            compressed, memory_used = self.measure_memory_usage(
                self.compressor.algorithms[algorithm], data
            )
            
            compression_time = time.time() - start_time
            compressed_size = len(compressed)
            
            # Calculate metrics
            ratio = original_size / compressed_size if compressed_size > 0 else 0
            speed = (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0
            
            ratios.append(ratio)
            speeds.append(speed)
            memories.append(memory_used)
            compressed_sizes.append(compressed_size)
        
        # Average metrics across runs
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'algorithm': algorithm,
            'original_size': original_size,
            'compressed_size': np.mean(compressed_sizes),
            'compressed_size_std': np.std(compressed_sizes),
            'compression_ratio': np.mean(ratios),
            'compression_ratio_std': np.std(ratios),
            'speed_mbps': np.mean(speeds),
            'speed_mbps_std': np.std(speeds),
            'memory_mb': np.mean(memories),
            'memory_mb_std': np.std(memories),
            'num_runs': runs
        }


class ParetoFrontier:
    """Calculate and analyze Pareto frontier"""
    
    @staticmethod
    def is_dominated(point_a, point_b):
        """
        Check if point_a is dominated by point_b
        For our objectives:
        - Ratio: higher is better
        - Speed: higher is better
        - Memory: LOWER is better
        """
        ratio_a, speed_a, memory_a = point_a
        ratio_b, speed_b, memory_b = point_b
        
        # point_a is dominated if point_b is better in all objectives
        # (or equal in some and strictly better in at least one)
        
        better_ratio = ratio_b >= ratio_a
        better_speed = speed_b >= speed_a
        better_memory = memory_b <= memory_a  # Lower is better
        
        at_least_one_strictly_better = (
            ratio_b > ratio_a or 
            speed_b > speed_a or 
            memory_b < memory_a
        )
        
        return (better_ratio and better_speed and better_memory and 
                at_least_one_strictly_better)
    
    @staticmethod
    def calculate_pareto_frontier(results_df):
        """
        Calculate Pareto frontier from results
        
        Args:
            results_df: DataFrame with columns: algorithm, compression_ratio, 
                       speed_mbps, memory_mb
        
        Returns:
            list: Indices of non-dominated solutions (Pareto optimal)
        """
        pareto_indices = []
        
        for i, row_i in results_df.iterrows():
            point_i = (row_i['compression_ratio'], 
                      row_i['speed_mbps'], 
                      row_i['memory_mb'])
            
            is_dominated = False
            
            for j, row_j in results_df.iterrows():
                if i == j:
                    continue
                
                point_j = (row_j['compression_ratio'], 
                          row_j['speed_mbps'], 
                          row_j['memory_mb'])
                
                if ParetoFrontier.is_dominated(point_i, point_j):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return pareto_indices


class PreferenceSelector:
    """Select best algorithm based on user preferences"""
    
    PREFERENCES = {
        'fast': {'ratio': 0.2, 'speed': 0.7, 'memory': 0.1},
        'small': {'ratio': 0.7, 'speed': 0.2, 'memory': 0.1},
        'balanced': {'ratio': 0.4, 'speed': 0.4, 'memory': 0.2},
        'memory_efficient': {'ratio': 0.3, 'speed': 0.2, 'memory': 0.5}
    }
    
    @staticmethod
    def normalize_objectives(results_df):
        """Normalize objectives to [0, 1] scale"""
        df_norm = results_df.copy()
        
        # Normalize ratio and speed (higher is better)
        for col in ['compression_ratio', 'speed_mbps']:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[f'{col}_norm'] = 1.0
        
        # Normalize memory (lower is better, so invert)
        min_val = df_norm['memory_mb'].min()
        max_val = df_norm['memory_mb'].max()
        if max_val > min_val:
            df_norm['memory_mb_norm'] = 1 - (df_norm['memory_mb'] - min_val) / (max_val - min_val)
        else:
            df_norm['memory_mb_norm'] = 1.0
        
        return df_norm
    
    @staticmethod
    def select_by_preference(results_df, preference='balanced'):
        """
        Select best algorithm based on preference
        
        Args:
            results_df: DataFrame with algorithm metrics
            preference: 'fast', 'small', 'balanced', or 'memory_efficient'
        
        Returns:
            tuple: (best_algorithm, score, all_scores)
        """
        if preference not in PreferenceSelector.PREFERENCES:
            raise ValueError(f"Unknown preference: {preference}")
        
        weights = PreferenceSelector.PREFERENCES[preference]
        
        # Normalize objectives
        df_norm = PreferenceSelector.normalize_objectives(results_df)
        
        # Calculate weighted scores
        df_norm['preference_score'] = (
            weights['ratio'] * df_norm['compression_ratio_norm'] +
            weights['speed'] * df_norm['speed_mbps_norm'] +
            weights['memory'] * df_norm['memory_mb_norm']
        )
        
        # Find best
        best_idx = df_norm['preference_score'].idxmax()
        best_algo = df_norm.loc[best_idx, 'algorithm']
        best_score = df_norm.loc[best_idx, 'preference_score']
        
        # Create scores dictionary
        all_scores = dict(zip(df_norm['algorithm'], df_norm['preference_score']))
        
        return best_algo, best_score, all_scores


def run_multi_objective_experiments(file_list, output_csv=None, runs_per_file=3):
    """
    Run multi-objective experiments on all files
    
    Args:
        file_list: List of file paths
        output_csv: Path to save results
        runs_per_file: Number of runs per file for averaging
    
    Returns:
        DataFrame: All results with multi-objective metrics
    """
    compressor = MultiObjectiveCompressor()
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"MULTI-OBJECTIVE EXPERIMENTS: {len(file_list)} files × {len(compressor.algorithms)} algorithms")
    print(f"Runs per measurement: {runs_per_file}")
    print(f"{'='*80}\n")
    
    total_experiments = len(file_list) * len(compressor.algorithms)
    completed = 0
    
    for i, filepath in enumerate(file_list, 1):
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            continue
        
        filename = os.path.basename(filepath)
        file_size_kb = os.path.getsize(filepath) / 1024
        
        print(f"\n[{i}/{len(file_list)}] {filename} ({file_size_kb:.1f} KB)")
        
        for algo in compressor.algorithms:
            result = compressor.compress_with_metrics(filepath, algo, runs=runs_per_file)
            
            if result:
                all_results.append(result)
                completed += 1
                
                print(f"  {algo:8s}: {result['compression_ratio']:.3f}x  "
                      f"{result['speed_mbps']:6.2f} MB/s  "
                      f"{result['memory_mb']:5.1f} MB")
    
    print(f"\n{'='*80}")
    print(f"✅ Completed {completed}/{total_experiments} experiments")
    print(f"{'='*80}")
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Save to CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_results.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")
    
    return df_results


if __name__ == "__main__":
    # Test on a few files
    test_files = [
        "data/academic_corpora/canterbury/alice29.txt",
        "data/academic_corpora/heterogeneous/mixed_formats.dat",
    ]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print("Testing multi-objective measurement...")
        results = run_multi_objective_experiments(
            existing_files,
            output_csv="results/experiments/multi_objective_test_14_DEC_2025.csv",
            runs_per_file=3
        )
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        summary = results.groupby('algorithm').agg({
            'compression_ratio': ['mean', 'std'],
            'speed_mbps': ['mean', 'std'],
            'memory_mb': ['mean', 'std']
        }).round(3)
        print(summary)
    else:
        print("⚠️  No test files found!")