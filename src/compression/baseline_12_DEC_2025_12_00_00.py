"""
Baseline Compression Module
Created: December 12, 2025
Purpose: Implement and test 5 compression algorithms on various file types
"""

import gzip
import bz2
import lzma
import time
import os
import sys
from pathlib import Path

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    print("Warning: zstandard not available")

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    print("Warning: brotli not available")

class BaselineCompressor:
    """
    Baseline compressor implementing 5 modern compression algorithms
    """
    
    def __init__(self):
        self.algorithms = {
            'gzip': self._gzip,
            'bzip2': self._bzip2,
            'lzma': self._lzma
        }
        if ZSTD_AVAILABLE:
            self.algorithms['zstd'] = self._zstd
        if BROTLI_AVAILABLE:
            self.algorithms['brotli'] = self._brotli
        
        print(f"Initialized with {len(self.algorithms)} algorithms: {list(self.algorithms.keys())}")
    
    def compress_file(self, filepath, algorithm='gzip'):
        """
        Compress a file using specified algorithm and measure performance
        
        Args:
            filepath: Path to file to compress
            algorithm: Algorithm name ('gzip', 'bzip2', 'lzma', 'zstd', 'brotli')
            
        Returns:
            dict: Compression results including ratio, time, speed
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {list(self.algorithms.keys())}")
        
        # Read file
        try:
            with open(filepath, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
        
        original_size = len(data)
        
        # Compress
        start_time = time.time()
        try:
            compressed = self.algorithms[algorithm](data)
        except Exception as e:
            print(f"Error compressing {filepath} with {algorithm}: {e}")
            return None
        
        compression_time = time.time() - start_time
        compressed_size = len(compressed)
        
        # Calculate metrics
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        speed_mbps = (original_size / (1024*1024)) / compression_time if compression_time > 0 else 0
        
        return {
            'filename': os.path.basename(filepath),
            'filepath': filepath,
            'algorithm': algorithm,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': ratio,
            'compression_time': compression_time,
            'speed_mbps': speed_mbps
        }
    
    def _gzip(self, data):
        """gzip compression at maximum level"""
        return gzip.compress(data, compresslevel=9)
    
    def _bzip2(self, data):
        """bzip2 compression at maximum level"""
        return bz2.compress(data, compresslevel=9)
    
    def _lzma(self, data):
        """lzma compression at maximum preset"""
        return lzma.compress(data, preset=9)
    
    def _zstd(self, data):
        """zstd compression at maximum level"""
        cctx = zstd.ZstdCompressor(level=19)
        return cctx.compress(data)
    
    def _brotli(self, data):
        """brotli compression at maximum quality"""
        return brotli.compress(data, quality=11)


def test_on_files(file_list, output_csv=None):
    """
    Test all algorithms on a list of files
    
    Args:
        file_list: List of file paths to compress
        output_csv: Optional path to save results CSV
        
    Returns:
        list: All compression results
    """
    import pandas as pd
    
    compressor = BaselineCompressor()
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"Testing {len(file_list)} files with {len(compressor.algorithms)} algorithms")
    print(f"{'='*80}\n")
    
    for i, filepath in enumerate(file_list, 1):
        print(f"[{i}/{len(file_list)}] Processing: {os.path.basename(filepath)}")
        
        if not os.path.exists(filepath):
            print(f"  ⚠️  File not found: {filepath}")
            continue
        
        file_size_kb = os.path.getsize(filepath) / 1024
        print(f"  Size: {file_size_kb:.2f} KB")
        
        for algo in compressor.algorithms.keys():
            result = compressor.compress_file(filepath, algo)
            if result:
                all_results.append(result)
                print(f"    {algo:8s}: {result['compression_ratio']:.2f}x  "
                      f"({result['compressed_size']/1024:.2f} KB, "
                      f"{result['compression_time']:.3f}s)")
        print()
    
    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save to CSV if specified
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"✅ Results saved to: {output_csv}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}\n")
        
        summary = df.groupby('algorithm').agg({
            'compression_ratio': ['mean', 'std', 'min', 'max'],
            'speed_mbps': ['mean', 'std']
        }).round(3)
        
        print(summary)
        print()
        
        return all_results
    else:
        print("⚠️  No results collected!")
        return []


if __name__ == "__main__":
    # Example usage
    print("Baseline Compressor - Testing Module")
    print("=" * 80)
    
    # Test with a few sample files if they exist
    test_files = [
        "data/academic_corpora/canterbury/alice29.txt",
        "data/academic_corpora/canterbury/asyoulik.txt",
        "data/academic_corpora/canterbury/cp.html",
    ]
    
    # Filter to files that exist
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if existing_files:
        print(f"\nFound {len(existing_files)} test files")
        test_on_files(
            existing_files,
            output_csv="results/experiments/baseline_test_12_DEC_2025_12_00_00.csv"
        )
    else:
        print("\n⚠️  No test files found in data/academic_corpora/canterbury/")
        print("Please download the Canterbury corpus first!")