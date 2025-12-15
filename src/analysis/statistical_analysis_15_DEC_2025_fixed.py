"""
Comprehensive Statistical Analysis Module - FIXED (Publication Ready)
Created: December 15, 2025
Purpose: Perform all statistical tests for publication, incorporating rigorous checks.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, kruskal, levene
from statsmodels.stats.power import ttest_power
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.sandbox.stats.multicomp import multipletests
import sys
import os

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for compression experiments"""
    
    def __init__(self, results_df):
        """
        Initialize with results dataframe
        
        Args:
            results_df: DataFrame with columns including 'algorithm', 
                       'compression_ratio', 'speed_mbps', 'memory_mb', 'filename'
        """
        self.df = results_df
        self.algorithms = sorted(self.df['algorithm'].unique())
        print(f"Statistical Analyzer initialized with {len(self.df)} measurements")
        print(f"Algorithms: {', '.join(self.algorithms)}")
        print(f"Files: {self.df['filename'].nunique()}")
    
    def paired_t_tests(self, baseline_algo='gzip', test_algos=None):
        """
        Perform paired t-tests comparing each algorithm to baseline.
        Assumes normality of the difference scores.
        
        Args:
            baseline_algo: Baseline algorithm name
            test_algos: List of algorithms to test (default: all except baseline)
        
        Returns:
            DataFrame: Test results with t-statistic, p-value, effect size
        """
        if test_algos is None:
            test_algos = [a for a in self.algorithms if a != baseline_algo]
        
        results = []
        baseline_data = self.df[self.df['algorithm'] == baseline_algo].sort_values('filename')
        
        for test_algo in test_algos:
            test_data = self.df[self.df['algorithm'] == test_algo].sort_values('filename')
            common_files = set(baseline_data['filename']) & set(test_data['filename'])
            
            baseline_ratios = baseline_data[baseline_data['filename'].isin(common_files)]['compression_ratio'].values
            test_ratios = test_data[test_data['filename'].isin(common_files)]['compression_ratio'].values
            
            if len(common_files) < 2:
                continue 
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(test_ratios, baseline_ratios)
            
            # Effect size (Cohen's d for paired samples)
            diff = test_ratios - baseline_ratios
            std_diff = np.std(diff, ddof=1)
            cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0
            
            mean_improvement = ((np.mean(test_ratios) - np.mean(baseline_ratios)) / 
                              np.mean(baseline_ratios) * 100)
            
            results.append({
                'test_algorithm': test_algo,
                'baseline_algorithm': baseline_algo,
                'n_files': len(common_files),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'mean_test': np.mean(test_ratios),
                'mean_baseline': np.mean(baseline_ratios),
                'improvement_pct': mean_improvement
            })
        
        return pd.DataFrame(results)
    
    def correct_p_values(self, p_values, method='fdr_bh', alpha=0.05):
        """
        Apply multiple testing correction (Bonferroni or FDR).
        
        Args:
            p_values: List, array, or Pandas Series of p-values
            method: 'bonferroni' or 'fdr_bh' (Benjamini/Hochberg for FDR)
            alpha: Significance level
        
        Returns:
            DataFrame: Results including adjusted p-values
        """
        # FIX: Convert Pandas Series to numpy array immediately to eliminate the 
        # "ambiguous truth value" error when checking for emptiness.
        if isinstance(p_values, pd.Series):
            p_values = p_values.values
            
        if len(p_values) == 0:
            return pd.DataFrame({
                'p_value_raw': [],
                'p_value_adjusted': [],
                'reject_H0': [],
                'method': method,
                'n_comparisons': 0
            })
            
        reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)
        
        return pd.DataFrame({
            'p_value_raw': p_values,
            'p_value_adjusted': p_adjusted,
            'reject_H0': reject,
            'method': method,
            'n_comparisons': len(p_values)
        })

    def anova_test(self, metric='compression_ratio'):
        """
        Perform one-way ANOVA across all algorithms (Parametric Test).
        
        Returns:
            dict: ANOVA results including Levene's test for homoscedasticity.
        """
        groups = [self.df[self.df['algorithm'] == algo][metric].values 
                  for algo in self.algorithms if len(self.df[self.df['algorithm'] == algo][metric]) > 1]
        
        if len(groups) < 2:
            return {'metric': metric, 'f_statistic': np.nan, 'p_value': np.nan, 
                    'eta_squared': np.nan, 'levene_p': np.nan, 'error': "Not enough groups (min 2)"}
        
        # 1. Homogeneity of Variance (Levene's Test)
        try:
            levene_stat, levene_p = levene(*groups, center='mean')
        except ValueError:
            levene_p = np.nan # Can fail if groups are too small or identical
        
        # 2. ANOVA F-test
        try:
            f_stat, p_value = f_oneway(*groups)
        except ValueError:
            f_stat, p_value = np.nan, np.nan
        
        # 3. Effect size (eta-squared)
        grand_mean = self.df[metric].mean()
        # Calculate SS_total based only on the groups used for the test
        all_data = np.concatenate(groups)
        ss_total = np.sum((all_data - grand_mean)**2) 
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'metric': metric,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'levene_p': levene_p,
            'n_groups': len(groups),
            'df_between': len(groups) - 1,
            'df_within': len(self.df) - len(groups)
        }

    def kruskal_wallis_test(self, metric='compression_ratio'):
        """
        Perform Kruskal-Wallis H-test (Non-Parametric alternative to ANOVA).
        Used when ANOVA assumptions (e.g., normality) are violated.
        """
        groups = [self.df[self.df['algorithm'] == algo][metric].values 
                  for algo in self.algorithms if len(self.df[self.df['algorithm'] == algo][metric]) > 1]
        
        if len(groups) < 2:
            return {'metric': metric, 'h_statistic': np.nan, 'p_value': np.nan, 'error': "Not enough groups (min 2)"}
        
        h_stat, p_value = kruskal(*groups)
        
        return {
            'metric': metric,
            'h_statistic': h_stat,
            'p_value': p_value,
        }

    def tukey_hsd_posthoc(self, metric='compression_ratio'):
        """
        Perform Tukey's Honestly Significant Difference (HSD) post-hoc test.
        Used after a significant ANOVA to find specific differing pairs.
        
        Returns:
            DataFrame: Tukey's HSD results (mean difference, confidence interval, p-adj)
        """
        data = self.df[self.df['algorithm'].isin(self.algorithms)]
        
        if len(data) < 2 or len(data['algorithm'].unique()) < 2:
             return pd.DataFrame()

        try:
            tukey_results = pairwise_tukeyhsd(endog=data[metric], groups=data['algorithm'], alpha=0.05)
            
            results_df = pd.DataFrame(data=tukey_results._results_table.data[1:], 
                                      columns=tukey_results._results_table.data[0])
            
            results_df.columns = ['group1', 'group2', 'mean_diff', 'lower_ci', 'upper_ci', 'reject_H0', 'p_value_adj']
            return results_df
            
        except ValueError as e:
            # This happens if there is low variance or too few points
            print(f"Tukey HSD failed for {metric}: {e}")
            return pd.DataFrame()


    def bonferroni_correction(self, p_values, alpha=0.05):
        """
        Retained for original compatibility. Use correct_p_values instead.
        """
        # Ensure p_values is not ambiguous before passing it
        if isinstance(p_values, pd.Series):
            p_values = p_values.values
            
        if len(p_values) == 0:
            return {'original_alpha': alpha, 'corrected_alpha': alpha, 'n_comparisons': 0, 'n_significant': 0, 'significant_indices': []}
            
        corrected_df = self.correct_p_values(p_values, method='bonferroni', alpha=alpha)
        
        n_comparisons = corrected_df['n_comparisons'].iloc[0]
        corrected_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha
        
        return {
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'n_comparisons': n_comparisons,
            'n_significant': corrected_df['reject_H0'].sum(),
            'significant_indices': corrected_df.index[corrected_df['reject_H0']].tolist()
        }
    
    def power_analysis(self, effect_size, n_samples, alpha=0.05):
        """
        Calculate statistical power for a two-sample t-test (Cohen's d).
        
        Args:
            effect_size: Effect size (Cohen's d)
            n_samples: Number of samples per group
            alpha: Significance level
        
        Returns:
            float: Statistical power
        """
        try:
            # nobs=n_samples assumes equal sample sizes per group
            power = ttest_power(effect_size, nobs=n_samples, alpha=alpha, alternative='two-sided')
            return power
        except:
            return np.nan # Handle potential errors like non-positive nobs
    
    def effect_size_interpretation(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def comprehensive_analysis(self):
        """
        Perform comprehensive statistical analysis, including assumption checks 
        and post-hoc tests, necessary for publication.
        
        Returns:
            dict: All statistical test results
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS (Publication-Ready)")
        print("="*80)
        
        results = {}
        
        # 1. Descriptive statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 80)
        desc_stats = self.df.groupby('algorithm').agg({
            'compression_ratio': ['mean', 'std', 'min', 'max', 'count'],
            'speed_mbps': ['mean', 'std'],
            'memory_mb': ['mean', 'std']
        }).round(3)
        print(desc_stats)
        results['descriptive'] = desc_stats
        
        # 2. ANOVA and Non-Parametric tests for each metric
        print("\n2. ANOVA AND KRUSKAL-WALLIS TESTS")
        print("-" * 80)
        anova_results = {}
        kruskal_results = {}
        tukey_results = {}
        
        for metric in ['compression_ratio', 'speed_mbps', 'memory_mb']:
            anova_result = self.anova_test(metric)
            kruskal_result = self.kruskal_wallis_test(metric)
            
            if 'error' in anova_result:
                # ASCII FIX: Replaced '🔴 ERROR' with '[!] ERROR'
                print(f"\n[!] ERROR: Skipped {metric} due to insufficient data/groups.")
                continue

            print(f"\n--- {metric.upper()} ---")
            print(f"  ANOVA F-stat: {anova_result['f_statistic']:.3f}, p-value: {anova_result['p_value']:.4e}")
            print(f"  Levene's p (Homogeneity): {anova_result['levene_p']:.4f} (H0: Variances Equal)")
            print(f"  Kruskal H-stat: {kruskal_result['h_statistic']:.3f}, p-value: {kruskal_result['p_value']:.4e}")
            # ASCII FIX: Replaced 'η²' with 'eta^2'
            print(f"  eta^2 (eta-squared): {anova_result['eta_squared']:.3f}")
            
            is_significant = anova_result['p_value'] < 0.05
            is_homoscedastic = anova_result['levene_p'] >= 0.05
            
            if is_significant and is_homoscedastic:
                # ASCII FIX: Replaced '✅' with '[+]'
                print(f"  [+] ANOVA Significant and Assumptions Met. Running Tukey's HSD post-hoc...")
                tukey_df = self.tukey_hsd_posthoc(metric)
                tukey_results[metric] = tukey_df
                # Show only significant pairs
                print(tukey_df[tukey_df['reject_H0'] == True].to_string(index=False))
            elif is_significant:
                 # ASCII FIX: Replaced '⚠️' with '[!]'
                 print(f"  [!] ANOVA Significant, but Levene's Test Failed (p < 0.05). Use caution. Tukey's HSD may be unreliable.")
            elif kruskal_result['p_value'] < 0.05:
                # ASCII FIX: Replaced '⚠️' with '[!]'
                print(f"  [!] Kruskal-Wallis Significant (Non-Parametric). Assumptions for ANOVA likely violated.")
            else:
                 # ASCII FIX: Replaced '❌' with '[x]'
                 print(f"  [x] No significant difference found.")
                
            anova_results[metric] = anova_result
            kruskal_results[metric] = kruskal_result
        
        results['anova'] = anova_results
        results['kruskal'] = kruskal_results
        results['tukey'] = tukey_results
        
        # 3. Pairwise comparisons (all algorithms vs each other) - Focus on Compression Ratio
        print("\n3. PAIRWISE T-TESTS (Compression Ratio)")
        print("-" * 80)
        
        pairwise_results = []
        all_p_values_raw = []
        
        for i, algo1 in enumerate(self.algorithms):
            for algo2 in self.algorithms[i+1:]:
                # Data preparation (identical to original code)
                data1 = self.df[self.df['algorithm'] == algo1].sort_values('filename')
                data2 = self.df[self.df['algorithm'] == algo2].sort_values('filename')
                common_files = set(data1['filename']) & set(data2['filename'])
                ratios1 = data1[data1['filename'].isin(common_files)]['compression_ratio'].values
                ratios2 = data2[data2['filename'].isin(common_files)]['compression_ratio'].values
                
                if len(common_files) < 2: continue

                # Paired t-test
                t_stat, p_value = stats.ttest_rel(ratios1, ratios2)
                all_p_values_raw.append(p_value)
                
                # Effect size (Cohen's d)
                diff = ratios1 - ratios2
                std_diff = np.std(diff, ddof=1)
                cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0
                
                pairwise_results.append({
                    'algorithm_1': algo1,
                    'algorithm_2': algo2,
                    't_statistic': t_stat,
                    'p_value_raw': p_value,
                    'cohens_d': cohens_d,
                    'effect_size': self.effect_size_interpretation(cohens_d)
                })
        
        pairwise_results_df = pd.DataFrame(pairwise_results)
        
        # Apply multiple comparison correction (Bonferroni and FDR)
        bonf_corrected_df = self.correct_p_values(pairwise_results_df['p_value_raw'], method='bonferroni')
        fdr_corrected_df = self.correct_p_values(pairwise_results_df['p_value_raw'], method='fdr_bh')
        
        pairwise_results_df['p_value_bonf_adj'] = bonf_corrected_df['p_value_adjusted']
        pairwise_results_df['p_value_fdr_adj'] = fdr_corrected_df['p_value_adjusted']
        
        results['pairwise'] = pairwise_results_df
        
        # Summary of Correction
        print(f"\nMultiple Comparison Correction (Pairwise T-tests):")
        print(f"  Total Comparisons: {len(all_p_values_raw)}")
        if not bonf_corrected_df.empty:
            print(f"  Significant (Bonf.): {bonf_corrected_df['reject_H0'].sum()}/{len(all_p_values_raw)}")
            print(f"  Significant (FDR-BH): {fdr_corrected_df['reject_H0'].sum()}/{len(all_p_values_raw)}")
        
        # 4. Power analysis
        print("\n4. POWER ANALYSIS")
        print("-" * 80)
        
        n_files = self.df['filename'].nunique()
        print(f"Sample size per paired test (max n files): {n_files}")
        
        # Power for different Cohen's d (t-test)
        typical_effect_sizes = [0.2, 0.5, 0.8]
        print(f"\nPower for different effect sizes (Paired T-test, alpha=0.05):")
        for es in typical_effect_sizes:
            power = self.power_analysis(es, n_files)
            print(f"  Cohen's d = {es:.1f} ({self.effect_size_interpretation(es)}): Power = {power:.3f} ({power*100:.1f}%)")
            
        # Power for observed effect sizes - Compression Ratio
        print(f"\nPower for observed effect sizes (Compression Ratio):")
        
        # Calculate power for observed effect sizes in pairwise results
        for i, row in enumerate(pairwise_results_df.head(5).itertuples()):
            power = self.power_analysis(abs(row.cohens_d), n_files)
            print(f"  {row.algorithm_1} vs {row.algorithm_2}: d={row.cohens_d:.3f}, power={power:.3f}")
        
        return results

def create_publication_tables(stats_results, multi_obj_df, output_dir='results/tables'):
    """
    Create all publication tables, incorporating rigorous statistical outputs.
    
    Args:
        stats_results: Results from comprehensive_analysis()
        multi_obj_df: Multi-objective results DataFrame
        output_dir: Directory to save tables
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CREATING PUBLICATION TABLES (Phase 2)")
    print("="*80)
    
    # TABLE 1: Overall Performance Summary (Unchanged)
    # ASCII FIX: Replaced '📊' with '[TABLE]' and '✅' with '[+]'
    print("\n[TABLE] Table 1: Overall Performance Summary")
    table1 = stats_results['descriptive'].reset_index()
    table1.columns = ['Algorithm', 'Ratio_Mean', 'Ratio_Std', 'Ratio_Min', 'Ratio_Max', 'N_Files', 
                      'Speed_Mean', 'Speed_Std', 'Memory_Mean', 'Memory_Std']
    
    table1['Compression Ratio'] = table1.apply(lambda x: f"{x['Ratio_Mean']:.2f} ± {x['Ratio_Std']:.2f}", axis=1)
    table1['Speed (MB/s)'] = table1.apply(lambda x: f"{x['Speed_Mean']:.1f} ± {x['Speed_Std']:.1f}", axis=1)
    table1['Memory (MB)'] = table1.apply(lambda x: f"{x['Memory_Mean']:.2f} ± {x['Memory_Std']:.2f}", axis=1)
    
    table1_final = table1[['Algorithm', 'Compression Ratio', 'Speed (MB/s)', 'Memory (MB)', 'N_Files']]
    table1_final.to_csv(f'{output_dir}/table1_overall_performance_300plus_files.csv', index=False)
    print("[+] Saved: table1_overall_performance.csv")
    print(table1_final.to_string(index=False))
    
    # TABLE 2: Statistical Tests (Paired Comparisons) - UPDATED
    # ASCII FIX: Replaced '📊' with '[TABLE]' and '✅' with '[+]'
    print("\n[TABLE] Table 2: Pairwise Statistical Comparisons (Compression Ratio)")
    
    table2 = stats_results['pairwise'][[
        'algorithm_1', 'algorithm_2', 't_statistic', 'p_value_raw', 
        'p_value_bonf_adj', 'p_value_fdr_adj', 'cohens_d', 'effect_size'
    ]]
    table2 = table2.round({'t_statistic': 3, 'p_value_raw': 4, 'p_value_bonf_adj': 4, 'p_value_fdr_adj': 4, 'cohens_d': 3})
    
    # Add significance flags based on Bonferroni correction
    table2['Sig. (Bonf)'] = table2['p_value_bonf_adj'].apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else '')))
    
    table2.columns = ['Algo 1', 'Algo 2', 't', 'p (raw)', 'p (Bonf. adj)', 'p (FDR adj)', "Cohen's d", 'Effect Size', 'Sig.']
    
    table2_final = table2.head(20).copy() # Limit printout size
    
    table2_final.to_csv(f'{output_dir}/table2_statistical_tests_300plus_files.csv', index=False)
    print("[+] Saved: table2_statistical_tests.csv (Includes Adjusted P-values)")
    print(table2_final.to_string(index=False))
    
    # TABLE 3: ANOVA Summary - UPDATED (Incorporating Assumptions/Non-Parametric)
    # ASCII FIX: Replaced '📊' with '[TABLE]' and '✅' with '[+]'
    print("\n[TABLE] Table 3: ANOVA and Non-Parametric Summary")
    table3_data = []
    
    for metric in ['compression_ratio', 'speed_mbps', 'memory_mb']:
        anova_result = stats_results['anova'][metric]
        kruskal_result = stats_results['kruskal'][metric]
        
        # Determine the primary significance based on ANOVA/Kruskal
        sig_flag = ''
        if anova_result['p_value'] < 0.05:
            sig_flag = 'V (ANOVA)'
        elif kruskal_result['p_value'] < 0.05:
            sig_flag = 'V (Kruskal)'
            
        table3_data.append({
            'Metric': metric,
            'Levene\'s p (H0: Homogeneity)': f"{anova_result['levene_p']:.4f}",
            'ANOVA F(df)': f"{anova_result['f_statistic']:.3f} ({anova_result['df_between']}, {anova_result['df_within']})",
            'ANOVA p': f"{anova_result['p_value']:.4e}",
            'ANOVA eta^2': f"{anova_result['eta_squared']:.3f}",
            'Kruskal H': f"{kruskal_result['h_statistic']:.3f}",
            'Kruskal p': f"{kruskal_result['p_value']:.4e}",
            'Significant': sig_flag
        })
    
    table3 = pd.DataFrame(table3_data)
    table3.to_csv(f'{output_dir}/table3_anova_summary_300plus_files.csv', index=False)
    print("[+] Saved: table3_anova_summary.csv (Includes Levene's and Kruskal's results)")
    print(table3.to_string(index=False))
    
    # TABLE 4: Corpus-Specific Results (Unchanged, but robust to new data)
    # ASCII FIX: Replaced '📊' with '[TABLE]' and '✅' with '[+]'
    print("\n[TABLE] Table 4: Performance by Corpus")
    
    # Add corpus labels to data
    def get_corpus(filename):
        if any(x in filename for x in ['alice', 'asyoulik', 'cp.html', 'fields', 
                                       'grammar', 'kennedy', 'lcet', 'plrabn', 
                                       'ptt5', 'sum', 'xargs']):
            return 'Canterbury'
        else:
            return 'Heterogeneous'
    
    multi_obj_df['corpus'] = multi_obj_df['filename'].apply(get_corpus)
    
    table4 = multi_obj_df.groupby(['corpus', 'algorithm']).agg({
        'compression_ratio': ['mean', 'std', 'count']
    }).round(3)
    
    table4.columns = ['Mean_Ratio', 'Std_Ratio', 'N_Files']
    table4 = table4.reset_index()
    table4['Ratio'] = table4.apply(lambda x: f"{x['Mean_Ratio']:.2f} ± {x['Std_Ratio']:.2f}", axis=1)
    
    table4_pivot = table4.pivot(index='algorithm', columns='corpus', values='Ratio')
    table4_pivot = table4_pivot.reset_index()
    table4_pivot.columns.name = None
    
    table4_pivot.to_csv(f'{output_dir}/table4_corpus_specific_300plus_files.csv', index=False)
    print("[+] Saved: table4_corpus_specific.csv")
    print(table4_pivot.to_string(index=False))
    
    # TABLE 5: Tukey's HSD Post-Hoc Results (New Table for Publication)
    # ASCII FIX: Replaced '📊' with '[TABLE]' and '✅' with '[+]'
    print("\n[TABLE] Table 5: Tukey's HSD Post-Hoc Tests (Only Significant Results)")
    
    tukey_list = []
    for metric, df_tukey in stats_results['tukey'].items():
        if not df_tukey.empty:
            df_sig = df_tukey[df_tukey['reject_H0'] == True].copy()
            if not df_sig.empty:
                df_sig.insert(0, 'Metric', metric)
                df_sig['mean_diff'] = df_sig['mean_diff'].round(3)
                df_sig['p_value_adj'] = df_sig['p_value_adj'].round(4)
                df_sig = df_sig[['Metric', 'group1', 'group2', 'mean_diff', 'p_value_adj', 'reject_H0']]
                tukey_list.append(df_sig)
    
    if tukey_list:
        table5 = pd.concat(tukey_list, ignore_index=True)
        table5.columns = ['Metric', 'Group 1', 'Group 2', 'Mean Diff', 'p (adj)', 'Significant']
        table5.to_csv(f'{output_dir}/table5_tukey_hsd_significant.csv', index=False)
        print("[+] Saved: table5_tukey_hsd_significant.csv")
        print(table5.to_string(index=False))
    else:
        # ASCII FIX: Replaced '⚠️' with '[!]'
        print("[!] No significant Tukey's HSD results found to create Table 5.")
        table5 = pd.DataFrame(columns=['Metric', 'Group 1', 'Group 2', 'Mean Diff', 'p (adj)', 'Significant'])

    # ASCII FIX: Replaced '✅' with '[+]'
    print("\n[+] All tables created successfully!")
    
    return {
        'table1': table1_final,
        'table2': table2_final,
        'table3': table3,
        'table4': table4_pivot,
        'table5': table5
    }


if __name__ == "__main__":
    # Assuming the input file exists in the correct location
    try:
        # Load multi-objective results
        df = pd.read_csv('results/experiments/multi_objective_all_300plusfiles_complete_15_DEC_2025.csv')
    except FileNotFoundError:
        print("ERROR: CSV file not found. Please ensure 'results/experiments/multi_objective_all_57files_complete_15_DEC_2025.csv' exists.")
        sys.exit(1)
        
    # Run statistical analysis
    analyzer = StatisticalAnalyzer(df)
    stats_results = analyzer.comprehensive_analysis()
    
    # Create publication tables
    tables = create_publication_tables(stats_results, df)
    
    print("\n" + "="*80)
    # ASCII FIX: Replaced '✅' with '[+]'
    print("[+] STATISTICAL ANALYSIS COMPLETE! (Phases 1 & 2 Code Fixed)")
    print("Next: Phase 3 - Manuscript Integration & Journal Compliance.")
    print("="*80)