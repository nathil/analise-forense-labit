import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ChartGenerator:
    """
    Generates analytical charts for CBIR pipeline evaluation.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply standard seaborn theme
        sns.set_theme(style="whitegrid", palette="muted")
        
    def generate_all_charts(self, csv_path: str, json_path: str, test_name: str):
        print(f"\n[+] Generating charts for {test_name}...")
        df = pd.read_csv(csv_path)
        with open(json_path, 'r') as f:
            meta = json.load(f)
            
        self._plot_cmc_curve(df, test_name, top_k=meta.get('top_k', 5))
        self._plot_distance_density(df, test_name)
        self._plot_execution_times(meta, test_name)
        
        # Radar chart if performance metrics exist (i.e. full pipeline, not just reduction analyzer)
        if "performance_metrics" in meta:
            self._plot_metrics_radar(meta, test_name)
            self._plot_confusion_matrix(meta, test_name)
            
        print(f"    Charts saved to {self.output_dir}")
            
    def _plot_confusion_matrix(self, meta: dict, test_name: str):
        cm = meta.get("confusion_matrix", {})
        if not cm:
            return
            
        tp = cm.get("tp", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tn = cm.get("tn", 0)
        
        # 2x2 matrix: [[TP, FP], [FN, TN]]
        # In standard layout it is often: 
        # Predicted Positive, Predicted Negative
        # Actual Positive [ TP, FN ]
        # Actual Negative [ FP, TN ]
        matrix = np.array([[tp, fn], [fp, tn]])
        
        plt.figure(figsize=(7, 5))
        # Use a logarithmic color scale because TN is usually massive compared to others
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=1, vmax=matrix.max() + 1)
        
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', norm=norm,
                    xticklabels=['Predicted Match', 'Predicted Mismatch'],
                    yticklabels=['Actual Match', 'Actual Mismatch'])
                    
        plt.title(f"Confusion Matrix - {test_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{test_name}_confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_cmc_curve(self, df: pd.DataFrame, test_name: str, top_k: int):
        cmc_scores = []
        ranks = list(range(1, top_k + 1))
        
        probes = df['probe_image'].unique()
        num_probes = len(probes)
        
        for k in ranks:
            # Drop duplicates keeping the first occurrence (which is the lowest rank)
            # This counts if there is ANY correct match at rank <= k
            correct_at_k = df[(df['match_rank'] <= k) & (df['is_correct'])].drop_duplicates(subset=['probe_image'])
            cmc_scores.append((len(correct_at_k) / num_probes * 100) if num_probes > 0 else 0)
            
        plt.figure(figsize=(8, 5))
        plt.plot(ranks, cmc_scores, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
        plt.fill_between(ranks, cmc_scores, alpha=0.2, color='#2ca02c')
        plt.title(f"Cumulative Match Characteristic (CMC) - {test_name}", fontsize=14)
        plt.xlabel("Rank", fontsize=12)
        plt.ylabel("Recognition Rate (%)", fontsize=12)
        plt.xticks(ranks)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{test_name}_cmc_curve.png"), dpi=150)
        plt.close()
        
    def _plot_distance_density(self, df: pd.DataFrame, test_name: str):
        plt.figure(figsize=(10, 6))
        
        tp_dist = df[df['is_correct'] == True]['distance']
        fp_dist = df[df['is_correct'] == False]['distance']
        
        if len(tp_dist) > 0:
            sns.kdeplot(tp_dist, fill=True, color="blue", label="True Matches", alpha=0.5)
        if len(fp_dist) > 0:
            sns.kdeplot(fp_dist, fill=True, color="red", label="False Matches", alpha=0.5)
            
        plt.title(f"Distance Density Distribution - {test_name}", fontsize=14)
        plt.xlabel("Distance", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{test_name}_distance_dist.png"), dpi=150)
        plt.close()
        
    def _plot_execution_times(self, meta: dict, test_name: str):
        timers = meta.get("timers", {})
        if not timers:
            return
            
        # Sort timers by value
        sorted_timers = dict(sorted(timers.items(), key=lambda item: item[1]))
        stages = list(sorted_timers.keys())
        times = list(sorted_timers.values())
        
        plt.figure(figsize=(10, 5))
        bars = plt.barh(stages, times, color=sns.color_palette("viridis", len(stages)))
        plt.title(f"Execution Times by Stage - {test_name}", fontsize=14)
        plt.xlabel("Time (seconds)", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add labels on bars
        for bar in bars:
            width = bar.get_width()
            # Handle text position safely
            plt.text(width + (max(times)*0.01 if max(times) > 0 else 0.01), 
                     bar.get_y() + bar.get_height()/2, 
                     f'{width:.4f}s', va='center', fontsize=10)
                     
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{test_name}_execution_times.png"), dpi=150)
        plt.close()
        
    def _plot_metrics_radar(self, meta: dict, test_name: str):
        perf = meta.get("performance_metrics", {})
        if not perf:
            return
            
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "mAP"]
        values = [perf.get(m, 0) * 100 for m in metrics_to_plot]  # Convert to percentage
        
        N = len(metrics_to_plot)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1] # Close the loop
        values += values[:1] # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        plt.xticks(angles[:-1], [m.upper() for m in metrics_to_plot], color='grey', size=11)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
        plt.ylim(0, 100)
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
        ax.fill(angles, values, '#1f77b4', alpha=0.25)
        
        plt.title(f"Performance Metrics Radar - {test_name}", size=15, y=1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{test_name}_metrics_radar.png"), dpi=150)
        plt.close()
