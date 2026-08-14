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
    def __init__(self, output_dir: str = "results/charts"):
        self.output_dir = output_dir
        self.pipeline_dir = os.path.join(self.output_dir, "pipeline")
        self.multiple_pipelines_dir = os.path.join(self.output_dir, "multiple-pipelines")
        self.query_dir = os.path.join(self.output_dir, "query")
        self.grid_search_dir = os.path.join(self.output_dir, "grid-search")
        os.makedirs(self.pipeline_dir, exist_ok=True)
        os.makedirs(self.multiple_pipelines_dir, exist_ok=True)
        os.makedirs(self.query_dir, exist_ok=True)
        os.makedirs(self.grid_search_dir, exist_ok=True)
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
            self._plot_metrics_radar(meta, test_name, self.pipeline_dir)
            self._plot_confusion_matrix(meta, test_name, self.pipeline_dir)
            
        print(f"    Charts saved to {self.pipeline_dir}")
            
    def _plot_confusion_matrix(self, meta: dict, test_name: str, out_dir: str = None):
        if out_dir is None: out_dir = self.pipeline_dir
        cm = meta.get("confusion_matrix", {})
        if not cm:
            return
            
        tp = cm.get("tp", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tn = cm.get("tn", 0)
        
        matrix = np.array([[tp, fn], [fp, tn]])
        
        plt.figure(figsize=(7, 5))
        import matplotlib.colors as mcolors
        norm = mcolors.LogNorm(vmin=1, vmax=matrix.max() + 1)
        
        sns.heatmap(matrix + 1, annot=matrix, fmt='d', cmap='Blues', norm=norm,
                    xticklabels=['Predicted Match', 'Predicted Mismatch'],
                    yticklabels=['Actual Match', 'Actual Mismatch'])
                    
        plt.title(f"Confusion Matrix - {test_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{test_name}_confusion_matrix.png"), dpi=150)
        plt.close()

    def _plot_cmc_curve(self, df: pd.DataFrame, test_name: str, top_k: int, out_dir: str = None):
        if out_dir is None: out_dir = self.pipeline_dir
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
        plt.savefig(os.path.join(out_dir, f"{test_name}_cmc_curve.png"), dpi=150)
        plt.close()
        
    def _plot_distance_density(self, df: pd.DataFrame, test_name: str, out_dir: str = None):
        if out_dir is None: out_dir = self.pipeline_dir
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
        plt.savefig(os.path.join(out_dir, f"{test_name}_distance_dist.png"), dpi=150)
        plt.close()
        
    def _plot_execution_times(self, meta: dict, test_name: str, out_dir: str = None):
        if out_dir is None: out_dir = self.pipeline_dir
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
        plt.savefig(os.path.join(out_dir, f"{test_name}_execution_times.png"), dpi=150)
        plt.close()
        
    def _plot_metrics_radar(self, meta: dict, test_name: str, out_dir: str = None):
        if out_dir is None: out_dir = self.pipeline_dir
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
        plt.savefig(os.path.join(out_dir, f"{test_name}_metrics_radar.png"), dpi=150)
        plt.close()

    def generate_comparative_charts(self, batch_name: str, test_names: list, csv_paths: list, json_paths: list):
        print(f"\n[+] Generating comparative charts for {batch_name}...")
        
        dfs = []
        metas = []
        for csv_file, json_file in zip(csv_paths, json_paths):
            dfs.append(pd.read_csv(csv_file))
            with open(json_file, 'r') as f:
                metas.append(json.load(f))
                
        # 1. Comparative CMC Curve
        plt.figure(figsize=(8, 5))
        ranks = list(range(1, metas[0].get('top_k', 5) + 1))
        
        for df, name in zip(dfs, test_names):
            cmc_scores = []
            probes = df['probe_image'].unique()
            num_probes = len(probes)
            for k in ranks:
                correct_at_k = df[(df['match_rank'] <= k) & (df['is_correct'])].drop_duplicates(subset=['probe_image'])
                cmc_scores.append((len(correct_at_k) / num_probes * 100) if num_probes > 0 else 0)
            plt.plot(ranks, cmc_scores, marker='o', linestyle='-', linewidth=2, label=name)
            
        plt.title(f"Comparative CMC Curve - {batch_name}", fontsize=14, pad=20)
        plt.xlabel("Rank", fontsize=12)
        plt.ylabel("Recognition Rate (%)", fontsize=12)
        plt.xticks(ranks)
        plt.ylim(0, 105)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right', fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_cmc_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Comparative Radar
        metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "mAP"]
        N = len(metrics_to_plot)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], [m.upper() for m in metrics_to_plot], color='grey', size=11)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=8)
        plt.ylim(0, 100)
        
        for meta, name in zip(metas, test_names):
            perf = meta.get("performance_metrics", {})
            if perf:
                values = [perf.get(m, 0) * 100 for m in metrics_to_plot]
                values += values[:1]
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=name)
                ax.fill(angles, values, alpha=0.1)
                
        plt.title(f"Comparative Performance Radar - {batch_name}", size=15, y=1.2)
        plt.legend(loc='upper right', fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_metrics_radar.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Grouped Bar Execution Times
        time_data = []
        for meta, name in zip(metas, test_names):
            timers = meta.get("timers", {})
            for stage, time_val in timers.items():
                time_data.append({"Pipeline": name, "Stage": stage, "Time": time_val})
                
        if time_data:
            df_times = pd.DataFrame(time_data)
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df_times, y='Stage', x='Time', hue='Pipeline')
            plt.title(f"Comparative Execution Times - {batch_name}", fontsize=14, pad=20)
            plt.xlabel("Time (seconds)", fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.legend(loc='upper right', fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_execution_times.png"), dpi=150, bbox_inches='tight')
            plt.close()
            
        # 4. Confusion Matrix Subplots (Stacked vertically)
        num_pipelines = len(test_names)
        fig, axes = plt.subplots(num_pipelines, 1, figsize=(7, 5 * num_pipelines))
        if num_pipelines == 1: axes = [axes]
        import matplotlib.colors as mcolors
        
        for i, (meta, name) in enumerate(zip(metas, test_names)):
            cm = meta.get("confusion_matrix", {})
            if not cm: continue
            tp = cm.get("tp", 0)
            fp = cm.get("fp", 0)
            fn = cm.get("fn", 0)
            tn = cm.get("tn", 0)
            matrix = np.array([[tp, fn], [fp, tn]])
            norm = mcolors.LogNorm(vmin=1, vmax=matrix.max() + 1)
            sns.heatmap(matrix + 1, annot=matrix, fmt='d', cmap='Blues', norm=norm,
                        xticklabels=['Predicted Match', 'Predicted Mismatch'],
                        yticklabels=['Actual Match', 'Actual Mismatch'], ax=axes[i])
            axes[i].set_title(f"{name}", fontsize=12)
            
        plt.suptitle(f"Comparative Confusion Matrices - {batch_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_confusion_matrix.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 5. Distance Density (Overlaid)
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(test_names))
        for i, (df, name) in enumerate(zip(dfs, test_names)):
            tp_dist = df[df['is_correct'] == True]['distance']
            fp_dist = df[df['is_correct'] == False]['distance']
            
            base_color = colors[i]
            # Lighter for true
            light_c = tuple(min(1.0, c + 0.15) for c in base_color)
            # Darker for false
            dark_c = tuple(max(0.0, c - 0.15) for c in base_color)
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(tp_dist) > 0:
                    sns.kdeplot(tp_dist, fill=True, color=light_c, label=f"True - {name}", alpha=0.3)
                if len(fp_dist) > 0:
                    sns.kdeplot(fp_dist, fill=False, color=dark_c, label=f"False - {name}", linewidth=2)
                
        plt.title(f"Comparative Distance Density (Overlaid) - {batch_name}", fontsize=14, pad=20)
        plt.xlabel("Distance", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(loc='upper right', fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_distance_dist_overlaid.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # 6. Distance Density (Subplots Stacked)
        fig, axes = plt.subplots(num_pipelines, 1, figsize=(10, 5 * num_pipelines))
        if num_pipelines == 1: axes = [axes]
        for i, (df, name) in enumerate(zip(dfs, test_names)):
            tp_dist = df[df['is_correct'] == True]['distance']
            fp_dist = df[df['is_correct'] == False]['distance']
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if len(tp_dist) > 0:
                    sns.kdeplot(tp_dist, fill=True, color="blue", label="True Matches", alpha=0.5, ax=axes[i])
                if len(fp_dist) > 0:
                    sns.kdeplot(fp_dist, fill=True, color="red", label="False Matches", alpha=0.5, ax=axes[i])
            axes[i].set_title(f"{name}", fontsize=12)
            axes[i].legend()
            
        plt.suptitle(f"Comparative Distance Density (Subplots) - {batch_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.multiple_pipelines_dir, f"{batch_name}_distance_dist_subplots.png"), dpi=150, bbox_inches='tight')
        plt.close()

    def generate_single_query_chart(self, batch_name: str, query_info: dict, pipeline_results: dict):
        """
        Generates a visual comparison of single query results across pipelines.
        query_info = {"query_image_path": str, "query_label": str}
        pipeline_results = { "pipe1": [{"match_image_path": str, "distance": float, "is_correct": bool}], ... }
        """
        import matplotlib.image as mpimg
        
        num_pipelines = len(pipeline_results)
        if num_pipelines == 0:
            return
            
        first_val = list(pipeline_results.values())[0]
        if isinstance(first_val, dict) and "matches" in first_val:
            top_k = len(first_val["matches"])
        else:
            top_k = len(first_val)
        
        fig, axes = plt.subplots(num_pipelines, top_k + 1, figsize=(3 * (top_k + 1), 3 * num_pipelines))
        if num_pipelines == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, (pipe_name, pipe_data) in enumerate(pipeline_results.items()):
            if isinstance(pipe_data, dict) and "matches" in pipe_data:
                matches = pipe_data["matches"]
            else:
                matches = pipe_data
            # Query Image
            ax_q = axes[i, 0]
            try:
                img_q = mpimg.imread(query_info['query_image_path'])
                ax_q.imshow(img_q)
            except Exception:
                ax_q.text(0.5, 0.5, "Image not found", ha='center')
            
            # Truncate pipeline name if too long
            short_name = pipe_name[:30] + '...' if len(pipe_name) > 30 else pipe_name
            ax_q.set_title(f"Query\n({query_info['query_label']})", fontsize=10)
            ax_q.axis('off')
            ax_q.text(-0.1, 0.5, short_name, transform=ax_q.transAxes, fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
            
            # Matches
            for j in range(top_k):
                ax_m = axes[i, j + 1]
                if j < len(matches):
                    match = matches[j]
                    try:
                        img_m = mpimg.imread(match['match_image_path'])
                        ax_m.imshow(img_m)
                    except Exception:
                        ax_m.text(0.5, 0.5, "Not found", ha='center')
                        
                    color = 'green' if match['is_correct'] else 'red'
                    ax_m.set_title(f"Rank {j+1}\nDist: {match['distance']:.4f}", fontsize=10, color=color)
                    
                    # Add colored border
                    for spine in ax_m.spines.values():
                        spine.set_edgecolor(color)
                        spine.set_linewidth(3)
                        spine.set_visible(True)
                    ax_m.set_xticks([])
                    ax_m.set_yticks([])
                else:
                    ax_m.axis('off')
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.query_dir, f"{batch_name}_query_results.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Single Query Chart saved to {self.query_dir}")

    def generate_grid_search_charts(self, batch_name: str, summary_df: pd.DataFrame):
        """
        Generates high-density comparative charts specifically designed for Grid Search evaluation.
        """
        print(f"\n[+] Generating Grid Search analytical charts for {batch_name}...")
        if summary_df.empty:
            print("[-] Warning: Summary dataframe is empty, skipping charts.")
            return

        # 1. Leaderboard (Top 15 Pipelines by mAP)
        top_n = min(15, len(summary_df))
        df_top = summary_df.head(top_n).copy()
        
        plt.figure(figsize=(12, max(6, top_n * 0.45)))
        # Reverse order for horizontal plot so #1 is at the top
        df_plot = df_top.iloc[::-1]
        
        colors = ["#3B82F6" if i != len(df_plot) - 1 else "#F59E0B" for i in range(len(df_plot))]
        bars = plt.barh(df_plot["pipeline_name"], df_plot["mAP"], color=colors, edgecolor="black", alpha=0.85)
        
        for bar, val, top1 in zip(bars, df_plot["mAP"], df_plot["top1_accuracy"]):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f"mAP: {val:.3f} (Top-1: {top1*100:.1f}%)",
                     va='center', fontsize=9, fontweight='bold')
                     
        best_pipe = summary_df.iloc[0]
        plt.title(f"Grid Search Leaderboard (Top {top_n} Pipelines)\nBest: {best_pipe['pipeline_name']} (mAP: {best_pipe['mAP']:.4f})",
                  fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Mean Average Precision (mAP)", fontsize=12)
        plt.xlim(0, max(0.1, summary_df["mAP"].max() * 1.25))
        plt.tight_layout()
        plt.savefig(os.path.join(self.grid_search_dir, f"{batch_name}_leaderboard_top_pipelines.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Extractor x Reducer Heatmap
        if len(summary_df["extractor"].unique()) >= 1 and len(summary_df["reducer"].unique()) >= 1:
            plt.figure(figsize=(9, 6))
            pivot_map = summary_df.pivot_table(index="extractor", columns="reducer", values="mAP", aggfunc="mean")
            sns.heatmap(pivot_map, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Average mAP'}, linewidths=0.5)
            plt.title(f"Component Interaction: Mean mAP by Extractor & Reducer\n{batch_name}", fontsize=13, fontweight='bold', pad=15)
            plt.ylabel("Feature Extractor", fontsize=11)
            plt.xlabel("Dimensionality Reducer", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(self.grid_search_dir, f"{batch_name}_extractor_reducer_heatmap.png"), dpi=150, bbox_inches='tight')
            plt.close()

        # 3. Execution Time vs Performance Trade-off (Pareto / Scatter)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=summary_df,
            x="total_time",
            y="mAP",
            hue="extractor",
            style="reducer",
            s=120,
            alpha=0.85
        )
        
        # Highlight best pipeline with star
        plt.scatter(
            best_pipe["total_time"],
            best_pipe["mAP"],
            color="gold",
            edgecolor="black",
            s=300,
            marker="*",
            label=f"★ Best ({best_pipe['pipeline_name'][:25]}...)",
            zorder=10
        )
        
        plt.title(f"Execution Time vs Performance Trade-off\n{batch_name}", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Total Execution Time (seconds)", fontsize=11)
        plt.ylabel("Mean Average Precision (mAP)", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(self.grid_search_dir, f"{batch_name}_time_vs_accuracy_tradeoff.png"), dpi=150, bbox_inches='tight')
        plt.close()

        # 4. Metric Distribution by Extractor
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=summary_df, x="extractor", y="mAP", hue="extractor", legend=False, palette="Set2", boxprops=dict(alpha=0.7))
        sns.stripplot(data=summary_df, x="extractor", y="mAP", color="black", alpha=0.5, jitter=0.2, size=6)
        plt.title(f"Performance Distribution by Extractor Architecture\n{batch_name}", fontsize=13, fontweight='bold', pad=15)
        plt.xlabel("Feature Extractor", fontsize=11)
        plt.ylabel("Mean Average Precision (mAP)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.grid_search_dir, f"{batch_name}_metric_distributions_by_extractor.png"), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"    Grid Search Charts saved to {self.grid_search_dir}")

    def generate_pipeline_journey_chart(self, journey_data: dict, output_path: str):
        """
        Generates a step-by-step dashboard for a single query.
        """
        import matplotlib.patches as patches
        import cv2
        
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 3)
        
        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(journey_data["raw_image"])
        ax1.set_title("1. Original Image")
        ax1.axis("off")
        
        # 2. Preprocessed Image
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(journey_data["cropped_image"])
        ax2.set_title("2. Preprocessed (Cropped)")
        ax2.axis("off")
        
        # 3. Feature Vector Heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        vec = journey_data["feature_vector"]
        width = int(np.ceil(np.sqrt(len(vec))))
        pad = width**2 - len(vec)
        vec_padded = np.pad(vec, (0, pad), 'constant')
        vec_2d = vec_padded.reshape((width, width))
        im3 = ax3.imshow(vec_2d, cmap='viridis', aspect='auto')
        ax3.set_title(f"3. Raw Features (Dim: {len(vec)})")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        # 4. Reduced Space Projection
        ax4 = fig.add_subplot(gs[1, 0:2])
        red_gal = journey_data["reduced_gallery"]
        labels = journey_data.get("gallery_labels", None)
        query_red = journey_data["reduced_vector"]
        
        if red_gal is not None and len(red_gal) > 0:
            if red_gal.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                red_gal_2d = pca.fit_transform(red_gal)
                query_red_2d = pca.transform(query_red.reshape(1, -1))[0]
            else:
                red_gal_2d = red_gal[:, :2]
                query_red_2d = query_red[:2]
                
            if labels is not None:
                ax4.scatter(red_gal_2d[:, 0], red_gal_2d[:, 1], c=labels, cmap='tab20', alpha=0.5, s=20)
            else:
                ax4.scatter(red_gal_2d[:, 0], red_gal_2d[:, 1], c='gray', alpha=0.5, s=20)
                
            ax4.scatter(query_red_2d[0], query_red_2d[1], c='red', marker='*', s=300, edgecolor='black', label="Query")
            ax4.legend()
            
        ax4.set_title(f"4. Reduced Space (Dim: {len(query_red)})")
        
        # 5. Top Matches
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis("off")
        ax5.set_title("5. Top Matches", pad=20)
        
        matches = journey_data["top_matches"]
        n_matches = min(len(matches), 5)
        if n_matches > 0:
            match_gs = gs[1, 2].subgridspec(1, n_matches)
            for i in range(n_matches):
                match_meta, dist = matches[i]
                match_ax = fig.add_subplot(match_gs[0, i])
                try:
                    m_img = cv2.imread(match_meta['image_path'])
                    m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
                    match_ax.imshow(m_img)
                except:
                    match_ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                    
                match_ax.axis("off")
                match_ax.set_title(f"Rank {i+1}\nD: {dist:.2f}", fontsize=9)
                
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
