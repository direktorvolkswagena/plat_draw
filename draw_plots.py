import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class PlotDrawer:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)

    def draw_plots(self):
        data = pd.read_json(self.json_path)

        plot_paths = []

        # Scatter Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="gt_corners", y="rb_corners", data=data)
        plt.xlabel("Ground Truth Corners")
        plt.ylabel("Predicted Corners")
        plt.title("Ground Truth vs. Predicted Corners")
        gt_rb_plot_path = os.path.join(self.plots_dir, "gt_vs_rb_corners.png")
        plt.savefig(gt_rb_plot_path)
        plt.close()
        plot_paths.append(gt_rb_plot_path)

        # Box Plot
        deviation_columns = ['mean', 'max', 'min', 'floor_mean', 'floor_max', 'floor_min', 'ceiling_mean',
                             'ceiling_max', 'ceiling_min']
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=data[deviation_columns])
        plt.xticks(rotation=45)
        plt.title("Box Plot of Deviation Metrics")
        box_plot_path = os.path.join(self.plots_dir, "deviation_box_plot.png")
        plt.savefig(box_plot_path)
        plt.close()
        plot_paths.append(box_plot_path)

        # Correlation Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = data[deviation_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Correlation Heatmap of Deviation Metrics")
        heatmap_path = os.path.join(self.plots_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        plot_paths.append(heatmap_path)

        # Pair Plots
        pair_columns = ['mean', 'floor_mean', 'ceiling_mean']
        pair_plot = sns.pairplot(data[pair_columns], diag_kind="kde")
        pair_plot_path = os.path.join(self.plots_dir, "pair_plot_mean.png")
        pair_plot.savefig(pair_plot_path)
        plt.close()
        plot_paths.append(pair_plot_path)

        pair_columns = ['max', 'floor_max', 'ceiling_max']
        pair_plot = sns.pairplot(data[pair_columns], diag_kind="kde")
        pair_plot_path = os.path.join(self.plots_dir, "pair_plot_max.png")
        pair_plot.savefig(pair_plot_path)
        plt.close()
        plot_paths.append(pair_plot_path)

        pair_columns = ['min', 'floor_min', 'ceiling_min']
        pair_plot = sns.pairplot(data[pair_columns], diag_kind="kde")
        pair_plot_path = os.path.join(self.plots_dir, "pair_plot_min.png")
        pair_plot.savefig(pair_plot_path)
        plt.close()
        plot_paths.append(pair_plot_path)

        return plot_paths


plot_drawer = PlotDrawer('deviation.json')
print(plot_drawer.draw_plots())
