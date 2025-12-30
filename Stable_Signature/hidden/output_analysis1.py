import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.2
})
plt.style.use('ggplot')

def extract_val_acc_with_tile_size(main_folder, prefix):
    records = []
    for folder_name in os.listdir(main_folder):
        if not folder_name.startswith(prefix):
            continue
        m = re.search(rf"{prefix}_(\d+)_", folder_name)
        if not m:
            continue
        tile_size = int(m.group(1))

        log_path = os.path.join(main_folder, folder_name, "log.txt")
        if not os.path.isfile(log_path):
            continue

        with open(log_path, "r") as f:
            for line in reversed(f.readlines()):
                try:
                    d = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue
                if {"val_bit_acc_avg", "val_word_acc_avg"} <= d.keys():
                    records.append({
                        "tile_size": tile_size,
                        "val_bit_acc_avg": d["val_bit_acc_avg"],
                        "val_word_acc_avg": d["val_word_acc_avg"]
                    })
                    break
    return pd.DataFrame(records)

def plot_grouped_bars_combined(df_dict):
    """Visualise Bit & Word accuracy for different tile sizes and sampling methods."""

    import numpy as np
    import matplotlib.pyplot as plt

    # --------------------- data prep --------------------- #
    tile_sizes = sorted({tile for df in df_dict.values() for tile in df["tile_size"]})
    methods = list(df_dict.keys())
    method_labels = {
        "exp_tile_random": "Random",
        "exp_tile_random_grid": "Random Grid",
        "exp_tile_grid": "Grid",
    }

    num_tile_sizes = len(tile_sizes)
    num_methods = len(methods)
    num_metrics = 2  # Bit + Word

    # --------------------- layout ------------------------ #
    bar_width = 0.4
    group_gap = 0.15
    group_width = num_methods * num_metrics * bar_width

    x = np.arange(num_tile_sizes)
    group_lefts = x * (group_width + group_gap)
    xtick_pos = group_lefts + group_width / 2 - bar_width / 2

    base_colors = {
        "exp_tile_random": "#1f77b4",       # blue
        "exp_tile_random_grid": "#2ca02c",  # green
        "exp_tile_grid": "#d62728",         # red
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # --------------------- plotting ---------------------- #
    for i, method in enumerate(methods):
        df = df_dict[method].set_index("tile_size").reindex(tile_sizes).sort_index()
        bit_vals  = df["val_bit_acc_avg"].values
        word_vals = df["val_word_acc_avg"].values
        base_color = base_colors[method]

        for j, (vals, metric) in enumerate([(bit_vals, "Bit"), (word_vals, "Word")]):
            xpos  = group_lefts + (i * num_metrics + j) * bar_width
            alpha = 1.0 if metric == "Bit" else 0.45

            label = f"{method_labels[method]} - {metric} Acc" 
            ax.bar(
                xpos, vals, width=bar_width,
                color=base_color, alpha=alpha, edgecolor="black",
                label=label
            )

            # 数值标签
            for xi, val in zip(xpos, vals):
                ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=6.5)

    # ------------------ axis & legend -------------------- #
    ax.set_xlabel("Tile Size (pixels)", color="black")
    ax.set_ylabel("Validation Accuracy", color="black")
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(tile_sizes, color="black")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="y", colors="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend now includes BOTH Bit & Word explanations
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    plt.savefig("combined_grouped_bar_accuracy_vs_tile_size.pdf", bbox_inches="tight")
    print("Saved: combined_grouped_bar_accuracy_vs_tile_size.pdf")
    plt.close()





# ============== 主流程 ==============
main_folder = "hidden/output"
prefix_list = [
    "exp_tile_random",
    "exp_tile_random_grid",
    "exp_tile_grid",
]

df_dict = {}
for prefix in prefix_list:
    df = extract_val_acc_with_tile_size(main_folder, prefix)
    if not df.empty:
        df_dict[prefix] = df

plot_grouped_bars_combined(df_dict)
