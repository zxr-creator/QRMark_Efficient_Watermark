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
    df = pd.DataFrame(records)
    return df.sort_values("tile_size")


# ============== 绘图函数 ==============
def plot_df(df, prefix, xlabel, ylabel):
    if df.empty:
        print(f"No valid data found for {prefix}.")
        return

    tile_sizes = df["tile_size"].astype(str)
    n = len(tile_sizes)
    bar_width = 0.3         # 缩小柱宽
    group_gap = 0.3         # 分组间距可以略大一些

    group_positions = np.arange(n) * (2 * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=(8, 4.5))  # 可以略宽些防止压缩

    bars1 = ax.bar(group_positions, df["val_bit_acc_avg"], width=bar_width,
                   label="Bit Accuracy", edgecolor="black")
    bars2 = ax.bar(group_positions + bar_width, df["val_word_acc_avg"], width=bar_width,
                   label="Word Accuracy", edgecolor="black")

    max_height = max(df["val_bit_acc_avg"].max(), df["val_word_acc_avg"].max())
    ax.set_ylim(0.0, min(1.0, max_height + 0.06))

    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * 0.015
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + y_offset,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7)  # 更小字体

    ax.set_xlabel(xlabel, color='black')
    ax.set_ylabel(ylabel, color='black')
    ax.set_xticks(group_positions + bar_width / 2)
    ax.set_xticklabels(tile_sizes, color='black')
    ax.tick_params(axis='y', colors='black')

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              borderaxespad=0, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_filename = f"{prefix}_val_accuracy_vs_tile_size.pdf"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Saved: {output_filename}")
    plt.close()



# ============== 主流程 ==============
main_folder = "hidden/output"
prefix_list = [
    "exp_tile_size",         
    "exp_num_bits",
]
xlabels=[
    "Tile Size (pixels)",
    "Number of Total Bits",
]

ylabels=[
    "Validation Accuracy",
    "Validation Accuracy",
]
for i, prefix in enumerate(prefix_list):
    df = extract_val_acc_with_tile_size(main_folder, prefix)
    plot_df(df, prefix, xlabels[i], ylabels[i])
