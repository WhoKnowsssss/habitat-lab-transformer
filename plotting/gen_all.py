import pandas as pd
import os.path as osp
import os
from rl_utils.plotting.auto_table import plot_table
from rl_utils.plotting.auto_bar import plot_bar
from rl_utils.plotting.utils import MISSING_VALUE
from rl_utils.plotting.utils import fig_save
import seaborn as sns

rename_map = {
    "seen": "Train",
    "unseen": "Eval",
    "hardunseen": "Eval Hard",
    "SR": "Success Rate (%)",
}
data_folder = "plotting/data/"
table_dir = "/Users/andrewszot/Documents/writing/distill-icml2023/sections/tables_figures/"
save_dir = "data/vis"

plot_data = [
    ("reasy", ["Unseen", "Seen"], ["seen", "unseen"]),
    ("rhard", ["Unseen", "Unseen Hard"], ["unseen", "hardunseen"]),
]

for name, df_names, row_names in plot_data:
    df = pd.read_csv(osp.join(data_folder, name + ".csv"), sep="\t")
    df.fillna(MISSING_VALUE, inplace=True)
    df = df.rename({"Unnamed: 0": "Method"}, axis=1)

    names = df["Method"].tolist()

    all_df = []
    for df_name, row_name in zip(df_names, row_names):
        sub_df = df[["Method", df_name]].rename({df_name: "SR"}, axis=1)
        sub_df["split"] = row_name
        all_df.append(sub_df)
    df = pd.concat(all_df)
    df["rank"] = 0

    plot_table(
        df,
        "Method",
        "split",
        "SR",
        names,
        row_names,
        rename_map,
        include_err=False,
        write_to=osp.join(table_dir, name + ".tex"),
        value_scaling=100.0,
    )


plot_data = [
    ("context_len", "Context Length"),
    ("dataset_size", "Dataset Size (# Trajectories)"),
    ("arch", None),
]

for (name, xlabel) in plot_data:
    df = pd.read_csv(osp.join(data_folder, name + ".csv"), sep="\t")
    df.fillna(MISSING_VALUE, inplace=True)
    df = df.rename({"Unnamed: 0": "Method"}, axis=1)

    names = df["Method"].tolist()

    unseen_df = df[["Method", "Unseen"]].rename({"Unseen": "SR"}, axis=1)
    unseen_df["split"] = "unseen"
    seen_df = df[["Method", "Seen"]].rename({"Seen": "SR"}, axis=1)
    seen_df["split"] = "seen"
    df = pd.concat([unseen_df, seen_df])
    df["rank"] = 0
    df["SR"] *= 100.0

    os.makedirs(save_dir, exist_ok=True)
    colors = sns.color_palette()
    fig = plot_bar(
        df,
        "Method",
        "SR",
        names,
        bar_group_key="split",
        group_colors={"unseen": colors[0], "seen": colors[1]},
        legend=True,
        y_disp_bounds=(0, 100),
        rename_map=rename_map,
        axis_font_size=16,
        legend_font_size=18,
        xlabel=xlabel,
    )
    fig_save(save_dir, name, fig)
