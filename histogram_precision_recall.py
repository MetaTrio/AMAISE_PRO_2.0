import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_precision_recall(length_file, predictions_file, output_file):
    # Load the sequences_over_1000_ids_with_len
    length_data = pd.read_csv(length_file, sep=",", header=None, names=["id", "len"])

    # Load the trueAndPredictedLabels.csv, skip the first row if it has headers
    predictions_data = pd.read_csv(predictions_file, header=0, dtype={"id": str, "y_true": int, "pred_label": float})

    # If 'pred_label' contains unexpected values
    predictions_data['pred_label'] = pd.to_numeric(predictions_data['pred_label'], errors='coerce').fillna(0).astype(int)

    # Filter predictions to include only reads present in length_data
    filtered_data = predictions_data.merge(length_data, on="id")

    # Ensure the 'len' column is numeric (coerce any non-numeric values to NaN)
    filtered_data["len"] = pd.to_numeric(filtered_data["len"], errors='coerce')

    # Drop any rows where 'len' is NaN after conversion
    filtered_data = filtered_data.dropna(subset=["len"])

    # Define length bins
    bin_edges = np.arange(1000, filtered_data["len"].max() + 500, 1000)  # 500bp bins
    filtered_data["length_bin"] = pd.cut(filtered_data["len"], bins=bin_edges, right=False)

    # Group by length_bin and calculate precision and recall
    metrics = []
    for bin_range, group in filtered_data.groupby("length_bin"):
        true_positive = ((group["y_true"] == 1) & (group["pred_label"] == 1)).sum()
        false_positive = ((group["y_true"] == 0) & (group["pred_label"] == 1)).sum()
        false_negative = ((group["y_true"] == 1) & (group["pred_label"] == 0)).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        metrics.append({"bin": str(bin_range), "precision": precision, "recall": recall})

    # Convert metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Plot Precision and Recall as histograms
    x_labels = metrics_df["bin"]
    x = np.arange(len(x_labels))

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35

    # Precision bar
    ax.bar(x - width / 2, metrics_df["precision"], width, label="Precision", color="blue")

    # Recall bar
    ax.bar(x + width / 2, metrics_df["recall"], width, label="Recall", color="orange")

    # Update the x-axis labels to be vertical
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=90, ha="center")

    fig.set_size_inches(12, 6)

    # Add remaining labels and settings
    ax.set_xlabel("Read Length Bins", fontsize=14)
    ax.set_ylabel("Scores", fontsize=14)
    ax.set_title("Precision and Recall Across Read Length Bins", fontsize=16)

    # Move the legend to the bottom left corner
    ax.legend(fontsize=12, loc="lower left")

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Save plot to file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# File paths
length_file_path = "/home/cseroot/sequences_over_1000_ids_with_len.txt"
predictions_file_path = "/home/cseroot/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/real_human_SRR11547004/Binary_FFNN_modified_final_2_2mer_3mer/trueAndPredictedLabels.csv"
output_file_path = "/home/cseroot/AMAISE_train_dataset_human_host_7000_1200_length_range_with_all_virus/real_human_SRR11547004/Binary_FFNN_modified_final_2_2mer_3mer/precision_recall_histogram.png"

# Call the function
plot_precision_recall(length_file_path, predictions_file_path, output_file_path)
