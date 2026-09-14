"""Plotting script for ConflictNet modality router analysis."""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style='whitegrid', font_scale=1.2)

def plot_ablation_bar(results, out_dir, fmt):
    # results is a dict: mode -> { f1_binary, f1_weighted, ... }
    modes = []
    metrics = []
    values = []
    
    # Prettier names
    mode_names = {
        "text_only": "Text Only",
        "audio_only": "Audio Only",
        "concat_linear": "Concat (Linear)",
        "fixed_moe": "Fixed MoE",
        "adaptive_router": "Adaptive Router\n(Ours)"
    }
    
    for mode, metrics_dict in results.items():
        name = mode_names.get(mode, mode)
        
        modes.append(name)
        metrics.append("Binary Conflict F1")
        values.append(metrics_dict["f1_binary"] * 100) # percentage
        
        modes.append(name)
        metrics.append("Weighted F1")
        values.append(metrics_dict["f1_weighted"] * 100)
        
    df = pd.DataFrame({"Fusion Mode": modes, "Metric": metrics, "Score (%)": values})
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.barplot(data=df, x="Fusion Mode", y="Score (%)", hue="Metric", palette="mako", ax=ax)
    
    ax.set_ylim(0, 100)
    ax.set_title("Ablation Results on MELD Dev Set")
    sns.despine(top=True, right=True)
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=10)
    
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"ablation_bar.{fmt}")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_alpha_violin(results, out_dir, fmt):
    if "adaptive_router" not in results or "raw_alphas" not in results["adaptive_router"]:
        print("No raw alpha values found in adaptive_router mode. Skipping violin plot.")
        return
        
    alphas = results["adaptive_router"]["raw_alphas"]
    classes = results["adaptive_router"]["raw_classes"]
    
    df = pd.DataFrame({"alpha": alphas, "emotion": classes})
    # Filter unknown just in case
    df = df[df["emotion"] != "unknown"]
    
    # Capitalize
    df["emotion"] = df["emotion"].str.capitalize()
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.violinplot(data=df, x="emotion", y="alpha", inner="quartile", palette="Set2", ax=ax)
    
    # add a horizontal line at 0.5
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Equal Weight')
    
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(r"Router Gate $\alpha$")
    ax.set_xlabel("Emotion Class")
    ax.set_title(r"Modality Gate Distribution ($\alpha > 0.5$ implies text-dominant)")
    ax.legend(loc="upper right")
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, f"router_alpha_violin.{fmt}")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.ablation_json, "r") as f:
        results = json.load(f)
        
    plot_ablation_bar(results, args.output_dir, args.format)
    plot_alpha_violin(results, args.output_dir, args.format)

if __name__ == "__main__":
    main()
