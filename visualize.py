import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置字体（保留以支持中文示例文本显示，但图表使用英文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_model_data():
    model_data = {}

    # 1. Load results/*.json
    results_dir = Path("results")
    if results_dir.exists():
        for res_file in results_dir.glob("results_*.json"):
            model_name = res_file.stem.replace("results_", "")
            try:
                with open(res_file, 'r', encoding='utf-8') as f:
                    res = json.load(f)
                model_data[model_name] = {
                    "bleu": res.get("bleu_score", 0),
                    "val_loss": res.get("validation_loss", 0),
                    "examples": res.get("examples", [])
                }
            except Exception as e:
                print(f"❌ Failed to load result file {res_file}: {e}")

    # 2. Load outputs/*/*/train_log_*.json AND train_log.json (handle both cases)
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for model_type_dir in outputs_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            # Match both train_log.json and train_log_*.json
            for log_file in model_type_dir.glob("train_log*.json"):
                stem = log_file.stem  # e.g., "train_log" or "train_log_additive"
                if stem == "train_log":
                    model_name = model_type_dir.name  # e.g., "transformer"
                else:
                    variant = stem.replace("train_log_", "", 1)
                    model_name = f"{model_type_dir.name}_{variant}" if variant else model_type_dir.name

                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    
                    train_losses = log_data.get("train_losses", [])
                    if not train_losses:
                        print(f"⚠️ No 'train_losses' field found in {log_file}")
                        continue

                    if model_name not in model_data:
                        model_data[model_name] = {}
                    model_data[model_name]["train_loss"] = train_losses

                except Exception as e:
                    print(f"❌ Failed to read training log {log_file}: {e}")

    return model_data


def ensure_figures_dir():
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    return figures_dir


def plot_multi_model_bleu(model_data):
    figures_dir = ensure_figures_dir()
    model_names = list(model_data.keys())
    bleu_scores = [model_data[name].get("bleu", 0) for name in model_names]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_names, y=bleu_scores, color="skyblue")
    for i, score in enumerate(bleu_scores):
        plt.text(i, score + max(bleu_scores) * 0.01, f"{score:.2f}", ha='center', fontsize=10)

    plt.title("BLEU Scores Across Models", fontsize=14)
    plt.xlabel("Model Name", fontsize=12)
    plt.ylabel("BLEU Score (%)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(figures_dir / "multi_model_bleu.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Multi-model BLEU comparison saved: figures/multi_model_bleu.png")


def plot_train_loss_curve(model_data):
    figures_dir = ensure_figures_dir()
    plt.figure(figsize=(10, 6))
    has_data = False
    for model_name, data in model_data.items():
        if "train_loss" in data and data["train_loss"]:
            epochs = range(1, len(data["train_loss"]) + 1)
            plt.plot(epochs, data["train_loss"], label=model_name, linewidth=2)
            has_data = True

    if not has_data:
        print("⚠️ No training loss data found. Skipping plot.")
        return

    plt.title("Training Loss Curves", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "train_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training loss curve saved: figures/train_loss_curve.png")


def plot_bleu_vs_loss(model_data):
    figures_dir = ensure_figures_dir()
    for model_name, data in model_data.items():
        fig, ax1 = plt.subplots(figsize=(8, 5))
        bleu = data.get("bleu", 0)
        ax1.bar([model_name], [bleu], color='skyblue')
        ax1.set_ylabel('BLEU Score (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        val_loss = data.get("val_loss", 0)
        ax2.plot([model_name], [val_loss], 'ro', markersize=10)
        ax2.set_ylabel('Validation Loss', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title(f"Performance of {model_name}", fontsize=14)
        fig.tight_layout()
        plt.savefig(figures_dir / f"{model_name}_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    print("✅ Individual model performance plots saved in figures/")


def plot_translation_examples(model_data):
    for name, data in model_data.items():
        if data.get("examples"):
            print("\n" + "="*60)
            print(f"Translation Examples (Model: {name})")
            print("="*60)
            for i, ex in enumerate(data["examples"][:5]):
                print(f"\nExample {i+1}:")
                print(f"  Source (zh): {ex.get('src', 'N/A')}")
                print(f"  Reference (en): {ex.get('ref', 'N/A')}")
                print(f"  Prediction (en): {ex.get('pred', 'N/A')}")
            break


def main():
    print("🔍 Loading model data...")
    model_data = load_model_data()
    if not model_data:
        print("❌ No model data found. Please check 'results/' and 'outputs/' directories.")
        return

    print(f"✅ Successfully loaded {len(model_data)} models: {list(model_data.keys())}")

    plot_multi_model_bleu(model_data)
    plot_train_loss_curve(model_data)
    plot_bleu_vs_loss(model_data)
    plot_translation_examples(model_data)

    print("\n🎉 All visualizations completed!")


if __name__ == "__main__":
    main()