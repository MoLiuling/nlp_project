# visualize.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（避免中文显示为方框）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def plot_bleu_and_loss():
    """假设你有多个实验的结果，这里先展示单模型情况"""
    if not os.path.exists('results.json'):
        print("❌ results.json not found. Run valid.py first!")
        return

    with open('results.json', 'r', encoding='utf-8') as f:
        res = json.load(f)

    # 创建子图
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 柱状图：BLEU
    ax1.bar(['BLEU Score'], [res['bleu_score']], color='skyblue', label='BLEU')
    ax1.set_ylabel('BLEU Score (%)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 在同一图上添加 loss（右侧Y轴）
    ax2 = ax1.twinx()
    ax2.plot(['Validation Loss'], [res['validation_loss']], 'ro', markersize=10, label='Loss')
    ax2.set_ylabel('Validation Loss', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Model Performance on Validation Set')
    fig.tight_layout()
    plt.savefig('performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved performance.png")

def plot_examples():
    """打印翻译样例（适合放入报告）"""
    if not os.path.exists('results.json'):
        return

    with open('results.json', 'r', encoding='utf-8') as f:
        res = json.load(f)

    print("\n" + "="*60)
    print("TRANSLATION EXAMPLES")
    print("="*60)
    for i, ex in enumerate(res['examples']):
        print(f"\nExample {i+1}:")
        print(f"  中文 (Src): {ex['src']}")
        print(f"  英文 (Ref): {ex['ref']}")
        print(f"  模型 (Pred): {ex['pred']}")

def main():
    plot_bleu_and_loss()
    plot_examples()
    print("\n🎉 All visualizations completed!")

if __name__ == "__main__":
    main()