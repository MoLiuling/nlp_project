# # 评估 RNN 模型
# python evaluate.py --model_type rnn --model_path outputs/rnn --output_result results_rnn.json

# # 评估 Transformer 模型
# python evaluate.py --model_type transformer --model_path outputs/transformer --output_result results_transformer.json

# # 评估 T5 模型
# python evaluate.py --model_type t5 --model_path outputs/t5 --output_result results_t5.json

# evaluate.py
import os
import json
import argparse
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ====== 1. 加载自研模型（RNN/Transformer）======
def load_custom_model(model_dir, model_type):
    # 动态导入模型类
    if model_type == 'rnn':
        module_name = "models.rnn"
        class_name = "RNN"
    elif model_type == 'transformer':
        module_name = "models.transformer"
        class_name = "Transformer"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 动态导入
    module = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(module, class_name)

    import pickle

    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)
    
    with open(os.path.join(model_dir, 'vocab_zh.pkl'), 'rb') as f:
        src_vocab = pickle.load(f)
    with open(os.path.join(model_dir, 'vocab_en.pkl'), 'rb') as f:
        tgt_vocab = pickle.load(f)
    
    model = ModelClass(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pt')))
    model.eval()
    
    return model, src_vocab, tgt_vocab

def translate_custom(model, src_vocab, tgt_vocab, zh_text, device):
    # 简化版：实际需处理 tokenization, <sos>/<eos>, beam search 等
    tokens = ['<sos>'] + list(zh_text) + ['<eos>']  # 中文按字分（假设）
    src_ids = [src_vocab.get(t, src_vocab['<unk>']) for t in tokens]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    # 假设有 generate 方法
    output_ids = model.generate(src_tensor, max_len=100)
    en_words = [tgt_vocab.get_idx(token_id) for token_id in output_ids if token_id not in [0,1,2]]
    return " ".join(en_words)

# ====== 2. 加载 T5 模型 ======
def load_t5_model(model_dir):
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    return model, tokenizer

def translate_t5(model, tokenizer, zh_text, device):
    input_text = f"translate Chinese to English: {zh_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(input_ids, max_length=128, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====== 3. 主评估函数 ======
def evaluate(model_type, model_path, data_path, device):
    # 加载验证集
    valid_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            valid_data.append((item['zh'], item['en']))
    
    # 加载模型
    if model_type == 't5':
        model, tokenizer = load_t5_model(model_path)
        src_vocab = tgt_vocab = None
    else:  # 'transformer' or 'rnn'
        model, src_vocab, tgt_vocab = load_custom_model(model_path, model_type)
        tokenizer = None
    
    model.to(device)
    predictions = []
    references = []
    
    for zh, en_ref in valid_data:
        if model_type == 't5':
            pred = translate_t5(model, tokenizer, zh, device)
        else:
            pred = translate_custom(model, src_vocab, tgt_vocab, zh, device)
        
        predictions.append(pred.split())
        references.append(en_ref.split())

    # 计算 BLEU（可选，作业不要求分数但可分析）
    smooth = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], pred, smoothing_function=smooth)
        for pred, ref in zip(predictions, references)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print(f"✅ Average BLEU: {avg_bleu:.4f}")
    return avg_bleu, predictions, references, valid_data

# ====== 4. 命令行接口 ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['transformer', 'rnn', 't5'])
    parser.add_argument('--model_path', type=str, required=True)  # 如 outputs/transformer 或 outputs/t5
    parser.add_argument('--data_path', type=str, default='data/valid.jsonl')
    parser.add_argument('--output_result', type=str, default=None,
                        help='Save evaluation result to JSON file (e.g., results_rnn.json)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_bleu, predictions, references, valid_data = evaluate(args.model_type, args.model_path, args.data_path, device)

    # 保存结果
    if args.output_result:
        output_dir = os.path.dirname(args.output_result)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        examples = []
        for i in range(min(5, len(valid_data))):  # 保存前5个样例
            examples.append({
                "src": valid_data[i][0],
                "ref": " ".join(references[i]),
                "pred": " ".join(predictions[i])
            })
        
        result = {
            "model_type": args.model_type,
            "bleu_score": round(avg_bleu, 4),
            "examples": examples
        }

        with open(args.output_result, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved evaluation result to {args.output_result}")