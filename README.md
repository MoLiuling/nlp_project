# nlp_project
# Text Translation Model Repository
This repository provides implementations of text translation models based on **RNN**, **T5**, and **Transformer**, including complete model training, inference, and evaluation workflows. It supports two decoding strategies (greedy decoding and Beam Search) and can be directly used for Chinese-English translation tasks.

## Repository Structure
nlp_project_main/
├── data/ # Data directory (test set, vocabularies)
│ ├── test.jsonl # Test set (JSONL format)
│ ├── prepared_vocab_zh.pkl # Chinese vocabulary
│ └── prepared_vocab_en.pkl # English vocabulary
├── outputs/ # Output directory (model weights, prediction results)
│ ├── rnn/ # RNN model outputs
│ ├── t5/ # T5 model outputs
│ └── transformer/ # Transformer model outputs
├── inference.py # Core model inference script
├── evaluate.py # Model evaluation script
├── data_tool.py # Data preprocessing/vocabulary generation script
├── requirements.txt # Environment dependency list
└── README.md # Usage instructions (this document)


# 🚀 Quick Start
## 1. Install Dependencies

```bash
pip install torch transformers nltk sentencepiece jieba
```
Note:
jieba is used for Chinese word segmentation (if your preprocessing uses it).
sentencepiece is required for the T5 tokenizer.

## 2. Interactive Translation (Inference)
Run interactive translation with your trained models:
```bash
python inference.py --model <MODEL_NAME> --decode <greedy|beam> [--beam-size N]
```
### Supported Models:
RNN variants: rnn_additive, rnn_dot, rnn_multiplicative
Transformer: transformer
T5 (fine-tuned): t5

Examples:
```bash
# RNN with dot-product attention + beam search
python inference.py --model rnn_dot --decode beam --beam-size 5

# Transformer with greedy decoding
python inference.py --model transformer --decode greedy

# Fine-tuned T5 model
python inference.py --model t5 --decode beam --beam-size 4
```
💡 Type a Chinese sentence and press Enter to get the English translation. Type quit to exit.

## 3. Evaluate Model Performance (BLEU Score)
Evaluate your model on the validation set (data/valid.jsonl) and compute BLEU scores:
```bash
python evaluate.py \
  --model_type <rnn|transformer|t5> \
  --model_path <PATH_TO_CHECKPOINT> \
  [--variant <additive|dot|multiplicative>] \
  [--data_path data/valid.jsonl] \
  [--output_result results/<model_name>.json]
  ```
Example Commands:
### Evaluate RNN (additive attention)
python evaluate.py \
  --model_type rnn \
  --model_path outputs/rnn \
  --variant additive \
  --output_result results/rnn_additive.json

### Evaluate Transformer
python evaluate.py \
  --model_type transformer \
  --model_path outputs/transformer \
  --output_result results/transformer.json

### Evaluate fine-tuned T5
python evaluate.py \
  --model_type t5 \
  --model_path outputs/t5 \
  --output_result results/t5.json
✅ The output JSON includes:
Average BLEU score
Sample translations (source, reference, prediction)

## 4. Data Format
The dataset is stored in JSONL format (data/train.jsonl, data/valid.jsonl). Each line is a JSON object:
{"zh": "你好世界", "en": "Hello world"}

