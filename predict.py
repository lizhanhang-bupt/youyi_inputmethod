import torch
import os
from transformers import BertTokenizer, BertForMaskedLM, AdamW, DataCollatorForLanguageModeling
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
PRETRAINED_MODEL_DIR = "./model"  # 本地预训练模型目录
FINETUNED_SAVE_DIR = "./finetuned_model"  # 微调模型保存路径
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.texts = self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if len(line) > 5]  # 过滤空行

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            truncation=True,
            return_special_tokens_mask=True
        )
        return encoding

def train(model, train_loader, epochs=3, learning_rate=5e-5):
    # 创建保存目录
    os.makedirs(FINETUNED_SAVE_DIR, exist_ok=True)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = GradScaler()
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loop = tqdm(train_loader, desc="Training", leave=True)
        total_loss = 0

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Average loss: {total_loss / len(train_loader):.4f}")

    # 保存微调后的模型
    model.save_pretrained(FINETUNED_SAVE_DIR)
    tokenizer.save_pretrained(FINETUNED_SAVE_DIR)

def predict_with_finetuned(text, top_k=3):
    # 加载微调后的模型
    finetuned_tokenizer = BertTokenizer.from_pretrained(FINETUNED_SAVE_DIR)
    finetuned_model = BertForMaskedLM.from_pretrained(FINETUNED_SAVE_DIR).to(device)

    inputs = finetuned_tokenizer(text, return_tensors="pt").to(device)
    mask_positions = torch.where(inputs.input_ids[0] == finetuned_tokenizer.mask_token_id)[0]

    with torch.no_grad():
        logits = finetuned_model(**inputs).logits

    predictions = []
    for pos in mask_positions:
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_tokens = torch.topk(probs, top_k)
        tokens = finetuned_tokenizer.convert_ids_to_tokens(top_tokens.indices)
        predictions.append({
            "position": pos.item(),
            "tokens": tokens,
            "probs": top_tokens.values.tolist()
        })

    return predictions

if __name__ == "__main__":
    # 数据准备
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=8
    )

    train_dataset = TextDataset(file_path='output.txt', tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        collate_fn=data_collator,
        shuffle=True
    )
    train(model, train_loader, epochs=8)

    # 使用示例
    test_texts = [
       "今天的天气好[MASK]。",
        #"这个电影真是太[MASK]了。",
        #"这个苹果吃起来好[MASK]啊!"
    ]

    for text in test_texts:
        print(f"Input: {text}")
        results = predict_with_finetuned(text)
        for res in results:
            for token, prob in zip(res['tokens'], res['probs']):
                print(f"  {token} ")