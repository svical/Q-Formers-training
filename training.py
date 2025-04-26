import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
from torchvision import transforms
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"


class QFormerLayer(nn.Module):
    def __init__(self, hidden_dim, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, query_tokens, image_embeds):
        q = query_tokens
        q, _ = self.self_attn(q, q, q)
        q, _ = self.cross_attn(q, image_embeds, image_embeds)
        out = self.ffn(q)
        return out

class QFormer(nn.Module):
    def __init__(self, hidden_dim, num_query_tokens=16, num_layers=4, nhead=16):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, nhead) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.imgproj = nn.Linear(768, hidden_dim)

    def forward(self, image_embeds):
        B = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        image_embeds = self.imgproj(image_embeds)
        for layer in self.layers:
            query_tokens = layer(query_tokens, image_embeds)
        return self.fc(query_tokens)

# === Setup ===
clip_model = CLIPModel.from_pretrained("models/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")
clip_model.eval()

llm = AutoModelForCausalLM.from_pretrained("models/Llama-3.2-3B-Instruct", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("models/Llama-3.2-3B-Instruct")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llm.resize_token_embeddings(len(tokenizer))
llm.eval()

qformer = QFormer(hidden_dim=3072, num_query_tokens=16, num_layers=4, nhead=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
qformer.to(device)

for param in llm.parameters():
    param.requires_grad = False

# === Dataset ===
dataset = load_dataset("parquet", data_dir="datasets/Image_Caption", split="train")

# === Training Step ===
def train_step(images, instructions, target_texts, optimizer, criterion):
    clip_model.eval()
    llm.eval()
    qformer.train()

    # === CLIP ===
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_outputs = clip_model.vision_model(**inputs)
    image_embeds = image_outputs.last_hidden_state  # [B, N, D_clip]

    # === Q-Former ===
    query_outputs = qformer(image_embeds)  # [B, Q, D_llm]

    # === Instruction + Caption ===
    instruction_tokens = tokenizer(instructions, return_tensors="pt", padding=True, truncation=True).to(device)
    caption_tokens = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(device)

    instruction_embeds = llm.model.embed_tokens(instruction_tokens.input_ids)
    caption_embeds = llm.model.embed_tokens(caption_tokens.input_ids)

    full_input = torch.cat([query_outputs, instruction_embeds, caption_embeds], dim=1).half()

    B = query_outputs.size(0)
    query_mask = torch.ones(B, query_outputs.size(1), dtype=torch.long).to(device)
    attention_mask = torch.cat([query_mask, instruction_tokens.attention_mask, caption_tokens.attention_mask], dim=1)

    # === Labels ===
    ignore_mask = torch.full_like(query_mask, -100)
    instruction_ignore = torch.full_like(instruction_tokens.input_ids, -100)
    labels = torch.cat([ignore_mask, instruction_ignore, caption_tokens.input_ids], dim=1)

    outputs = llm(inputs_embeds=full_input, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qformer.parameters(), max_norm=1.0)
    optimizer.step()

    predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
    texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    print(texts)

    return loss.item()


# === Inference Pipeline ===
def process_inputs(image, instruction):
    with torch.no_grad():
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        image_outputs = clip_model.vision_model(**inputs)
        image_embeds = image_outputs.last_hidden_state

        query_outputs = qformer(image_embeds)

        tokens = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True).to(device)
        instruction_embeds = llm.model.embed_tokens(tokens.input_ids)

        full_input = torch.cat([query_outputs, instruction_embeds], dim=1)
        attention_mask = torch.ones(full_input.shape[:2], dtype=torch.long).to(device)

        output_ids = llm.generate(inputs_embeds=full_input, attention_mask=attention_mask, max_length=30)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def collate_fn(batch):
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),
        transforms.ToTensor()])
    images = [transform(Image.open(BytesIO(item["image"]['bytes'])).convert("RGB")) for item in batch]
    images = torch.stack(images, dim=0)

    captions = [item["text"] for item in batch]

    instructions = ["Describe the image."] * len(batch)
    
    return images, instructions, captions

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(qformer.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

os.makedirs("checkpoints", exist_ok=True)
losses = []

n_epochs = 3
for epoch in range(n_epochs):
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for images, instructions, captions in pbar:
        loss = train_step(images, instructions, captions, optimizer, criterion)
        pbar.set_postfix({"Batch Loss": loss})
        losses.append(loss)
        epoch_loss += loss

    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(qformer.state_dict(), f"checkpoints/qformer_epoch{epoch+1}.pt")
    if epoch > 1:
        os.remove(f'checkpoints/qformer_epoch{epoch-1}.pt')

plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Q-Former Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints/loss_curve.png")
plt.close()
