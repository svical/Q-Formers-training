import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset, Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO
from torchvision import transforms
import matplotlib.pyplot as plt
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
bad_images = [169, 292, 459, 523, 531, 664, 983, 1063, 1077, 1090, 1266, 1310, 1328, 1363, 1744, 1758, 1775, 2155, 2232, 2285, 2488, 2539, 2611, 2644, 2819, 2869, 3084, 3123, 3128, 3291, 3320, 3332, 3353, 3590, 3603, 3660, 3671, 3750, 3818, 4279, 4289, 4631, 4648, 4667, 4726, 4747, 4816, 4906, 5087, 5210, 5377, 5442, 5450, 5583, 5726, 5728, 5838, 5900, 5943, 5952, 5955, 5962, 5999, 6004, 6118, 6174, 6272, 6465, 6569, 6596, 6609, 7215, 7254, 7263, 7284, 7312, 7315, 7386, 7444, 7631, 7752, 7814, 7853, 8019, 8183, 8560, 8803, 8915, 8945, 9121, 9229, 9329, 9418, 9420, 9441, 9555, 9580, 9608, 9734, 9841, 9869, 10085, 10299, 10408, 10411, 10445, 10466, 10484, 10519, 10900, 10913, 10931, 11312, 11389, 11441, 11645, 11696, 11856, 11915, 12123, 12187, 12752, 12852, 12872, 13002, 13119, 13214, 13259, 13307, 13319, 13365, 13506, 13512, 13650, 13738, 13871, 14336, 14390, 14410, 14535, 14540, 14613, 14739, 14761, 14927, 15033]

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

    def forward(self, image_embeds):
        B = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            query_tokens = layer(query_tokens, image_embeds)
        return self.fc(query_tokens)

# === Setup ===
clip_model = CLIPModel.from_pretrained("models/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")
clip_model.eval()

llm = T5ForConditionalGeneration.from_pretrained("models/t5-base", torch_dtype=torch.float16, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained("models/t5-base")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llm.resize_token_embeddings(len(tokenizer))
llm.eval()

qformer = QFormer(hidden_dim=768, num_query_tokens=16, num_layers=3, nhead=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
qformer.to(device)
print("Before load:", qformer.fc.weight[0][:5])
qformer.load_state_dict(torch.load("checkpoints_t5/qformer_epoch50.pt"))
print("After load:", qformer.fc.weight[0][:5])

for param in llm.parameters():
    param.requires_grad = False

# === Dataset ===
dataset = load_dataset("parquet", data_dir="datasets/Image_Captioning_GCC_Embeddings", split="train")
dataset = dataset.select([i for i in range(len(dataset)) if (i not in bad_images) & (i < 15100)])


# === Training Step ===
def train_step(images, instructions, target_texts, optimizer):
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

    instruction_embeds = llm.get_input_embeddings()(instruction_tokens.input_ids)
    caption_embeds = llm.get_input_embeddings()(caption_tokens.input_ids)

    full_input = torch.cat([query_outputs, instruction_embeds, caption_embeds], dim=1)

    B = query_outputs.size(0)
    query_mask = torch.ones(B, query_outputs.size(1), dtype=torch.long).to(device)
    attention_mask = torch.cat([query_mask, instruction_tokens.attention_mask, caption_tokens.attention_mask], dim=1)

    # === Labels ===
    ignore_mask = torch.full_like(query_mask, -100)
    instruction_ignore = torch.full_like(instruction_tokens.input_ids, -100)
    labels = torch.cat([ignore_mask, instruction_ignore, caption_tokens.input_ids], dim=1)

    outputs = llm(inputs_embeds=full_input, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    pred_ids = outputs.logits.argmax(dim=-1)  # [B, T]
    decoded_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    print("Predicted output:", decoded_text)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(qformer.parameters(), max_norm=1.0)
    optimizer.step()

    # predicted_token_ids = torch.argmax(outputs.logits, dim=-1)
    # texts = tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)
    # print(texts)

    return loss.item()

def collate_fn(batch):
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),
        transforms.ToTensor()])
    images = [transform(item['image_data']) for item in batch]
    images = torch.stack(images, dim=0)

    captions = [item["caption"] for item in batch]

    instructions = ["Describe the image."] * len(batch)
    
    return images, instructions, captions

dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
optimizer = torch.optim.AdamW(qformer.parameters(), lr=2e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

os.makedirs("checkpoints_t5", exist_ok=True)
losses = []
epoch_loss_list = []


n_epochs = 50
for epoch in range(n_epochs):
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for images, instructions, captions in pbar:
        loss = train_step(images, instructions, captions, optimizer)
        pbar.set_postfix({"Batch Loss": loss})
        losses.append(loss)
        epoch_loss += loss
    epoch_loss_list.append(epoch_loss / len(dataloader))
    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dataloader):.4f}")
    torch.save(qformer.state_dict(), f"checkpoints_t5/qformer_epoch{epoch+1}.pt")
    if epoch > 1:
        os.remove(f'checkpoints_t5/qformer_epoch{epoch-1}.pt')

plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Q-Former Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints_t5/loss_curve.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(epoch_loss_list, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.title("Q-Former Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("checkpoints_t5/loss_curve_epoch.png")
plt.close()
