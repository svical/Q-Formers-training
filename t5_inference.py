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
    
clip_model = CLIPModel.from_pretrained("models/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")

llm = T5ForConditionalGeneration.from_pretrained("models/t5-base", torch_dtype=torch.float16)
tokenizer = T5Tokenizer.from_pretrained("models/t5-base")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llm.resize_token_embeddings(len(tokenizer))

qformer = QFormer(hidden_dim=768, num_query_tokens=16, num_layers=3, nhead=16)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
qformer.to(device)
llm.to(device)
qformer.load_state_dict(torch.load("checkpoints_t5_noCaption_noIns/qformer_epoch50.pt", map_location=device))

clip_model.eval()
qformer.eval()
llm.eval()

def get_image_embeds(image):
    transform = transforms.Compose([
        transforms.Resize((768, 1024)),
        transforms.ToTensor()])
    image = transform(image)
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_outputs = clip_model.vision_model(**inputs)
        image_embeds = vision_outputs.last_hidden_state  # [1, num_patches+1, 768]
    return image_embeds  # shape: [1, N, 768]

# Inference function
def generate_caption(image, instruction_text="Describe the image."): # Pass the PIL image directly
    image_embeds = get_image_embeds(image) # Expects a PIL image

    # Run through Q-Former
    with torch.no_grad():
        qformer_outputs = qformer(image_embeds).to(dtype=torch.float16) # shape: [1, num_query_tokens, 768]
        print(qformer_outputs.max(), qformer_outputs.mean(), qformer_outputs.min())
    instruction_tokens = tokenizer(instruction_text, return_tensors="pt", padding=False, truncation=True).to(device)
    instruction_embeds = llm.get_input_embeddings()(instruction_tokens.input_ids).to(dtype=torch.float16) # [1, S, D_llm]


    full_input_embeds = torch.cat([qformer_outputs, instruction_embeds], dim=1) # [1, Q+S, D_llm]

    query_mask = torch.ones(1, qformer_outputs.size(1), dtype=torch.long).to(device) # Mask for Q-Former outputs (all attend)
    instruction_mask = instruction_tokens.attention_mask # Mask from tokenizer for instruction
    full_attention_mask = torch.cat([query_mask, instruction_mask], dim=1) # [1, Q+S]


    # Generate
    with torch.no_grad():
        outputs = llm.generate(
            inputs_embeds=full_input_embeds,
            attention_mask=full_attention_mask, # **** ADDED ATTENTION MASK ****
            max_new_tokens=50,                  # Increased max length slightly
            repetition_penalty=1.1,             # **** ADDED REPETITION PENALTY ****
            num_beams=4,                        # Added beam search (optional, often helps quality)
            do_sample=False                     # Ensure sampling is off if using beams/greedy
            # temperature=1.0,                  # Removed or set to 1.0 (not needed for do_sample=False)
        )
    # print("Generated Token IDs:", outputs[0])
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Testing ---
dataset = load_dataset("parquet", data_dir="datasets/Image_Captioning_GCC_Embeddings", split="train")
# Ensure you get a PIL Image object from the dataset
image_data = dataset[1]['image_data'] # Assuming this field contains the PIL Image or bytes to decode

# If 'image_data' contains bytes, decode it first:
# from io import BytesIO
# image = Image.open(BytesIO(image_data)).convert("RGB")
# Otherwise, if it's already a PIL image:
image = image_data.convert("RGB") # Ensure it's RGB

caption = generate_caption(image) # Pass the PIL image
print("Generated Caption:", caption)