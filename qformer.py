import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, T5Tokenizer, T5ForConditionalGeneration

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

    def forward(self, query_tokens, instruction_embeds, image_embeds):
        # Self-attention over queries and instruction tokens
        x = torch.cat([query_tokens, instruction_embeds], dim=1)  # [B, Q+T, D]
        x, _ = self.self_attn(x, x, x)

        # Cross-attention: only query tokens attend to image embeddings
        q = x[:, :query_tokens.size(1), :]  # isolate query tokens
        q, _ = self.cross_attn(q, image_embeds, image_embeds)

        # Feedforward
        out = self.ffn(q)
        return out

class QFormer(nn.Module):
    def __init__(self, hidden_dim=768, num_query_tokens=16, num_layers=4, nhead=8):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim))
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_dim, nhead) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Project to LLM dimension if needed

    def forward(self, image_embeds, instruction_embeds):
        B = image_embeds.size(0)
        query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, Q, D]
        for layer in self.layers:
            query_tokens = layer(query_tokens, instruction_embeds, image_embeds)
        return self.fc(query_tokens)  # [B, Q, D_out]



# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Load LLM and tokenizer
llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
llm.eval()

# Initialize Q-Former
qformer = QFormer(hidden_dim=768, num_query_tokens=16, num_layers=4)

# === Inference Pipeline ===

def process_inputs(image, instruction):
    with torch.no_grad():
        # 1. Extract image embeddings from CLIP
        inputs = clip_processor(images=image, return_tensors="pt")
        image_outputs = clip_model.vision_model(**inputs)
        image_embeds = image_outputs.last_hidden_state  # [1, N, D]

        # 2. Tokenize instruction
        tokens = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        instruction_embeds = llm.encoder.embed_tokens(input_ids)  # [1, T, D]

    # 3. Pass through Q-Former
    query_outputs = qformer(image_embeds, instruction_embeds)  # [1, Q, D]

    # 4. Concatenate with instruction and feed into LLM
    full_input = torch.cat([query_outputs, instruction_embeds], dim=1)

    # Optional: Create new attention mask
    new_attention_mask = torch.ones(full_input.shape[:2], dtype=torch.long)

    # 5. Generate response
    output_ids = llm.generate(inputs_embeds=full_input, attention_mask=new_attention_mask, max_length=30)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# === Usage ===
# response = process_inputs(image, "Which picture shows the pizza inside the oven?")
# print("Response:", response)
