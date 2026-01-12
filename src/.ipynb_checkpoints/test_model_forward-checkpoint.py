import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from dataset import WikiLinguaGroupDataset, make_collate_fn
from model import LatentRendererModel

tok = AutoTokenizer.from_pretrained("google/mt5-small")
ds = WikiLinguaGroupDataset("data/groups_valid.jsonl", max_examples=8)
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok))

batch = next(iter(dl))

model = LatentRendererModel(backbone_name="google/mt5-small", num_latents=16)
model.eval()

with torch.no_grad():
    out = model(
        en_input_ids=batch["en_input_ids"],
        en_attention_mask=batch["en_attention_mask"],
        zh_input_ids=batch["zh_input_ids"],
        zh_attention_mask=batch["zh_attention_mask"],
        labels=batch["labels"],
    )

print(out.logits_en.shape, out.logits_zh.shape)
print(out.zbar_en.shape, out.zbar_zh.shape)
print(out.z_en.shape, out.z_zh.shape)
