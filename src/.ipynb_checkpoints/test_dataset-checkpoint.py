from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from dataset import WikiLinguaGroupDataset, make_collate_fn

tok = AutoTokenizer.from_pretrained("google/mt5-small")

ds = WikiLinguaGroupDataset("data/groups_train.jsonl", max_examples=8)
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=make_collate_fn(tok))

batch = next(iter(dl))
print(f'en_input_ids: {batch["en_input_ids"].shape}\n zh_input_ids： {batch["zh_input_ids"].shape}\n labels: {batch["labels"].shape}')
print(f'gid: {batch["gids"][0]}')

