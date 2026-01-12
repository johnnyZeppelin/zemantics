# main.py
from datasets import load_dataset

ds_en = load_dataset("esdurmus/wiki_lingua", "english")
ds_zh = load_dataset("esdurmus/wiki_lingua", "chinese")

ex_en = ds_en["train"][0]
ex_zh = ds_zh["train"][0]
print(ex_en.keys())
print(ex_en["article"].keys())
print(ex_zh.keys())
print(ex_zh["article"].keys())

# something else 57945 in total 6541 for Chinese in total
# for kk, itt in ds_en["train"][57944].items():
#     print(f'{kk}: {itt}')
# print(f'It is {ds_zh["train"][6539]['article'].keys()}')
print(f'It is {ds_zh["train"][6539]['article']['section_name']}')
print(f'HERE WE GO: {ds_zh["train"][6539]['article']['english_section_name']}')
print(f'HERE WE SUMMARIZE: {ds_zh["train"][6539]['article']['summary']}')
