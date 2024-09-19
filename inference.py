from pathlib import Path
import json
from tqdm.auto import tqdm
import oimdp
import torch
from transformers import AutoModelForTextEncoding, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_keywords, contains_arabic, batched
from torch.nn.functional import cosine_similarity


def compute_scores(data_path):
    d = Path(data_path)
    all_d = d.glob("./**/*-ara1")
    all_texts = {i.name: oimdp.parse(i.read_text()) for i in all_d}

    all_scores = {kwd["keyword"]:{} for kwd in kwds}
    all_clean_texts = {}
    for name, i in tqdm(all_texts.items(), total=len(all_texts)):
        all_clean_texts[name] = []
        for j in i.content:
            if not isinstance(j, oimdp.structures.Paragraph):
                all_clean_texts[name].append([])
                for h in str(j).split():
                    if contains_arabic(h):
                        all_clean_texts[name][-1].append(h)

        with torch.inference_mode():
            for batch in batched(all_clean_texts[name], 16):
                batch = [" ".join(i) for i in batch]
                _batch = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
                hidden_states = m(**_batch).last_hidden_state
                for kwd in kwds:
                    keyword = kwd["keyword"]
                    sims = cosine_similarity(hidden_states.reshape(-1, hidden_states.shape[-1]),
                        kwd["emb"], dim=-1).cpu().numpy()

                    for i, j in zip(_batch.input_ids.reshape(-1), sims):
                        j = j.item()

                        if i in [0, 1, 2, 3]:
                            continue
                        i = tok.convert_ids_to_tokens(i.cpu().tolist())
                        if i in all_scores[keyword]:
                            all_scores[keyword][i].append(j)
                        else:
                            all_scores[keyword][i] = [j]

    for k, w in all_scores.items():
        for i, j in w.items():
            w[i] = sum(j) / len(j)

    return all_scores

if __name__ == "__main__":
    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    m = AutoModelForTextEncoding.from_pretrained(model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    m.to("cuda")

    kwds = get_keywords()
    with torch.inference_mode():
        for kwd in tqdm(kwds):
            kwd["emb"] = m(**tok(kwd["sentence"], return_tensors="pt").to("cuda"))["last_hidden_state"][0, 1]

    all_scores = compute_scores("data")

    all_data = Path("all_dta")

    with open("all_scores.json", "w") as f:
        json.dump(all_scores, f)

    df = pd.DataFrame.from_dict(all_scores).sort_values(list(all_scores.keys())[0], ascending=False).reset_index()
    df = df.loc[~df.loc[:, "index"].str.contains("##"), :]