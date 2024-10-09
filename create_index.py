from pathlib import Path
import json
import faiss
from tqdm.auto import tqdm
import oimdp
import torch
from transformers import AutoModelForTextEncoding, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import get_keywords, contains_arabic, batched
from torch.nn.functional import cosine_similarity


def compute_scores(data_path, ids_to_ignore=[]):
    d = Path(data_path)
    all_d = d.glob("./**/*-ara1")
    all_texts = {i.name: oimdp.parse(i.read_text()) for i in all_d}

    count = 0
    token_line_count = 0
    input_ids_index = faiss.IndexFlatIP(1)
    hidden_states_index = faiss.IndexFlatIP(768)
    all_scores = {kwd["keyword"]:{} for kwd in kwds}
    all_clean_texts = {}
    for name, text in tqdm(all_texts.items(), total=len(all_texts)):
        all_clean_texts[name] = {"samples": [], "metadata": {}}
        all_clean_texts[name]["metadata"]["first_line"] = count
        all_clean_texts[name]["metadata"]["first_token"] = input_ids_index.ntotal
        for text_chunk in text.content:
            if not isinstance(text_chunk, oimdp.structures.Paragraph):
                all_clean_texts[name]["samples"]
                new_sample = []
                for h in str(text_chunk).split():
                    if contains_arabic(h):
                        new_sample.append(h)
                all_clean_texts[name]["samples"].append({
                    "count": count,
                    "text": " ".join(new_sample),
                })
            count += 1
        all_clean_texts[name]["metadata"]["last_line"] = count - 1

        with torch.inference_mode():
            for batch_infos in batched(all_clean_texts[name]["samples"], 512):
                batch = [sample["text"] for sample in batch_infos]
                tokenized_batch = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256, return_special_tokens_mask=True).to("cuda")
                mask = tokenized_batch.pop("special_tokens_mask").bool()
                input_ids_index.add(
                    tokenized_batch.input_ids[~mask].cpu().numpy().reshape(-1, 1))

                line_infos = []
                for sample, tokens, tokens_mask in zip(batch_infos, tokenized_batch.input_ids, mask):
                    sample["first_token"] = token_line_count
                    token_line_count += tokens[~tokens_mask].shape[0]
                    sample["last_token"] = token_line_count

                hidden_states = m(**tokenized_batch).last_hidden_state
                vectors = hidden_states[~mask].reshape(-1, hidden_states.shape[-1]).cpu().numpy()
                vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
                hidden_states_index.add(vectors)
                for kwd in kwds:
                    keyword = kwd["keyword"]
                    sims = cosine_similarity(
                        hidden_states.reshape(-1, hidden_states.shape[-1]), kwd["emb"], dim=-1
                    ).cpu().numpy()
                    for input_id, sim_score in zip(tokenized_batch.input_ids.reshape(-1), sims):
                        sim_score = sim_score.item()
                        if input_id in ids_to_ignore:
                            continue
                        input_id = tok.convert_ids_to_tokens(input_id.cpu().tolist())
                        if input_id in all_scores[keyword]:
                            all_scores[keyword][input_id].append(sim_score)
                        else:
                            all_scores[keyword][input_id] = [sim_score]

        all_clean_texts[name]["metadata"]["last_token"] = input_ids_index.ntotal - 1

    for k, w in all_scores.items():
        for i, j in w.items():
            w[i] = sum(j) / len(j)

    return all_scores, input_ids_index, hidden_states_index, all_clean_texts

if __name__ == "__main__":

    import sys
    IS_TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForTextEncoding.from_pretrained(model_id)
    m.to("cuda")

    ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

    kwds = get_keywords()
    with torch.inference_mode():
        for kwd in tqdm(kwds):
            kwd["emb"] = m(**tok(kwd["sentence"], return_tensors="pt").to("cuda"))["last_hidden_state"][0, 1]

    data_paths = sorted(list(Path("all_data").iterdir()), key=str)
    if IS_TEST:
        data_paths = [Path("all_data/0025AH")]
    for data_path in data_paths:

        out_path = Path("out_data") / data_path.name
        if IS_TEST:
            out_path = Path("test_out_data") / data_path.name
        out_path.mkdir(exist_ok=True, parents=True)

        print(out_path)
        if Path(out_path).exists() and not IS_TEST:
            continue
        out_path = str(out_path / "all_scores.csv")
        all_scores, input_ids_index, hidden_states_index, all_clean_texts = compute_scores(
            (data_path / "data").resolve(),
            ids_to_ignore=ids_to_ignore)

        # Save the scores
        with open(out_path.replace(".csv", ".json"), "w") as f:
            json.dump(all_scores, f)

        with open(out_path.replace("all_scores", "clean_texts").replace(".csv", ".json"), "w") as f:
            json.dump(all_clean_texts, f)

        # Create and save the dataframe
        df = (pd.DataFrame
              .from_dict(all_scores)
              .sort_values(list(all_scores.keys())[0], ascending=False)
              .reset_index())

        df = df.loc[~df.loc[:, "index"].str.contains("##"), :]
        df.to_csv(out_path, index=False)

        # Save the indexes
        faiss.write_index(input_ids_index, out_path.replace(".csv", "_input_ids_index.faiss"))
        faiss.write_index(hidden_states_index, out_path.replace(".csv", "_hidden_states_index.faiss"))
        if IS_TEST:
            break
