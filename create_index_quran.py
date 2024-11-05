from pathlib import Path
import os
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def compute_scores(data_path, ids_to_ignore=[]):
    d = Path(data_path)
    all_d = d.glob("./*-ara*.txt")
    all_texts = {}
    for i in all_d:
        all_texts[str(i)] = oimdp.parse(i.read_text())

    count = 0
    token_line_count = 0
    input_ids_index = faiss.IndexFlatIP(1)
    # hidden_states_index = faiss.IndexFlatIP(768)

    d = 768
    nlist = 100
    faiss_m = 8
    hidden_states_quantizer = faiss.IndexFlatL2(d)
    hidden_states_index = faiss.IndexIVFPQ(
        hidden_states_quantizer, d, nlist, faiss_m, 8) # 8 specifies that each sub-vector is encoded as 8 bits

    all_clean_texts = {}
    for name, text in tqdm(all_texts.items(), total=len(all_texts)):
        all_clean_texts[name] = {"samples": [], "metadata": {}}
        all_clean_texts[name]["metadata"]["first_line"] = count
        all_clean_texts[name]["metadata"]["first_token"] = input_ids_index.ntotal
        for text_chunk in text.content:
            if not isinstance(text_chunk, oimdp.structures.Paragraph):
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

        count_skipped = 0
        with torch.inference_mode():
            for batch_infos in batched(all_clean_texts[name]["samples"], 1024):
                batch = [sample["text"] for sample in batch_infos]
                tokenized_batch = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256, return_special_tokens_mask=True).to("cuda")
                mask = tokenized_batch.pop("special_tokens_mask").bool()
                input_ids_index.add(
                    tokenized_batch.input_ids[~mask].cpu().numpy().reshape(-1, 1))

                for sample, tokens, tokens_mask in zip(batch_infos, tokenized_batch.input_ids, mask):
                    sample["first_token"] = token_line_count
                    token_line_count += tokens[~tokens_mask].shape[0]
                    sample["last_token"] = token_line_count

                hidden_states = m(**tokenized_batch).last_hidden_state
                vectors = hidden_states[~mask].reshape(-1, hidden_states.shape[-1]).cpu().numpy()
                vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
                if vectors.shape[0] > 256:
                    hidden_states_index.train(vectors)
                else:
                    print("#################", vectors.shape)
                    count_skipped += 1

                if count_skipped > 10:
                    raise ValueError("Too many samples skipped")
                hidden_states_index.add(vectors)

        all_clean_texts[name]["metadata"]["last_token"] = input_ids_index.ntotal - 1

    return input_ids_index, hidden_states_index, all_clean_texts

if __name__ == "__main__":

    import sys
    IS_TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
    print(f"########## IS_TEST: {IS_TEST} ##########")

    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForTextEncoding.from_pretrained(model_id)
    m.to("cuda")

    ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

    data_paths = sorted(list(Path("all_data/fonti_arabo_wp8").iterdir()), key=str)
    data_paths = [i for i in data_paths if i.is_dir() if i.name != "hadith_collections"]
    if IS_TEST:
        data_paths = data_paths[:10]
    for data_path in data_paths:

        out_path = Path("wp8_out_data") / data_path.name
        if IS_TEST:
            out_path = Path("test_wp8_out_data") / data_path.name
        out_path.mkdir(exist_ok=True, parents=True)

        print(out_path)
        if Path(out_path).exists() and not IS_TEST and len(list(Path(out_path).iterdir())) > 0:
            continue
        out_path = str(out_path / "clean_texts.csv")
        input_ids_index, hidden_states_index, all_clean_texts = compute_scores(
            data_path.resolve(),
            ids_to_ignore=ids_to_ignore)

        with open(out_path.replace(".csv", ".json"), "w") as f:
            json.dump(all_clean_texts, f)

        # Save the indexes
        faiss.write_index(input_ids_index, out_path.replace(".csv", "_input_ids_index.faiss"))
        faiss.write_index(hidden_states_index, out_path.replace(".csv", "_hidden_states_index.faiss"))
