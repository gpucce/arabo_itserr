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

def faiss_encode(vectors, quantizer, index, train=True):
    if quantizer is not None:
        if train and vectors.shape[0] > 256:
            quantizer.train(vectors)
        quantizer.add(vectors)
    else:
        index.add(vectors)

def compute_scores(data_path, ids_to_ignore=[]):
    d = Path(data_path)
    all_d = list(d.glob("./**/*-ara1"))
    all_texts = {}
    count = 0
    for i in all_d:
        try:
            all_texts[str(i)] = oimdp.parse(i.read_text())
            count += 1
        except:
            continue
        if IS_TEST and count > 10:
            break

    count = 0
    token_line_count = 0
    input_ids_index = faiss.IndexFlatIP(1)

    d = 768
    # nlist = 100
    # faiss_m = 32
    hidden_states_index = faiss.IndexFlatIP(d)
    hidden_states_quantized_index = None
    # hidden_states_quantized_index = faiss.IndexIVFPQ(
    #     hidden_states_index, d, nlist, faiss_m, 8) # 8 specifies that each sub-vector is encoded as 8 bits

    all_scores = {kwd["keyword"]:{} for kwd in kwds}
    all_clean_texts = {}
    for name, text in tqdm(all_texts.items(), total=len(all_texts)):
        all_clean_texts[name] = {"samples": [], "metadata": {}}
        all_clean_texts[name]["metadata"]["first_line"] = count
        all_clean_texts[name]["metadata"]["first_token"] = input_ids_index.ntotal
        new_sample = []
        for text_chunk in text.content:
            if not isinstance(text_chunk, oimdp.structures.Paragraph):

                try:
                    str(text_chunk)
                except:
                    count += 1
                    continue

                for h in str(text_chunk).split():
                    if contains_arabic(h):
                        new_sample.append(h)
            else:
                if not len(new_sample) == 0:
                    all_clean_texts[name]["samples"].append({"count": count, "text": " ".join(new_sample),})
                    new_sample = []
            count += 1
        all_clean_texts[name]["metadata"]["last_line"] = count - 1

        count_skipped = 0
        with torch.inference_mode():
            vectors = None
            for batch_infos in batched(all_clean_texts[name]["samples"], 1024):
                batch = [sample["text"] for sample in batch_infos]
                tokenized_batch = tok(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=512, return_special_tokens_mask=True).to("cuda")
                mask = tokenized_batch.pop("special_tokens_mask").bool()
                input_ids_index.add(
                    tokenized_batch.input_ids[~mask].cpu().numpy().reshape(-1, 1))

                for sample, tokens, tokens_mask in zip(batch_infos, tokenized_batch.input_ids, mask):
                    sample["first_token"] = token_line_count
                    token_line_count += tokens[~tokens_mask].shape[0]
                    sample["last_token"] = token_line_count

                out = m(**tokenized_batch)
                pooler_output = out.pooler_output.cpu().numpy()
                pooler_output /= np.linalg.norm(pooler_output, axis=-1, keepdims=True)
                if vectors is None:
                    vectors = pooler_output
                else:
                    vectors = np.concatenate((vectors, pooler_output), axis=0)

                if vectors.shape[0] > 10_000:
                    faiss_encode(vectors, hidden_states_quantized_index, hidden_states_index)
                    vectors = None
                else:
                    print("#################", vectors.shape)
                    count_skipped += 1

                # if hidden_states_quantized_index is not None and count_skipped > 10:
                #     raise ValueError("Too many samples skipped")

            if vectors is not None:
                faiss_encode(vectors, hidden_states_quantized_index, hidden_states_index)
                vectors = None

        all_clean_texts[name]["metadata"]["last_token"] = input_ids_index.ntotal - 1

    # for k, w in all_scores.items():
    #     for i, j in w.items():
    #         w[i] = sum(j) / len(j)

    return all_scores, input_ids_index, hidden_states_index, all_clean_texts

if __name__ == "__main__":

    import sys
    IS_TEST = len(sys.argv) > 1 and sys.argv[1] == "test"
    print(f"########## IS_TEST: {IS_TEST} ##########")

    model_id = "CAMeL-Lab/bert-base-arabic-camelbert-ca"
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForTextEncoding.from_pretrained(model_id)
    m.to("cuda")

    ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

    kwds = get_keywords()
    with torch.inference_mode():
        for kwd in tqdm(kwds):
            kwd["emb"] = m(**tok(kwd["sentence"], return_tensors="pt").to("cuda"))["last_hidden_state"][0, 1]

    data_paths_string = [f"0{600 + i*25}AH" for i in range(12)]
    data_paths = sorted([i for i in Path("all_data").iterdir() if i.name in data_paths_string], key=str)

    if IS_TEST:
        data_paths = data_paths[:10]
    for data_path in data_paths:

        out_path = Path("arabic_out_data") / data_path.name
        if IS_TEST:
            out_path = Path("test_arabic_out_data") / data_path.name
        out_path.mkdir(exist_ok=True, parents=True)

        print(out_path)
        if Path(out_path).exists() and not IS_TEST and len(list(Path(out_path).iterdir())) > 0:
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
