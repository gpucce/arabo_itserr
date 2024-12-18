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

def faiss_encode(vectors, quantizer, index):
    if quantizer is not None:
        quantizer.train(vectors)
        quantizer.add(vectors)
    else:
        index.add(vectors)

def compute_scores(data_path, ids_to_ignore=[]):
    dp = Path(data_path)
    if "hadith" not in str(dp):
        all_d = dp.glob("./*-ara*.txt")
    else:
        all_d = dp.glob("./*/*-ara*.txt")

    all_texts = {}
    for idx, i in enumerate(all_d):
        all_texts[str(i)] = oimdp.parse(i.read_text())
        if IS_TEST and idx > 0:
            break

    input_ids_index = faiss.IndexFlatIP(1)
    # hidden_states_index = faiss.IndexFlatIP(768)
    d = 768
    hidden_states_index = faiss.IndexFlatL2(d)
    hidden_states_quantized_index = None

    is_small_enough = "quran" in str(data_path) or "hadith" in str(data_path) or "tafsir" in str(data_path)
    if not is_small_enough:
        nlist = 100
        faiss_m = 8
        hidden_states_quantized_index = faiss.IndexIVFPQ(
            hidden_states_index, d, nlist, faiss_m, 8) # 8 specifies that each sub-vector is encoded as 8 bits

    count = 0
    token_line_count = 0
    all_clean_texts = {}
    for name, text in tqdm(all_texts.items(), total=len(all_texts)):
        all_clean_texts[name] = {"samples": [], "metadata": {}}
        all_clean_texts[name]["metadata"]["first_line"] = count
        all_clean_texts[name]["metadata"]["first_token"] = input_ids_index.ntotal
        new_sample = []
        for text_chunk in text.content:
            if not isinstance(text_chunk, oimdp.structures.Paragraph):
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
                # hidden_states = out.pooler_output
                # vectors = hidden_states[~mask].reshape(-1, hidden_states.shape[-1]).cpu().numpy()
                # vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
                pooler_output = out.pooler_output.cpu().numpy()
                pooler_output /= np.linalg.norm(pooler_output, axis=-1, keepdims=True)
                # vectors.append(pooler_output)
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
        if hidden_states_quantized_index is None:
            assert sum(len(all_clean_texts[name]["samples"]) for name in all_clean_texts) == hidden_states_index.ntotal, f"{len(all_clean_texts[name]['samples'])} == {hidden_states_index.ntotal}"
        else:
            assert sum(len(all_clean_texts[name]["samples"]) for name in all_clean_texts) == hidden_states_quantized_index.ntotal, f"{len(all_clean_texts[name]['samples'])} == {hidden_states_index.ntotal}"
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

    # data_paths = sorted(list(Path("all_data/fonti_arabo_wp8").iterdir()), key=str)
    # data_paths = [i for i in data_paths if i.is_dir()]
    data_paths = [
        # Path("/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8/hadith_collections/sunni_collections"),
        # Path("/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8/hadith_collections/shia_collections"),
        # Path("/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8/quran"),
        # Path("/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8/sira"),
        Path("/home/gpucce/Repos/arabo_panzeca/all_data/fonti_arabo_wp8/tafsir"),
    ]
    if IS_TEST:
        data_paths = [i for i in data_paths if "hadith" in i.name]
    for data_path in data_paths:

        out_path = Path("wp8_out_data") / data_path.name
        out_path = Path(str(data_path).replace("all_data/fonti_arabo_wp8", "wp8_out_data"))
        if IS_TEST:
            out_path = Path("test_wp8_out_data") / data_path.name
        out_path.mkdir(exist_ok=True, parents=True)

        print(out_path)
        if (not IS_TEST) and Path(out_path).exists() and len(list(Path(out_path).iterdir())) > 0:
            continue
        out_path = str(out_path / "clean_texts.csv")

        (input_ids_index,
         hidden_states_index,
         all_clean_texts) = compute_scores(data_path.resolve(), ids_to_ignore=ids_to_ignore)

        with open(out_path.replace(".csv", ".json"), "w") as f:
            json.dump(all_clean_texts, f)

        # Save the indexes
        faiss.write_index(input_ids_index, out_path.replace(".csv", "_input_ids_index.faiss"))
        faiss.write_index(hidden_states_index, out_path.replace(".csv", "_hidden_states_index.faiss"))
