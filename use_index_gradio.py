from argparse import ArgumentParser
from pathlib import Path
import json
import re
import time

import faiss
import transformers
import torch
import numpy as np
import gradio as gr


def search_the_index(word="كذاك", passage="وليس بدائم أبدا نعيم كذاك البؤس ليس له بقاء", year="0025AH", n_samples=5):

    global CURRENT_YEAR, HS_INDEX, CLEAN_TEXTS

    assert word in passage, f"The word: {word} -- is not in the passage."
    index_loading_time = 0
    if year != CURRENT_YEAR:
        CURRENT_YEAR = year
        hidden_states_index_path = Path("out_data") / year / "all_scores_hidden_states_index.faiss"
        # input_ids_index_path = Path("out_data") / year / "all_scores_input_ids_index.faiss"
        clean_texts_path = Path("out_data") / year / "clean_texts.json"
        if args.is_test:
            hidden_states_index_path = Path("test_out_data") / year / "all_scores_hidden_states_index.faiss"
            # input_ids_index_path = Path("test_out_data") / f"{args.year}AH" / "all_scores_input_ids_index.faiss"
            clean_texts_path = Path("test_out_data") / year / "clean_texts.json"


        # id_index = faiss.read_index(str(input_ids_index_path))
        start = time.time()
        HS_INDEX = faiss.read_index(str(hidden_states_index_path))
        index_loading_time = time.time() - start    
        with open(clean_texts_path, "r") as f:
            CLEAN_TEXTS = json.load(f)



    # search the word in the passage
    tokenized_passage = tok(args.passage).input_ids
    tokenized_word = tok(args.word).input_ids

    # check if the tokenized word is in the tokenized passage
    indices = []
    for token in tokenized_word:
        if token in ids_to_ignore:
            continue
        assert token in tokenized_passage
        indices.append(tokenized_passage.index(token))
        if len(indices) >= 2:
            for idx, _position in enumerate(indices[1:]):
                assert _position == indices[idx] + 1

    encode_start_time = time.time()
    tokenized_passage = tok(args.passage, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = m(**tokenized_passage)["last_hidden_state"][0, indices[0]:indices[-1]+1].cpu().detach().numpy()
    encode_time = time.time() - encode_start_time

    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    out = np.mean(out, axis=0, keepdims=True)

    search_start_time = time.time()
    D, I = HS_INDEX.search(out, 3 * n_samples)
    search_time = time.time() - search_start_time

    outs = {
        "recovered": [], "documents": [], "similarities": []
        # "collected": [],
    }
    for _i, _d in zip(I[0], D[0]):
        _i = int(_i)
        # match_string = id_index.reconstruct_n(_i - 5, 10)
        # outs["collected"].append(
        #     tok.decode([int(i) for i in match_string.reshape(-1)]))

        for doc_name, doc in CLEAN_TEXTS.items():
            split_doc_name = doc_name.split("/")
            idx = [i for i, j in enumerate(split_doc_name) if re.search(r"\d{4}", j)][0]
            joint_doc_name = "/".join([i for i in  split_doc_name[idx:]]).replace("/data/", "/tree/master/data/")
            url = f"https://github.com/OpenITI/" + joint_doc_name
            for sample in doc["samples"]:
                if sample["first_token"] <= _i <= sample["last_token"]:
                    if len(sample["text"]) > 0 and sample["text"] not in outs["recovered"]:
                        outs["similarities"].append(_d)
                        outs["recovered"].append(sample["text"])
                        outs["documents"].append(url)
    print(outs)
    print(f"Index loading time: {index_loading_time:.4f}")
    print(f"Encode time: {encode_time:.4f}")
    print(f"Search time: {search_time:.4f}")

    return outs

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--is-test", action="store_true", default=False)
    # parser.add_argument("--year", type=str, default="0050")
    parser.add_argument("--word", type=str, default=None)
    parser.add_argument("--passage", type=str, default=None)
    # parser.add_argument("--start-gradio", action="store_true", default=False)
    # parser.add_argument("--share", action="store_true", default=False)
    return parser.parse_args()



CURRENT_YEAR = None
HS_INDEX = None
CLEAN_TEXTS = None

test_credo = """وقالو هذه الاَمانة نومن بالهٍ
وَاحد الاب ضابط الكل خالق السما والأرض مَا يُرى ومَا لا يرى وَربٍ واحَد يسوع المسيح
ابن الله الوَحيد المولود من الاب قبل ُل الدُهور نورٍ من نور الهِ حق من الهَ حقَ مولود 
غير مخلوق مسَاوي الاب في الجوهَر الذي كلِ شيً كان به هذا من اجلنا معَشر البشر ومن
اَجل خلاصَنا نزل من السمآءِ وتُجسَّد من الرُوح القدس ومن مَريم العدرَي تانس وصلب عنا
على عهد بيلاطس البنطي وتالم وقُبر وقام من الامَوات في اليوم الثالث كما في الكتُب وصعَد
الى السموات وجلس عن يمين الاب وايضًا يأتي بمجده ليدين الاحَيا والاموات الذي ليس
لملكه انقضآ ونومن بالرُوح
"""

test_creatore = "خالق"
args = parse_args()
years = [i for i in Path("out_data").iterdir() if i.is_dir() and any(i.iterdir())]
years = sorted([i.name for i in years])
if args.is_test:
    years = years[:2]
args.passage = test_credo
args.word = test_creatore
assert args.word in args.passage

tok = transformers.AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

m = transformers.AutoModelForTextEncoding.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
m.to("cuda")

# outs = search_the_index(args.word, args.passage)

def search_the_index_gradio(word, passage, year, n_samples):
    if word is None:
        word = test_creatore
    if passage is None:
        passage = test_credo
    if n_samples is None:
        n_samples = 5
    out = search_the_index(word, passage, year, n_samples)
    recovered = out["recovered"][:n_samples]
    doc_names = out["documents"][:n_samples]
    similarities = out["similarities"][:n_samples]
    out_lines = [
        f"### Text: {recovered}\n - Similarity {sim:.4f}\n - URL: {doc_name}"
        for sim, doc_name, recovered in zip(similarities, doc_names, recovered)
    ]
    return "\n\n".join(out_lines)

demo = gr.Blocks(theme=gr.themes.Soft())

gr.set_static_paths("/home/gpucce/Repos/arabo_panzeca/assets")
# "كذاك"
# "وليس بدائم أبدا نعيم كذاك البؤس لي"
with demo:
    with gr.Row():
        gr.Image("/home/gpucce/Repos/arabo_panzeca/assets/itserr_logo.png", width=100, height=100)
        gr.Image("/home/gpucce/Repos/arabo_panzeca/assets/nextgen_eu_logo.png", width=100, height=100)

    # gr.HTML("""
    #         <img src="file/itserr_logo.png" alt="Some text 1">
    #         <img src="/home/gpucce/Repos/arabo_panzeca/assets/itserr_logo.png" alt="Some text 2">
    #         <img src="./assets/itserr_logo.png" alt="Some text 3">
    #         <img src="/assets/itserr_logo.png" alt="Some text 4">
    #         """)

    with gr.Row():
        with gr.Column():
            word = gr.Textbox(label="Word to search.", placeholder=test_creatore)
            passage = gr.Textbox(label="Passage to embed the word.", placeholder=test_credo)
        year = gr.Dropdown(label="Year", choices=years, value=years[0])
    n_samples = gr.Number(label="Number of samples to show.", interactive=True, value=5)
    b1 = gr.Button("Search")
    out = gr.Markdown()
    b1.click(search_the_index_gradio, inputs=[word, passage, year, n_samples], outputs=out)

demo.launch(server_name="0.0.0.0", server_port=48737, allowed_paths=["/"])
