from argparse import ArgumentParser
from pathlib import Path
import json

import faiss
import transformers
import torch
import numpy as np
import gradio as gr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--is-test", action="store_true", default=False)
    parser.add_argument("--year", type=str, default="0050")
    parser.add_argument("--word", type=str, default=None)
    parser.add_argument("--passage", type=str, default=None)
    parser.add_argument("--start-gradio", action="store_true", default=False)
    return parser.parse_args()


def search_the_index(word="كذاك", passage="وليس بدائم أبدا نعيم كذاك البؤس ليس له بقاء"):

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

    tokenized_passage = tok(args.passage, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = m(**tokenized_passage)["last_hidden_state"][0, indices[0]:indices[-1]+1].cpu().detach().numpy()
    out = out / np.linalg.norm(out, axis=-1, keepdims=True)
    out = np.mean(out, axis=0, keepdims=True)

    D, I = hs_index.search(out, 50)

    outs = {
        "recovered": [], "documents": [], "similarities": []
        # "collected": [],
    }
    for _i, _d in zip(I[0], D[0]):
        _i = int(_i)
        match_string = id_index.reconstruct_n(_i - 5, 10)
        # outs["collected"].append(
        #     tok.decode([int(i) for i in match_string.reshape(-1)]))

        for doc_name, doc in clean_texts.items():
            for sample in doc["samples"]:
                if sample["first_token"] <= _i <= sample["last_token"]:
                    if sample["text"] not in outs["recovered"]:
                        outs["similarities"].append(_d)
                        outs["recovered"].append(sample["text"])
                        outs["documents"].append(doc_name)

    return outs

if __name__ == "__main__":


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
    if args.is_test:
        args.passage = test_credo
        args.word = test_creatore

    assert args.word in args.passage
    hidden_states_index_path = Path("out_data") / f"{args.year}AH" / "all_scores_hidden_states_index.faiss"
    input_ids_index_path = Path("out_data") / f"{args.year}AH" / "all_scores_input_ids_index.faiss"
    if args.is_test:
        hidden_states_index_path = Path("test_out_data") / f"{args.year}AH" / "all_scores_hidden_states_index.faiss"
        input_ids_index_path = Path("test_out_data") / f"{args.year}AH" / "all_scores_input_ids_index.faiss"
        clean_texts_path = Path("test_out_data") / f"{args.year}AH" / "clean_texts.json"

    id_index = faiss.read_index(str(input_ids_index_path))
    hs_index = faiss.read_index(str(hidden_states_index_path))
    with open(clean_texts_path, "r") as f:
        clean_texts = json.load(f)

    tok = transformers.AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

    m = transformers.AutoModelForTextEncoding.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    m.to("cuda")

    outs = search_the_index(args.word, args.passage)

    if args.start_gradio:

        def search_the_index_gradio(word, passage, n_samples):
            if word is None:
                word = test_creatore
            if passage is None:
                passage = test_credo
            if n_samples is None:
                n_samples = 5
            out = search_the_index(word, passage)
            recovered = out["recovered"][:n_samples]
            doc_names = out["documents"][:n_samples]
            similarities = out["similarities"][:n_samples]
            out_lines = [
                f"Similarity {sim:.4f} ||| doc name: {doc_name} ||| Text: {recovered}"
                for sim, doc_name, recovered in zip(similarities, doc_names, recovered)
            ]
            return "\n\n".join(out_lines)

        demo = gr.Blocks()

        with demo:
            # "كذاك"
            # "وليس بدائم أبدا نعيم كذاك البؤس لي"

            word = gr.Textbox(label="Word to search.", placeholder=test_creatore)
            passage = gr.Textbox(label="Passage to embed the word.", placeholder=test_credo)
            n_samples = gr.Number(label="Number of samples to show.", interactive=True)
            out = gr.Textbox()

            b1 = gr.Button("Search")
            b1.click(search_the_index_gradio, inputs=[word, passage, n_samples], outputs=out)

        demo.launch()
