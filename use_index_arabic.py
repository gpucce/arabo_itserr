from argparse import ArgumentParser
from pathlib import Path
import json
import re
import time
import os

import faiss
import transformers
import torch
import numpy as np
import gradio as gr



def search_the_index(passage, doc="quran", n_samples=5):

    global CURRENT_DOC, HS_INDEX, CLEAN_TEXTS

    index_loading_time = 0
    if doc != CURRENT_DOC:
        CURRENT_DOC = doc
        hidden_states_index_path = Path(data_path) / doc / "all_scores_hidden_states_index.faiss"
        clean_texts_path = Path(data_path) / doc / "clean_texts.json"
        # if IS_TEST:
        #     hidden_states_index_path = Path(data_path) / doc / "clean_texts_hidden_states_index.faiss"
        #     clean_texts_path = Path(data_path) / doc / "clean_texts.json"

        start = time.time()
        HS_INDEX = faiss.read_index(str(hidden_states_index_path))
        index_loading_time = time.time() - start
        with open(clean_texts_path, "r") as f:
            CLEAN_TEXTS = json.load(f)


    # search the word in the passage
    tokenized_passage = tok(passage).input_ids

    encode_start_time = time.time()
    tokenized_passage = tok(passage, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        out = m(**tokenized_passage)["pooler_output"]
        print(out.shape)
        out = out[[0], :].cpu().detach().numpy()
    encode_time = time.time() - encode_start_time

    out = out / np.linalg.norm(out, axis=-1, keepdims=True)

    search_start_time = time.time()
    D, I = HS_INDEX.search(out, 10 * n_samples)
    search_time = time.time() - search_start_time

    outs = {"recovered": [], "documents": [], "similarities": []}


    for _i, _d in zip(I[0], D[0]):
        _i = int(_i)

        current_idx = 0
        for doc_name, _doc in CLEAN_TEXTS.items():
            doc_name = doc_name.split("/")[-1]
            year = re.search(r"\d{4}", doc_name).group(0)
            aggregate_year = [25 * i for i in range(100) if 25 * i <= int(year) < 25 * (i + 1)][0] + 25
            url = "https://github.com/OpenITI/"
            url += f"{aggregate_year:04d}AH/tree/master/data/"
            url += doc_name.split('.')[0] + "/"
            url += ".".join(doc_name.split('.')[:2]) + "/"
            url += doc_name.replace(".txt", ".mARkdown")

            # test_response = requests.get(url)
            # if test_response.status_code == 404:
            #     url = url.replace(".mARkdown", "")
            print(url)
            if not os.path.exists(
                url
                .replace("https://github.com/OpenITI/", "./all_data/")
                .replace("tree/master/", "")
            ):
                url = url.replace(".mARkdown", "")

            for idx, sample in enumerate(_doc["samples"]):
                if (current_idx + idx) == _i:
                    outs["similarities"].append(_d)
                    outs["recovered"].append(sample["text"])
                    outs["documents"].append(url)
            current_idx += idx + 1

        if outs["documents"][-1] in outs["documents"][:-1]:
            outs["recovered"] = outs["recovered"][:-1]
            outs["documents"] = outs["documents"][:-1]
            outs["similarities"] = outs["similarities"][:-1]
        # if len(outs["recovered"]) >= n_samples:
        #     break

    # print(outs)
    print(f"Index loading time: {index_loading_time:.4f}")
    print(f"Encode time: {encode_time:.4f}")
    print(f"Search time: {search_time:.4f}")

    return outs

# def parse_args():
#     parser = ArgumentParser()
#     parser.add_argument("--is-test", action="store_true", default=False)
#     # parser.add_argument("--doc", type=str, default="0050")
#     # parser.add_argument("--word", type=str, default=None)
#     # parser.add_argument("--passage", type=str, default=None)
#     # parser.add_argument("--start-gradio", action="store_true", default=False)
#     # parser.add_argument("--share", action="store_true", default=False)
#     return parser.parse_args()

if __name__ == "__main__":

    import sys
    CURRENT_DOC = None
    HS_INDEX = None
    CLEAN_TEXTS = None
    IS_TEST = len(sys.argv) > 1 and sys.argv[1] == "test"

    test_hadith = """الثاتي منسوب إلى ثات بن زيد بن رعين، تمام النسب يأتي ذكره. منهم إبراهيم بن زيد بن مرة بن شرحبيل بن حجية بن زكة بن عمرو بن شؤحبيل بن هرم بن آزاذ بن شرحبيل بن حمرة بن ذي يكلان بن ثابت بن زيد بن رعين الرعيني الثاتي المصري أبو خزيمة، ولي القضاء بمصر بعد ان عرضه الأمير أبو عون عبد الملك بن يزيد علي السيف، وقبل ذلك كان يعمل الأرسان وكان من العابدين الزاهدين، حدث عن يزيد بن أبي حبيب؛ روى عنه المفضل ابن فضالة، وخالد بن حميد، وجرير بن حازم، وغيرهم."""

    data_path = "arabic_out_data" if not IS_TEST else "test_arabic_out_data"
    docs = [i for i in Path(data_path).iterdir() if i.is_dir() and any(i.iterdir())]
    docs = sorted(["/".join(str(i).split("/")[-2 if "hadith" in str(i) else -1:]) for i in docs])

    tok = transformers.AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    ids_to_ignore = [tok.convert_tokens_to_ids(i) for i in tok.special_tokens_map.values()]

    m = transformers.AutoModelForTextEncoding.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-ca")
    m.to("cuda")


    def search_the_index_gradio(passage, doc, n_samples):
        out = search_the_index(passage, doc, n_samples)
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

    with demo:

        # gr.HTML("""
        #         <img src="file/itserr_logo.png" alt="Some text 1">
        #         <img src="/home/gpucce/Repos/arabo_panzeca/assets/itserr_logo.png" alt="Some text 2">
        #         <img src="./assets/itserr_logo.png" alt="Some text 3">
        #         <img src="/assets/itserr_logo.png" alt="Some text 4">
        #         """)

        with gr.Row():
            with gr.Column():
                passage = gr.Textbox(label="Passage to search", placeholder=test_hadith, value=test_hadith)
            doc = gr.Dropdown(label="Document", choices=docs, value=docs[0])
        n_samples = gr.Number(label="Number of samples to show", interactive=True, value=5)
        b1 = gr.Button("Search")
        out = gr.Markdown()
        b1.click(search_the_index_gradio, inputs=[passage, doc, n_samples], outputs=out)

        with gr.Row():
            gr.Image("/home/gpucce/Repos/arabo_panzeca/assets/itserr_logo.png", width=100, height=100)
            gr.Image("/home/gpucce/Repos/arabo_panzeca/assets/nextgen_eu_logo.png", width=100, height=100)

    demo.launch(server_name="0.0.0.0", server_port=48737, allowed_paths=["/"])
