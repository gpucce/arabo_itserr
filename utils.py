import sys
import json
import torch

import unicodedata
import soundfile as sf
from itertools import islice
from transformers import VitsModel, AutoTokenizer, AutoModelForSpeechSeq2Seq, pipeline,AutoProcessor
import numpy as np

def get_keywords(text=None):
    # some code here
    if text is None:
        text = "keywords.jsonl"
    with open(text, "r") as f:
        keywords = [json.loads(i) for i in f.readlines()]
    return keywords

def contains_arabic(text):
    try:
        # Arabic Unicode ranges
        arabic_ranges = [
            (0x0600, 0x06FF),  # Arabic
            (0x0750, 0x077F),  # Arabic Supplement
            (0x08A0, 0x08FF)   # Arabic Extended-A
        ]

        # Check if any character in the text is within the Arabic ranges
        for char in text:
            if any(start <= ord(char) <= end for start, end in arabic_ranges):
                return True
    except:
        return False
    return False

def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


# For generating speech
model = VitsModel.from_pretrained("facebook/mms-tts-ara")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ara")
def generate_audio(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    return (16000,np.ravel(output.cpu().numpy()))

#  For speech input
model_id = "openai/whisper-large-v3-turbo"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    # torch_dtype=torch_dtype,
    device=device,
)
punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])

def remove_punctuation(word):
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()
        
def transcribe(audio):
    result = pipe(audio, generate_kwargs={"language": "arabic"})
    return remove_punctuation(result["text"])
         
