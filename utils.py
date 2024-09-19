import json

def get_keywords(text=None):
    # some code here
    if text is None:
        text = "keywords.jsonl"
    with open(text, "r") as f:
        keywords = [json.loads(i) for i in f.readlines()]
    return keywords

def contains_arabic(text):
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
    return False