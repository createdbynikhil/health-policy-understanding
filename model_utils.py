from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# 🔹 Cache model (prevents reload every time)
_generator = None

def get_generator():
    global _generator

    if _generator is None:
        model_name = "google/flan-t5-base"   # better than small

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        def generate(prompt):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=300,
                    temperature=0.3,
                    do_sample=True
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        _generator = generate

    return _generator


# 🔹 Clean noisy PDF text
def clean_text(text):
    import re

    # remove repeated numbers
    text = re.sub(r'(\b\d+\.\d+\b\s*){2,}', ' ', text)

    # remove too many digits-only lines
    text = re.sub(r'\b\d+\b', '', text)

    # remove weird repetition
    text = re.sub(r'(.)\1{3,}', r'\1', text)

    # normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text


# 🔹 Chunk long text
def chunk_text(text, chunk_size=700):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# 🔹 Smart Structured Summary
def generate_smart_summary(text):
    generator = get_generator()

    text = clean_text(text)
    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:
        prompt = prompt = f"""
You are a professional insurance analyst.

Analyze the document and provide a clean structured summary.

Focus on:
- Policy type
- Coverage details
- Benefits
- Risks / exclusions
- Policy duration
- Important conditions

STRICT RULES:
- Ignore numbering like 7.2, 8.1
- Ignore repeated text
- Do NOT copy raw text
- Extract meaningful insights only
- Write in clean bullet points

Document:
{chunk}
"""
        result = generator(prompt)
        answers.append(result)

    return "\n".join(answers)