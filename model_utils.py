from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re

# 🔹 Cache model
_generator = None

def get_generator():
    global _generator

    if _generator is None:
        model_name = "google/flan-t5-base"

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


# 🔹 Clean text
def clean_text(text):
    text = re.sub(r'(\b\d+\.\d+\b\s*){2,}', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# 🔹 Chunk text
def chunk_text(text, chunk_size=700):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# 🔹 Smart Summary
def generate_smart_summary(text):
    generator = get_generator()

    text = clean_text(text)
    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:
        prompt = f"""
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
        summaries.append(result)

    return "\n\n".join(summaries)


# 🔹 Smart Q&A (THIS WAS MISSING ❗)
def smart_answer(question, text):
    generator = get_generator()

    text = clean_text(text)
    chunks = chunk_text(text)

    answers = []

    for chunk in chunks:
        prompt = f"""
You are an expert health insurance advisor.

Answer intelligently based on the document.

IMPORTANT:
- Even if the question is vague (like "risks", "cons", "benefits")
- Infer meaning from the document

Question: {question}

Document:
{chunk}
"""

        result = generator(prompt)
        answers.append(result)

    return "\n".join(answers)