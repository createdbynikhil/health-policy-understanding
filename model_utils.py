from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 🔹 Load model (cached so it doesn't reload every time)
_generator = None

def get_generator():
    global _generator

    if _generator is None:
        model_name = "google/flan-t5-small"

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
                    max_length=200
                )

            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        _generator = generate

    return _generator


# 🔹 Chunk long document
def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# 🔹 Smart Summary
def generate_smart_summary(text):
    generator = get_generator()
    chunks = chunk_text(text)

    summaries = []

    for chunk in chunks:
        prompt = f"""
        Summarize this health policy document clearly.

        Include:
        - Key highlights
        - Benefits
        - Risks
        - Policy duration
        - Important conditions

        Text:
        {chunk}
        """

        result = generator(prompt)
        summaries.append(result)

    return " ".join(summaries)


# 🔹 Intelligent Q&A
def smart_answer(question, text):
    generator = get_generator()
    chunks = chunk_text(text)

    answers = []

    for chunk in chunks:
        prompt = f"""
        You are an intelligent health policy assistant.

        Answer the question based on the document.
        Even if exact words are missing, infer meaning.

        Question: {question}

        Document:
        {chunk}
        """

        result = generator(prompt)
        answers.append(result)

    return " ".join(answers)