from transformers import pipeline

# Lazy load model
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

# 🔹 Better chunking (~1000 tokens ≈ ~700–800 words)
def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# 🔹 Smart Summary (handles long docs)
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

        result = generator(
            prompt,
            max_length=300,
            truncation=True
        )[0]['generated_text']

        summaries.append(result)

    return " ".join(summaries)


# 🔹 Intelligent Q&A (handles vague queries)
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

        result = generator(
            prompt,
            max_length=200,
            truncation=True
        )[0]['generated_text']

        answers.append(result)

    return " ".join(answers)