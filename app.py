import streamlit as st
from pdf_utils import extract_text_from_pdf
from model_utils import generate_smart_summary, smart_answer

st.set_page_config(page_title="Health Policy Understanding", layout="centered")

# Header
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
Group 15 : Health Policy Understanding
</h1>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Health Policy PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    st.success("✅ Document Loaded Successfully")

    # Summary
    st.subheader("📌 Smart Summary")
    if st.button("Generate Smart Summary"):
        with st.spinner("Analyzing document intelligently..."):
            summary = generate_smart_summary(text)
            st.write(summary)

    # Q&A
    st.subheader("💬 Ask Anything About Policy")

    st.info("Try: benefits, risks, tenure, coverage, exclusions")

    question = st.text_input("Enter your question")

    if st.button("Get Intelligent Answer"):
        if question:
            with st.spinner("Thinking..."):
                answer = smart_answer(question, text)
                st.write(answer)

# Footer
st.markdown("---")
st.markdown(
    "<center>Created By - Nikhil Nanwani, Mayank Sahajramani, Sujal Nawani, Sujal Bhojwani, Roshan Chhugwani</center>",
    unsafe_allow_html=True
)