import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Advanced Multimodal RAG", layout="wide")

st.title("📈 Economic Survey RAG")
st.markdown(
    "Search the Economic Survey of India using Hybrid Search + Cross-Encoder Reranking."
)

query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What is the trend in direct tax collections for FY26?",
)

if st.button("Search"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving, Reranking, and Generating..."):
            try:
                res = requests.post(f"{API_URL}/query", json={"query": query})
                if res.status_code != 200:
                    st.error(f"API Error: {res.text}")
                else:
                    data = res.json()

                    st.subheader("✨ Final Answer")
                    st.success(data["final_response"])

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🔍 Stage 1: Hybrid Search (Top 20)")
                        for i, doc in enumerate(data["stage1_docs"][:10], 1):
                            with st.expander(
                                f"{i}. [{doc['type'].upper()}] {doc['source']}"
                            ):
                                st.write(doc["content"] + "...")

                    with col2:
                        st.subheader("🎯 Stage 2: Cross-Encoder (Top 5)")
                        for i, doc in enumerate(data["stage2_docs"], 1):
                            with st.expander(
                                f"{i}. (Score: {doc['score']:.2f}) [{doc['type'].upper()}] {doc['source']}"
                            ):
                                st.write(doc["content"] + "...")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect: {e}")
