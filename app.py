import streamlit as st
from build_db import build_database
from query import retrieve_chunks, generate_answer_hf

st.set_page_config(page_title="Polycab FAQ Bot", page_icon="💡")
st.title("Polycab Fans FAQ Bot")
st.write("Ask anything about the Polycab Fans catalogue!")

# User input
user_query = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            matched_chunks = retrieve_chunks(user_query)
            answer = generate_answer_hf(user_query, matched_chunks)
        st.subheader("Answer")
        st.success(answer)
