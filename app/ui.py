# ui.py
import streamlit as st
from rag import generate_answer

st.set_page_config(page_title="Air Quality AI", page_icon="🌍")

st.title("🌍 Air Pollution Chatbot")
with st.sidebar:
    st.title("Project Info")
    st.markdown("### Built by:")
    st.write("🎓 **Muhammad Raffay Ismat**")
    st.write("University of Central Punjab")
    st.write("Course: AI Lab")
    st.divider()
    st.markdown("""
    **Project Overview:**
    This project uses ElasticNet Regression for trend analysis 
    and RAG for interactive 
    querying of global air pollution data. It uses Kaggle's "Global Air Pollution" dataset" by Hasib Al Muzdadid.
    
    **Dataset Source:**
    [Kaggle Link](https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset)
    """)
st.markdown("Ask me about air quality levels in different cities!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is the PM2.5 in New York?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate RAG response
    with st.spinner("Analyzing dataset..."):
        try:
            answer = generate_answer(prompt)
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Error: {e}")