# app.py
import streamlit as st
from agent.agents import FinancialAgent

st.set_page_config(page_title="AI-powered CFO Assistant")

# --- Initialize agent ---
@st.cache_resource
def init_agent():
    return FinancialAgent("data.xlsx")

agent = init_agent()

# --- App UI ---
st.title("AI-powered CFO Assistant")

# Button to start a new chat
if st.button("New Chat"):
    st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input from user
if prompt := st.chat_input("Ask your question about finances..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = agent.run(prompt)  # will now include exact June 2025 revenue vs budget
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
