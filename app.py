# app.py
import streamlit as st
from agent.agents import FinancialAgent
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI-powered CFO Assistant")

@st.cache_resource
def init_agent():
    return FinancialAgent("fixtures/data.xlsx")

agent = init_agent()

st.title("AI-powered CFO Assistant")

if st.button("New Chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about finances..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    response = agent.run(prompt)

    llm_text = response.get("llm_response")
    plot_data = response.get("plot_data")

    # 1️⃣ Display LLM text if available
    if llm_text:
        formatted_response = ""
        for line in str(llm_text).split("\n"):
            if any(char.isdigit() for char in line):
                formatted_response += f"> **{line}**  \n"
            else:
                formatted_response += f"{line}  \n"

        with st.chat_message("assistant"):
            st.markdown(formatted_response)
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

    # 2️⃣ Display plot if available
    if plot_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(plot_data["x"], plot_data["y"], color="#4CAF50", alpha=0.7)
        ax.set_title(plot_data["title"])
        ax.set_xlabel(plot_data["xlabel"])
        ax.set_ylabel(plot_data["ylabel"])
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.session_state.messages.append({"role": "assistant", "content": "📊 Graph generated."})
