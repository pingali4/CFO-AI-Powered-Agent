# app.py
import streamlit as st
import matplotlib.pyplot as plt
import io
from fpdf import FPDF
from agent.agents import FinancialAgent
import tempfile
import os

st.set_page_config(page_title="AI-powered CFO Assistant", layout="wide")
@st.cache_resource
def init_agent():
    return FinancialAgent("fixtures/data.xlsx")

agent = init_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "plots" not in st.session_state:
    st.session_state.plots = []

st.title("AI-powered CFO Assistant")

if st.button("New Chat"):
    st.session_state.messages.clear()
    st.session_state.plots.clear()
    st.experimental_rerun()

for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    if idx < len(st.session_state.plots) and st.session_state.plots[idx] is not None:
        st.pyplot(st.session_state.plots[idx])

if prompt := st.chat_input("Ask about finances"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = agent.run(prompt)
    llm_text = response.get("llm_response")
    plot_data = response.get("plot_data")

    fig_to_add = None

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
        fig_to_add = fig

    st.session_state.plots.append(fig_to_add)

def export_chat_pdf(messages, plots):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "AI-powered CFO Assistant Chat Export", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.ln(5)

    for i, msg in enumerate(messages):
        role = msg["role"].capitalize()
        safe_content = msg["content"].encode("latin-1", errors="ignore").decode("latin-1")
        pdf.multi_cell(0, 6, f"{role}:\n{safe_content}\n\n")

        if plots and i < len(plots) and plots[i] is not None:
            fig = plots[i]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                fig.savefig(tmpfile.name, bbox_inches="tight")
                plt.close(fig)
                pdf.image(tmpfile.name, x=10, w=180)
            os.remove(tmpfile.name)
            pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        tmp_pdf.seek(0)
        pdf_bytes = io.BytesIO(tmp_pdf.read())
    os.remove(tmp_pdf.name)
    pdf_bytes.seek(0)
    return pdf_bytes


if st.button("Download Chat as PDF"):
    plots_list = st.session_state.get("plots", [None]*len(st.session_state.messages))
    
    pdf_file = export_chat_pdf(st.session_state.messages, plots_list)
    
    st.download_button(
        label="Download PDF",
        data=pdf_file,
        file_name="chat_export.pdf",
        mime="application/pdf"
    )