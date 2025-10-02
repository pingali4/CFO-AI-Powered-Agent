# Finn AI – Financial Analysis Assistant

This project is a lightweight **financial analysis assistant** designed for CFOs and finance teams.  
It can read multi-sheet Excel workbooks, apply financial formulas, and use LLMs to generate executive-friendly summaries and insights.  

---

## Features

- **Excel sheet ingestion**: Reads multiple sheets (`actuals`, `budget`, `cash`, `fx`).  
- **Formula engine**:  
  - Revenue (USD)  
  - Gross Margin % = (Revenue – COGS) / Revenue  
  - Opex totals  
  - EBITDA (proxy) = Revenue – COGS – Opex  
  - Cash runway = Cash ÷ avg. monthly net burn (last 3 months)  
- **Automatic FX handling**: Converts amounts into USD using the FX sheet (skips if already USD).  
- **LLM-powered answers**: Uses Ollama LLMs (`llama3.2` by default) to provide structured answers:
  - Executive summary  
  - Calculation steps (audit-friendly)  
  - Next-step suggestions (e.g., plotting trends)  
- **Trend plotting**: If the question mentions “trend” or “plot,” results are automatically visualized with Matplotlib.  

---

## Project Structure

    /agents 
    requirements.txt 
    README.md
---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/finn-ai.git
   cd finn-ai
2. Create a virtual environment and install dependencies:
    python -m venv venv
    venv\Scripts\activate

    pip install -r requirements.txt

3. Make sure Ollama
 is installed and running locally with the models you plan to use (e.g., llama3.2, mxbai-embed-large).


