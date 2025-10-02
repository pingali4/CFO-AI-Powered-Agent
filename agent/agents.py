# agents.py
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import calendar
import matplotlib.pyplot as plt


class FormulaTool:    
    def __init__(self, sheets: dict):
        self.sheets = sheets
        for sheet_name, df in self.sheets.items():
            if 'month' in df.columns:
                df['month'] = pd.to_datetime(df['month'], errors='coerce')  # let pandas infer
            self.sheets[sheet_name] = df
        self.debug = True

    def run(self, query: str, return_rows: bool = False):
        def parse_month_year(q: str):
            # month/year parsing logic as before
            m, y = None, None
            match = re.search(
                r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
                r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})",
                q, re.IGNORECASE)
            if match:
                name, year_s = match.groups()
                month_map = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
                            "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
                            "aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,
                            "nov":11,"november":11,"dec":12,"december":12}
                return month_map[name.lower()], int(year_s)
            match2 = re.search(r"(\d{4})[-/](0?[1-9]|1[0-2])", q)
            if match2:
                y, m = match2.groups()
                return int(m), int(y)
            match3 = re.search(r"\b(20\d{2})\b", q)
            if match3:
                return None, int(match3.group(1))
            return None, None

        def _filter_df(df, sheet_name, account_keyword=None, contains=False):
            if df is None or df.empty:
                return pd.DataFrame(columns=(list(df.columns) + ['sheet']) if df is not None else ['sheet'])
            out = df.copy()
            if 'month' in out.columns:
                if year is not None:
                    out = out[out['month'].dt.year == year]
                if month is not None:
                    out = out[out['month'].dt.month == month]
            if account_keyword:
                if 'account_category' in out.columns:
                    mask = (out['account_category'].astype(str).str.contains(account_keyword, case=False)
                            if contains else out['account_category'].astype(str).str.lower() == account_keyword.lower())
                    out = out[mask]
                else:
                    out = out.iloc[0:0]
            out['sheet'] = sheet_name

            # 🔑 Normalize "amount" column for consistency
            if 'amount' not in out.columns:
                if 'cash_usd' in out.columns:
                    out = out.rename(columns={'cash_usd': 'amount'})
                elif 'value' in out.columns:
                    out = out.rename(columns={'value': 'amount'})
                else:
                    # fallback: create an amount column with 0
                    out['amount'] = 0.0

            return out


        def _apply_fx(df, fx_df):
            """Merge with FX rates and compute USD equivalent."""

            if df is None or df.empty:
                return df  # nothing to do

            # 🔑 If already in USD or no currency column, just copy
            if "currency" not in df.columns or df["currency"].str.upper().eq("USD").all():
                df["amount_usd"] = df["amount"]
                return df

            # 🔑 If FX table missing, assume 1.0 rate (no conversion)
            if fx_df is None or fx_df.empty:
                df["amount_usd"] = df["amount"]
                return df

            # 🔑 Standard conversion if FX sheet has required columns
            if all(col in fx_df.columns for col in ["currency", "month", "rate_to_usd"]):
                df = df.merge(
                    fx_df[["currency", "month", "rate_to_usd"]],
                    on=["currency", "month"],
                    how="left"
                )
                df["rate_to_usd"] = df["rate_to_usd"].fillna(1.0)  # fallback if missing rate
                df["amount_usd"] = df["amount"] * df["rate_to_usd"]
            else:
                df["amount_usd"] = df["amount"]

            return df


        def _summarize(df):
            if df.empty:
                return {}
            return df.groupby('currency', dropna=False)['amount_usd'].sum().to_dict()

        q = query.lower()
        month, year = parse_month_year(q)

        actuals = self.sheets.get('actuals', pd.DataFrame())
        budget = self.sheets.get('budget', pd.DataFrame())
        cash = self.sheets.get('cash', pd.DataFrame())
        fx = self.sheets.get('fx', pd.DataFrame())

        filtered_rows = pd.DataFrame()

        if 'revenue' in q:
            df_actuals = _filter_df(actuals, 'actuals', 'revenue')
            df_budget = _filter_df(budget, 'budget', 'revenue')
            filtered_rows = pd.concat([df_actuals, df_budget], ignore_index=True)
        elif 'opex' in q:
            filtered_rows = _filter_df(actuals, 'actuals', 'opex', contains=True)
        elif 'cogs' in q:
            filtered_rows = _filter_df(actuals, 'actuals', 'cogs')
        elif 'cash' in q:
            filtered_rows = _filter_df(cash, 'cash')

        if "gross margin" not in q:
            filtered_rows = _apply_fx(filtered_rows, fx)

        if not filtered_rows.empty:
            df_grouped = (
                filtered_rows.groupby(['sheet','currency'], as_index=False)
                .agg(amount_orig=('amount', 'sum'), amount_usd=('amount_usd', 'sum'))
            )
            df_pivot = df_grouped.pivot(index='currency', columns='sheet', values='amount_usd').fillna(0).reset_index()

            lines = []
            for _, row in df_pivot.iterrows():
                actual = row.get('actuals', 0.0)
                budget_val = row.get('budget', 0.0)
                lines.append(f"- {row['currency'] or 'UNKNOWN'}: Actual {actual:,.2f} USD, Budget {budget_val:,.2f} USD")

            result = "\n".join(lines)
        else:
            result = "No rows found for the requested query."

        return (result, filtered_rows) if return_rows else result


class FinancialAgent:
    def __init__(self, excel_file: str, embedding_model="mxbai-embed-large", vector_db="./chroma_financial_db", llm_model="llama3.2"):
        # Load sheets
        self.debug = True
        self.sheets = pd.read_excel(excel_file, sheet_name=None)

        # Initialize embeddings and vector store
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.db_location = vector_db
        add_docs = not os.path.exists(vector_db)
        self.documents = []
        self.ids = []

        if add_docs:
            self._prepare_documents()
            self.vector_store = Chroma(
                collection_name="financial_data",
                persist_directory=vector_db,
                embedding_function=self.embeddings
            )
            self.vector_store.add_documents(documents=self.documents, ids=self.ids)
        else:
            self.vector_store = Chroma(
                collection_name="financial_data",
                persist_directory=vector_db,
                embedding_function=self.embeddings
            )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        self.formula_tool = FormulaTool(self.sheets)
        self.llm = OllamaLLM(model=llm_model)
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert Financial Analysis assistant for a company's CFO. The total amount will be in USD as it is already converted before. Do not do any currency conversions here! All data recieved here is USD since I have already converted it so even if it ssays anothjer currency use it directly as USD.
        Use only data in FORMULA_RESULT.                                                 
        GUIDELINES (must follow):
        1. Use the section FORMULA_RESULT as the authoritative, auditable numeric source. Do NOT use numbers from retrieved documents unless they appear in FORMULA_RESULT or unless not applying formula and just retrieving data.
        2. Use DATA_CONTEXT only as background (entities, sheet names, categories) — it contains no amounts.
        3. When answering, ALWAYS show:
        - A one-line executive summary.
        - A "Calculation Steps" section listing:
        4. If data is missing, state exactly which sheet or rows are missing and what assumptions you'd make to proceed.

        DATA CONTEXT (background only):
        {data_text}

        FORMULA_RESULT (auditable numbers & rows — use this for all calculations of the formulas below):
        {formula_result} 
        Revenue (USD):  Sum of all actual revenues and sum of all budget revenues.
        Gross Margin %: (Revenue - COGS) / Revenue.
        Opex total (USD): grouped by Opex:* categories.
        EBITDA (proxy): Revenue - COGS - Opex.
        Cash runway: cash ÷ avg monthly net burn (last 3 months).

        QUESTION:
        {question}

        Provide:
        - A concise executive answer,
        - A visible "Calculation Steps" section (audit-friendly),
        - Suggest Next steps to perhaps draw a graph etc.
        """)

    def run(self, question: str):
        # Step 1: Run formula tool
        formula_result, filtered_rows = self.formula_tool.run(question, return_rows=True)

        # Step 2: Build data_text
        data_text = ""
        if filtered_rows is not None and not filtered_rows.empty:
            for _, row in filtered_rows.iterrows():
                data_text += (
                    f"- Sheet: {row.get('sheet', 'unknown')}, "
                    f"Month: {row['month'].strftime('%Y-%m') if 'month' in row else 'NA'}, "
                    f"Entity: {row['entity']}, "
                    f"Category: {row['account_category']}, "
                    f"Amount: {row['amount']} {row['currency']}\n"
                )
        else:
            data_text = "No matching rows found."

        # Step 3: Prompt
        prompt_text = self.prompt_template.format(
            data_text=data_text,
            formula_result=formula_result,
            question=question
        )

        if self.debug:
            print("\n[DEBUG] Data text going into LLM:")
            print(data_text)
            print("\n[DEBUG] Formula result going into LLM:")
            print(formula_result)
            print("\n[DEBUG] Prompt text going into LLM:")
            print(prompt_text)

        # Step 4: LLM call
        response = self.llm.invoke(prompt_text)

        # Step 5: Optional trend plotting
        if "trend" in question.lower() or "plot" in question.lower():
            if not filtered_rows.empty and "month" in filtered_rows.columns:
                # Example trend chart
                trend_df = (
                    filtered_rows.groupby("month", as_index=False)
                    .agg(total_usd=("amount_usd", "sum"))
                    .sort_values("month")
                )
                plt.figure(figsize=(8,4))
                plt.plot(trend_df["month"], trend_df["total_usd"], marker="o")
                plt.title("Trend over Time (USD)")
                plt.xlabel("Month")
                plt.ylabel("Amount (USD)")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        return response
