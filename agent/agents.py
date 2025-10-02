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
from datetime import datetime


class FormulaTool:    
    def __init__(self, sheets: dict):
        self.sheets = sheets
        for sheet_name, df in self.sheets.items():
            if 'month' in df.columns:
                df['month'] = pd.to_datetime(df['month'], errors='coerce')
            self.sheets[sheet_name] = df
        self.debug = True

    def parse_months(self, query: str, default_last_n: int = 3):
        """Return list of datetime objects representing months to use."""
        months = []

        # 1️⃣ Explicit month + year (e.g., "Jan 2024")
        matches = re.findall(
            r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
            r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})",
            query, re.IGNORECASE
        )
        month_map = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,
                     "apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,
                     "aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,
                     "nov":11,"november":11,"dec":12,"december":12}
        for m_name, y in matches:
            months.append(datetime(int(y), month_map[m_name.lower()], 1))

        # 2️⃣ Numeric YYYY-MM or YYYY/MM
        matches2 = re.findall(r"(\d{4})[-/](0?[1-9]|1[0-2])", query)
        for y, m in matches2:
            months.append(datetime(int(y), int(m), 1))

        # 3️⃣ Range: "from Jan 2024 to Mar 2024"
        range_match = re.search(
            r"from\s+([a-zA-Z]+\s+\d{4})\s+to\s+([a-zA-Z]+\s+\d{4})", query, re.IGNORECASE
        )
        if range_match:
            start_str, end_str = range_match.groups()
            start_month, start_year = self.parse_month_year_single(start_str)
            end_month, end_year = self.parse_month_year_single(end_str)
            if start_month and end_month:
                dt_start = datetime(start_year, start_month, 1)
                dt_end = datetime(end_year, end_month, 1)
                current = dt_start
                while current <= dt_end:
                    months.append(current)
                    # increment month
                    next_month = current.month + 1
                    next_year = current.year
                    if next_month > 12:
                        next_month = 1
                        next_year += 1
                    current = datetime(next_year, next_month, 1)

        # 4️⃣ "last N months"
        last_n_match = re.search(r"last\s+(\d+)\s+months?", query.lower())
        n_months = int(last_n_match.group(1)) if last_n_match else default_last_n

        if not months:
            today = datetime.today()
            for i in range(n_months-1, -1, -1):
                month = today.month - i
                year = today.year
                if month <= 0:
                    month += 12
                    year -= 1
                months.append(datetime(year, month, 1))

        # remove duplicates and sort
        months = sorted(list({(m.year, m.month): m for m in months}.values()))
        return months

    def parse_month_year_single(self, q: str):
        """Parse single month-year string like 'Jan 2024' and return (month, year)."""
        match = re.search(r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{4})", q, re.IGNORECASE)
        month_map = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}
        if match:
            name, year_s = match.groups()
            return month_map[name.lower()], int(year_s)
        return None, None

    def filter_by_months(self, df, months):
        if df is None or df.empty or not months:
            return df
        df = df[df['month'].apply(lambda d: any(d.year==m.year and d.month==m.month for m in months))]
        return df

    def run(self, query: str, return_rows: bool = False):
        months_to_use = self.parse_months(query)
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
                out = self.filter_by_months(out, months_to_use)
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

        months_to_use = self.parse_months(query)  # ✅ get all months including ranges
        actuals = self.sheets.get('actuals', pd.DataFrame())
        budget = self.sheets.get('budget', pd.DataFrame())
        cash = self.sheets.get('cash', pd.DataFrame())
        fx = self.sheets.get('fx', pd.DataFrame())

        filtered_rows = pd.DataFrame()

        q = query.lower()

        if 'revenue' in q:
            df_actuals = self.filter_by_months(_filter_df(actuals, 'actuals', 'revenue'), months_to_use)
            df_budget = self.filter_by_months(_filter_df(budget, 'budget', 'revenue'), months_to_use)
            filtered_rows = pd.concat([df_actuals, df_budget], ignore_index=True)
        elif 'gross margin' in q:
            df_revenue = _apply_fx(self.filter_by_months(_filter_df(actuals, 'actuals', 'revenue'), months_to_use), fx)
            df_cogs = _apply_fx(self.filter_by_months(_filter_df(actuals, 'actuals', 'COGS'), months_to_use), fx)
            filtered_rows = pd.concat([df_revenue, df_cogs], ignore_index=True)
        elif 'opex' in q:
            filtered_rows = self.filter_by_months(_filter_df(actuals, 'actuals', 'opex', contains=True), months_to_use)
        elif 'cogs' in q:
            filtered_rows = self.filter_by_months(_filter_df(actuals, 'actuals', 'cogs'), months_to_use)
        elif 'cash' in q:
            filtered_rows = self.filter_by_months(_filter_df(cash, 'cash'), months_to_use)

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

        months_to_use = parse_month_year(query)
        if not filtered_rows.empty and "month" in filtered_rows.columns:
            trend_df = (
                filtered_rows.groupby("month", as_index=False)
                .agg(total_usd=("amount_usd", "sum"))
                .sort_values("month")
            )
            if "trend" in q or "plot" in q:
                return {
                    "x": [m.strftime("%b %Y") for m in trend_df["month"]],
                    "y": trend_df["total_usd"].tolist(),
                    "title": "Trend over Time (USD)",
                    "xlabel": "Month",
                    "ylabel": "Amount (USD)"
                }

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
        You are an expert Financial Analysis assistant for a company's CFO.  

        The total amount will be in **USD** as it is already converted before.  
        Do **not** do any currency conversions here!  
        All data you receive is **already in USD** (even if it says EUR, GBP, etc., treat it directly as USD).  

        Use **only** the data in FORMULA_RESULT to answer.  
        Provide clear, step-by-step reasoning when calculating.  
        If numbers are involved, always show intermediate steps.    
                                                                                                         
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
        Revenue (USD):  Sum of all actual revenues and sum of all budget revenues. Separate the actuals and budget sums.
        Gross Margin %: (Revenue - COGS) / Revenue
        Opex total (USD): grouped by Opex:* categories.
        EBITDA (proxy): Revenue - COGS - Opex.
        Cash runway: cash ÷ avg monthly net burn (last 3 months).

        QUESTION:
        {question}

        Provide:
        - A concise executive answer,
        - A visible "Calculation Steps" section (audit-friendly)
        """)

    def run(self, question: str):
        res = self.formula_tool.run(question, return_rows=True)

        if isinstance(res, dict):
            return res  # plotting data

        formula_result, filtered_rows = res

        # Build data_text
        data_text = ""
        if filtered_rows is not None and not filtered_rows.empty:
            for _, row in filtered_rows.iterrows():
                row_dict = row.to_dict()
                data_text += (
                    f"- Sheet: {row_dict.get('sheet','unknown')}, "
                    f"Month: {row_dict.get('month').strftime('%Y-%m') if pd.notnull(row_dict.get('month')) else 'NA'}, "
                    f"Entity: {row_dict.get('entity','NA')}, "
                    f"Category: {row_dict.get('account_category','NA')}, "
                    f"Amount: {row_dict.get('amount',0.0)} {row_dict.get('currency','USD')}\n"
                )
        else:
            data_text = "No matching rows found."


        # Prompt
        prompt_text = self.prompt_template.format(
            data_text=data_text,
            formula_result=formula_result,
            question=question
        )

        if self.debug:
            print("\n[DEBUG] Data text:")
            print(data_text)
            print("\n[DEBUG] Formula result:")
            print(formula_result)

        # LLM call
        response = self.llm.invoke(prompt_text)

                # Optional plotting
        if "trend" in question.lower() or "plot" in question.lower():
            months_to_plot = self.formula_tool.parse_months(question)
            filtered_rows = self.formula_tool.filter_by_months(filtered_rows, months_to_plot)
            if not filtered_rows.empty and "month" in filtered_rows.columns:
                trend_df = (
                    filtered_rows.groupby("month", as_index=False)['amount_usd']
                    .sum().sort_values("month")
                )
                return {
                    "llm_response": None,
                    "plot_data": {
                        "x": [m.strftime("%b %Y") for m in trend_df["month"]],
                        "y": trend_df["amount_usd"].tolist(),
                        "title": "Trend over Time (USD)",
                        "xlabel": "Month",
                        "ylabel": "Amount (USD)"
                    }
                }

        # Default return → always wrap in dict
        return {
            "llm_response": response,
            "plot_data": None
        }


        