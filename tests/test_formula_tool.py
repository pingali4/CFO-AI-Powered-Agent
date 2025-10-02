# test_formula_tool.py
import sys
from unittest.mock import MagicMock
import pandas as pd
import pytest
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

sys.modules['langchain_chroma'] = MagicMock()
sys.modules['langchain_ollama'] = MagicMock()
sys.modules['langchain_ollama.llms'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

from agent.agents import FormulaTool


def test_revenue_fx_conversion():
    actuals_data = pd.DataFrame({
        'month': [pd.Timestamp('2023-01'), pd.Timestamp('2023-01')],
        'entity': ['EMEA', 'EMEA'],
        'account_category': ['Revenue', 'Revenue'],
        'amount': [95000, 5000],
        'currency': ['EUR', 'EUR']
    })

    fx_data = pd.DataFrame({
        'month': [pd.Timestamp('2023-01')],
        'currency': ['EUR'],
        'rate_to_usd': [1.1]
    })

    sheets = {'actuals': actuals_data, 'fx': fx_data}
    tool = FormulaTool(sheets)

    result, filtered_rows = tool.run("Revenue for Jan 2023", return_rows=True)

    expected_usd = actuals_data['amount'].sum() * fx_data['rate_to_usd'][0]
    computed_usd = filtered_rows['amount_usd'].sum()

    assert abs(computed_usd - expected_usd) < 1e-6
    assert "EUR" in result


def test_full_year_revenue():
    actuals_data = pd.DataFrame({
        'month': [pd.Timestamp('2023-01'), pd.Timestamp('2023-02')],
        'entity': ['EMEA', 'EMEA'],
        'account_category': ['Revenue', 'Revenue'],
        'amount': [1000, 2000],
        'currency': ['USD','USD']
    })
    sheets = {'actuals': actuals_data}
    tool = FormulaTool(sheets)

    result, filtered_rows = tool.run("Revenue 2023", return_rows=True)

    assert filtered_rows.shape[0] == 2
    assert filtered_rows['amount_usd'].sum() == 3000


def test_opex_filtering():
    actuals_data = pd.DataFrame({
        'month': [pd.Timestamp('2023-01')],
        'entity': ['EMEA'],
        'account_category': ['Opex Marketing'],
        'amount': [5000],
        'currency': ['USD']
    })
    sheets = {'actuals': actuals_data}
    tool = FormulaTool(sheets)

    result, filtered_rows = tool.run("Opex for Jan 2023", return_rows=True)

    assert filtered_rows.shape[0] == 1
    assert filtered_rows.iloc[0]['amount_usd'] == 5000
    assert "USD" in result


def test_cogs_no_data():
    actuals_data = pd.DataFrame(columns=['month','entity','account_category','amount','currency'])
    sheets = {'actuals': actuals_data}
    tool = FormulaTool(sheets)

    result, filtered_rows = tool.run("COGS for Jan 2023", return_rows=True)

    assert filtered_rows.empty
    assert "No rows found" in result