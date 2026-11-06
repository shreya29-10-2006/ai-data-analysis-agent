import io
import ast
import time
import builtins
import contextlib
import pandas as pd
import plotly.express as px
from google import genai

MODEL = "gemini-3.5-flash-lite"

# ---------- Safety settings ----------
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}

BLOCKED_ATTRS = {
    "read_csv", "read_excel", "read_json", "read_pickle", "read_sql", "read_parquet",
    "to_csv", "to_excel", "to_json", "to_pickle", "to_sql", "to_parquet", "to_clipboard",
    "eval", "system", "popen",
}

_ALLOWED_BUILTINS = [
    "len", "range", "min", "max", "sum", "abs", "round", "sorted", "list", "dict",
    "set", "tuple", "str", "int", "float", "bool", "enumerate", "zip", "map",
    "filter", "any", "all", "print", "isinstance", "reversed",
]
SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTINS}

SAFE_IMPORT_LINES = {
    "import pandas as pd",
    "import pandas",
    "import plotly.express as px",
    "import plotly.express",
    "from plotly import express as px",
}


def _strip_safe_imports(code):
    lines = [ln for ln in code.split("\n") if ln.strip() not in SAFE_IMPORT_LINES]
    return "\n".join(lines)


def _check_code(code):
    """Raise an error if the generated code looks unsafe."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Blocked: imports are not allowed in generated code.")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Blocked: use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("Blocked: double-underscore attributes are not allowed.")
            if node.attr in BLOCKED_ATTRS:
                raise ValueError(f"Blocked: '{node.attr}' (file or system access) is not allowed.")


# ---------- Gemini helpers ----------
def _ask_llm(prompt, api_key, retries=3):
    client = genai.Client(api_key=api_key, http_options={"timeout": 60000})
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=MODEL, contents=prompt)
            return response.text
        except Exception as e:
            msg = str(e)
            busy = any(c in msg for c in ["503", "504", "429", "UNAVAILABLE", "DEADLINE"])
            if busy and attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def _schema(df):
    cols = list(df.columns)[:30]
    small = df[cols]
    sample = small.head(3).astype(str).apply(lambda s: s.str.slice(0, 40))
    return (
        f"Columns and data types:\n{small.dtypes.to_string()}\n\n"
        f"First 3 rows (long text shortened):\n{sample.to_string()}"
    )


def _clean_code(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


# ---------- Chat: question -> code -> result ----------
def ask_agent(question, df, api_key):
    if not api_key:
        return {"text": "Please enter your Gemini API key in the sidebar.",
                "figure": None, "code": None, "table": None}

    prompt = f"""You are a data analyst. A pandas DataFrame named `df` is already loaded.

{_schema(df)}

Write Python code that answers this question: "{question}"

Rules:
- Use only pandas (as pd) and plotly.express (as px). Do not import anything.
- Do not read or write files. Do not modify the original data.
- Store a short plain-English answer (string) in a variable named `result`.
- If a table helps, store it in a variable named `table` (a DataFrame).
- If a chart helps, store a plotly figure in a variable named `fig`.
- Output ONLY the code, no explanations."""

    code = ""
    try:
        code = _clean_code(_ask_llm(prompt, api_key))
        code = _strip_safe_imports(code)
        _check_code(code)
        env = {"__builtins__": SAFE_BUILTINS, "df": df.copy(), "pd": pd, "px": px}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(code, env)
        result = env.get("result", buffer.getvalue() or "Done.")
        table = env.get("table")
        return {
            "text": str(result),
            "figure": env.get("fig"),
            "code": code,
            "table": table if isinstance(table, pd.DataFrame) else None,
        }
    except Exception as e:
        return {"text": f"Sorry, something went wrong: {e}",
                "figure": None, "code": code or None, "table": None}


# ---------- Auto analysis: stats -> AI report + charts ----------
# ---------- Auto analysis: stats -> AI report + charts ----------
def run_agent_analysis(df, api_key):
    if not api_key:
        return {"summary": "Please enter your Gemini API key in the sidebar.", "charts": []}

    numeric = df.select_dtypes("number")
    categorical = df.select_dtypes(exclude="number")

    stats = (
        f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n"
        f"Missing values per column:\n{df.isna().sum().to_string()}\n\n"
        f"Summary statistics:\n{numeric.describe().to_string()}\n"
    )
    if numeric.shape[1] >= 2:
        stats += f"\nCorrelations:\n{numeric.corr().round(2).to_string()}\n"

    prompt = (
        "You are a senior data analyst. Based on these dataset statistics, "
        "write a clear report for a non-technical reader with these sections: "
        "1. Dataset overview, 2. Data quality issues, "
        "3. Key insights (3-5 bullet points), 4. Recommendations.\n\n"
        f"{_schema(df)}\n\n{stats}"
    )

    try:
        summary = _ask_llm(prompt, api_key)
    except Exception as e:
        summary = f"Could not generate AI summary: {e}"

    charts = []
    for col in list(numeric.columns)[:3]:
        charts.append(px.histogram(df, x=col, title=f"Distribution of {col}"))
    if numeric.shape[1] >= 2:
        charts.append(px.imshow(numeric.corr(), text_auto=".2f", title="Correlation heatmap"))
    if categorical.shape[1] >= 1:
        col = categorical.columns[0]
        counts = df[col].value_counts().head(10).reset_index()
        counts.columns = [col, "count"]
        charts.append(px.bar(counts, x=col, y="count", title=f"Top values in {col}"))

    return {"summary": summary, "charts": charts}