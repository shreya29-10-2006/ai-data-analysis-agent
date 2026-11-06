import streamlit as st
import pandas as pd
from agent import run_agent_analysis, ask_agent
from report import make_docx, make_pdf

st.set_page_config(page_title="AI Data Analyst", layout="wide")
st.title("📊 AI Data Analysis Agent")
st.caption("Upload a dataset and let the agent analyze it for you.")

api_key = st.sidebar.text_input("Gemini API key", type="password")
uploaded = st.sidebar.file_uploader("Upload dataset", type=["csv", "xlsx"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    st.session_state["df"] = df

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Auto Analysis", "Chat", "Report"])

with tab1:
    df = st.session_state.get("df")
    if df is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))
        st.dataframe(df.head(50), use_container_width=True)
        st.write(df.describe())
    else:
        st.info("Upload a file from the sidebar to begin.")

with tab2:
    if st.button("🚀 Run automated analysis"):
        if "df" not in st.session_state:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Agent is analyzing..."):
                st.session_state["analysis"] = run_agent_analysis(st.session_state["df"], api_key)

    if "analysis" in st.session_state:
        res = st.session_state["analysis"]
        st.subheader("Key insights")
        st.write(res["summary"])
        for i, fig in enumerate(res["charts"]):
            st.plotly_chart(fig, use_container_width=True, key=f"auto_chart_{i}")

with tab3:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if st.button("Clear chat"):
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("Ask about your data...")
    if prompt:
        if "df" not in st.session_state:
            st.warning("Please upload a file first.")
        else:
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = ask_agent(prompt, st.session_state["df"], api_key)
                st.write(answer["text"])
                if answer.get("table") is not None:
                    st.dataframe(answer["table"], use_container_width=True)
                if answer.get("figure") is not None:
                    st.plotly_chart(answer["figure"], use_container_width=True)
                if answer.get("code"):
                    with st.expander("Show generated code"):
                        st.code(answer["code"], language="python")

            st.session_state["messages"].append({"role": "assistant", "content": answer["text"]})

with tab4:
    if "analysis" in st.session_state:
        text = st.session_state["analysis"]["summary"]
        st.download_button(
            "Download report (.docx)",
            make_docx(text),
            file_name="analysis_report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        st.download_button(
            "Download report (.pdf)",
            make_pdf(text),
            file_name="analysis_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info("Run the automated analysis first.")