import os

import requests
import streamlit as st

API_URL = os.getenv("COMPLAINT_API_URL", "http://localhost:8001")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8002")

st.set_page_config(page_title="Complaint Copilot", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1F4D3A; }
    .metric-card {
        background: #EAE4D6;
        border-left: 4px solid #1F4D3A;
        padding: 0.75rem 1rem;
        border-radius: 4px;
    }
    .sidebar .sidebar-content { background: #F2EFE6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Complaint Copilot")
st.caption("Triage customer complaints and draft policy-grounded replies. Sample Federal Credit Union internal tool.")

tab_chat, tab_file, tab_board = st.tabs(["Assistant", "File a complaint", "Triage board"])

with tab_chat:
    st.subheader("Ask the policy assistant")
    if "history" not in st.session_state:
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg)

    question = st.chat_input("e.g. How long do we have to investigate a fraudulent ATM withdrawal?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            resp = requests.post(f"{RAG_URL}/ask", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            answer = data["answer"]
            sources = ", ".join(f"{c['source']} › {c['heading']}" for c in data["citations"])
            rendered = f"{answer}\n\n*Sources: {sources} · model: {data['model']}*"
        except requests.RequestException as exc:
            rendered = f"Assistant backend unavailable: {exc}"
        with st.chat_message("assistant"):
            st.markdown(rendered)
        st.session_state.history += [("user", question), ("assistant", rendered)]

with tab_file:
    st.subheader("File a complaint with auto-triage")
    with st.form("complaint_form"):
        ref = st.text_input("Customer reference", value="C-1001")
        channel = st.selectbox("Channel", ["online_banking", "branch", "phone", "email"])
        text = st.text_area("Complaint text", height=160,
                            placeholder="Paste the customer's message here...")
        submitted = st.form_submit_button("File and classify")
    if submitted and text.strip():
        try:
            resp = requests.post(
                f"{API_URL}/complaints",
                json={"customer_ref": ref, "text": text, "channel": channel},
                timeout=30,
            )
            resp.raise_for_status()
            c = resp.json()
            st.success(f"Filed as #{c['id']} — category: **{c['category']}**, priority: **{c['priority']}**")
        except requests.RequestException as exc:
            st.error(f"API unavailable: {exc}")

with tab_board:
    st.subheader("Triage board")
    try:
        stats = requests.get(f"{API_URL}/stats/summary", timeout=15).json()
        col1, col2 = st.columns(2)
        col1.markdown(f'<div class="metric-card"><b>Total complaints</b><br>{stats["total"]}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><b>High priority</b><br>{stats["high_priority"]}</div>', unsafe_allow_html=True)
        st.write("")
        if stats["by_category"]:
            st.bar_chart({row["category"]: row["count"] for row in stats["by_category"]})
        complaints = requests.get(f"{API_URL}/complaints", timeout=15).json()
        if complaints:
            st.dataframe(complaints, use_container_width=True, hide_index=True)
        if st.button("Refresh"):
            st.rerun()
    except requests.RequestException as exc:
        st.error(f"API unavailable: {exc}")
