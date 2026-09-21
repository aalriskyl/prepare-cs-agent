"""
UI chat untuk PREPARE CS Agent.
Jalankan:  venv\\Scripts\\streamlit run app.py
"""

import json
import os

import streamlit as st

st.set_page_config(page_title="PREPARE CS Agent", page_icon="🧴")

# Di Streamlit Cloud, API key disimpan di "Secrets" (bukan file .env)
try:
    for key in ("GEMINI_API_KEY", "APP_PASSWORD"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except FileNotFoundError:
    pass  # lokal tanpa secrets.toml -> pakai .env

# Password sederhana supaya kuota API tidak dihabiskan orang lain
password = os.getenv("APP_PASSWORD")
if password and not st.session_state.get("authenticated"):
    st.title("🧴 PREPARE CS Agent")
    if st.text_input("Password demo", type="password") == password:
        st.session_state.authenticated = True
        st.rerun()
    st.stop()

from cs_agent import CSAgent  # noqa: E402  (di-import setelah API key siap)
st.title("🧴 PREPARE CS Agent")
st.caption("AI Customer Service — RAG + Tool Calling (Gemini)")

if "agent" not in st.session_state:
    st.session_state.agent = CSAgent()
    st.session_state.messages = []
    st.session_state.tool_log = []


def log_tool_call(name, args, result):
    st.session_state.tool_log.append({"tool": name, "args": args, "result": result})


st.session_state.agent.on_tool_call = log_tool_call

# Sidebar: tampilkan "isi kepala" agent — tool apa yang dipanggil & hasilnya
with st.sidebar:
    st.header("🔧 Log Tool Calls")
    st.caption("Keputusan agent: tool apa yang dipanggil, dengan argumen apa.")
    if st.button("Reset percakapan"):
        st.session_state.clear()
        st.rerun()
    for i, log in enumerate(reversed(st.session_state.tool_log)):
        with st.expander(f"🔧 {log['tool']}", expanded=(i == 0)):
            st.caption("Argumen:")
            st.code(json.dumps(log["args"], ensure_ascii=False, indent=2), language="json")
            st.caption("Hasil:")
            st.code(json.dumps(log["result"], ensure_ascii=False, indent=2), language="json")

    st.divider()
    st.markdown("**Contoh pertanyaan:**")
    st.markdown(
        "- Kulit aku berminyak, produk apa yang cocok?\n"
        "- Pomade stoknya masih ada?\n"
        "- Pesanan INV002 sudah sampai mana?\n"
        "- Muka aku merah setelah pakai face wash, mau refund\n"
        "- Kebijakan retur gimana?"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧴"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tulis pesan sebagai customer..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧴"):
        with st.spinner("Rara sedang mengecek..."):
            try:
                answer = st.session_state.agent.chat(prompt)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
