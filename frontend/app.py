import streamlit as st
import requests

st.set_page_config(page_title="LLM Recommender (Chat)", layout="centered")
st.title("🛍️ LLM Recommender — Chat Agent")

BACKEND = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hi! Tell me what you want. Example:\n'recommend eco-friendly bamboo kitchen items under 1000, rating above 4, no plastic'."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your request…"):
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{BACKEND}/chat",
                                  json={"message": prompt, "diversity_lambda": 0.6, "return_why_not": True},
                                  timeout=60)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

            results = data.get("results", [])
            if not results:
                st.markdown("_No results. Try relaxing constraints or rephrasing._")
            else:
                st.subheader("Top recommendations")
                lines = []
                for i, it in enumerate(results, 1):
                    line = f"**#{i}. {it.get('title','(no title)')}** — ₹{it.get('price','?')} | ⭐ {it.get('rating','?')} | score {round(it.get('score',0),1)}\n\n" \
                           f"_Why:_ {it.get('reason','')}"
                    lines.append(line)
                st.markdown("\n\n---\n\n".join(lines))

            why_not = data.get("why_not", [])
            if why_not:
                with st.expander(f"Why-NOT items ({len(why_not)})"):
                    for row in why_not[:50]:
                        st.markdown(f"- **{row.get('title','(item)')}** → {row.get('why_not','')}")

    st.rerun()
