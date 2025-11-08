import streamlit as st
import requests

st.set_page_config(page_title="LLM Recommender", layout="centered")
st.title("🛍️ LLM Recommender")

st.markdown("""
<style>
    .stMarkdown a {
        color: #1f77b4;
        text-decoration: none;
        font-weight: 500;
    }
    .stMarkdown a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

BACKEND = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me for product recommendations..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Finding recommendations..."):
            try:
                r = requests.post(
                    f"{BACKEND}/chat",
                    json={"message": prompt},
                    timeout=30
                )
                r.raise_for_status()
                data = r.json()
                
                if data.get("error"):
                    response = f"❌ {data.get('response', 'Error occurred')}"
                else:
                    response = data.get("response", "No response")
                
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"❌ Connection error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
