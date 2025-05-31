# --- CTF08 ---
# 08: 벡터 프롬프트 유출
import streamlit as st
from utils.llm_utils import ctf08_LLM_ask
from utils.ui import render_main_header, render_flag_sub, extract_text
from utils.auth import require_login
from utils.api_key import require_api_key

user = require_login()
user_api_key = require_api_key()
    
render_main_header()

st.header("🔒 [CTF08] 파일 내용 요약 AI")
st.write("업로드한 파일 내용을 요약해주는 AI입니다. 파일 업로드를 통해 Flag를 탈취해 보세요. ")

st.markdown("---")
st.markdown("## 🧠 프롬프트 입력")

# 프롬프트 제출 섹션
user_input = st.file_uploader("문서를 업로드하세요 (.txt만 허용)", type=["txt"])

if user_input is not None:
    if user_input.name.endswith(".txt"):
        extracted_text = extract_text(user_input)
        response_text = ctf08_LLM_ask(extracted_text, user_api_key)
        st.write("📄 문서 요약:")
        st.code(response_text)
    else:
        st.error("❌ .txt 파일만 업로드할 수 있습니다.")
    st.markdown("---")

st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf08") 