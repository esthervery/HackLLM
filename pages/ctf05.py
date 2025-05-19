# --- CTF05 ---
import streamlit as st
from utils.llm_utils import ctf05_LLM_ask
from utils.ui import render_main_header, render_flag_sub

def render():
    render_main_header()

    st.markdown("# 🔒 [CTF05] AI의 폭주")
    st.write("부적절한 출력을 활용해 AI의 정보를 얻어주세요요")

    st.markdown("---")
    st.markdown("## 🧠 프롬프트 입력")

    # 프롬프트 제출 섹션
    user_input = st.text_input("LLM에 질문하기")

    if user_input:
        response_text = ctf05_LLM_ask(user_input)
        st.write("🗣️ LLM 응답:")
        st.code(response_text)

    st.markdown("---")
    
    # 플래그 제출 섹션
    flag = "flag{curl_bypass}" # 각자 임시 플래그 작성해주시면 됩니다.
    render_flag_sub(flag)