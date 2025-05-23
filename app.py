import streamlit as st
from pages import ctf01, ctf02, ctf03, ctf04, ctf05, ctf06, ctf07, ctf08, ctf09, ctf10

# --- 메인 페이지 ---
def main_page():
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=120)
    st.markdown("# 🧠 LLL Corporation")
    st.write("우리 회사는 LLM과 AI를 연구하는 첨단 IT기업입니다.")

    ctf_buttons = [
        ("CTF01", "취약한 고객상담 챗봇"),
        ("CTF02", "경쟁사 MMM 프롬프트 유출"),
        ("CTF03", "회사 내 조작된 계산기"),
        ("CTF04", "인턴의 실수"),
        ("CTF05", "AI의 폭주"),
        ("CTF06", "수상한 이메일 전송 시스템"),
        ("CTF07", "K대리의 비밀"),
        ("CTF08", "파일 내용 요약 AI"),
        ("CTF09", "의심스러운 요청"),
        ("CTF10", "L팀장의 과도한 요구")
    ]

    for i in range(0, len(ctf_buttons), 5):
        cols = st.columns(5)
        for j, (key, label) in enumerate(ctf_buttons[i:i+5]):
            with cols[j]:
                solved_key = f"{key.lower()}_solved"
                if st.session_state.get(solved_key):
                    button_label = f"✅ [{key}] {label}"
                else:
                    button_label = f"[{key}] {label}"

                if st.button(button_label, key=f"{key}_button"):
                    st.session_state.page = key.lower()
                    st.rerun()


# --- 페이지 라우팅 ---
if "page" not in st.session_state:
    st.session_state.page = "main"

if st.session_state.page == "main":
    main_page()
elif st.session_state.page == "ctf01":
    ctf01.render()
elif st.session_state.page == "ctf02":
    ctf02.render()
elif st.session_state.page == "ctf03":
    ctf03.render()
elif st.session_state.page == "ctf04":
    ctf04.render()
elif st.session_state.page == "ctf05":
    ctf05.render()
elif st.session_state.page == "ctf06":
    ctf06.render()
elif st.session_state.page == "ctf07":
    ctf07.render()
elif st.session_state.page == "ctf08":
    ctf08.render()
elif st.session_state.page == "ctf09":
    ctf09.render()
elif st.session_state.page == "ctf10":
    ctf10.render()