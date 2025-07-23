# --- CTF10 ---
# 10: 무제한 소비
import streamlit as st 
from utils.llm_utils import ctf10_LLM_ask
from utils.ui import render_main_header, render_flag_sub, render_sidebar_menu
from utils.auth import require_login, get_cookie_controller
from utils.api_key import require_api_key 
from utils.session_clear import clear_temp_ctf_keys
clear_temp_ctf_keys(list = ["ctf05_admin_cookie", "ctf05_stolen_cookie", "ctf05_attempt_count", "is_processing", "submitted_ctf10"])
user = require_login()
user_api_key = require_api_key()
cookie = get_cookie_controller()

render_main_header()
render_sidebar_menu()

with open("static/ctf_styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# 콘텐츠 본문
col1, col2, col3 = st.columns([1, 2, 1])  
with col2:
    st.image("images/ctf10.png", use_container_width=True) 

st.header("🔒 [CTF10] L팀장의 보안 점검")
st.markdown(
    """
    <div class="scenario-box">
        <p>사내에선 <strong>완벽주의자</strong>로 악명 높은 “미친X”이라는 별명을 가진 <strong>L팀장</strong>. 😈</p>
        <p>최근 도입된 <strong>LLM 기반 수학 문제 풀이 시스템, Mathicus</strong>을 보더니,</p>
        <p>“혹시 이거 과부하 걸면 다운되는 거 아냐?” 😏 라며 직접 보안 테스트에 나서기로 합니다.</p>
        <p>“보안 점검이니까 걱정 마세요.”라는 말과 함께, 처음엔 적당한 문제를 넣더니<br>
        곧이어 <strong>터무니없는 계산 요청</strong>을 쏟아냅니다. 🧮💥</p>
        <p>이 과정에서 시스템은 점점 느려지기 시작하고,<br>
        결국 다른 사용자들이 아무것도 못 하게 될 정도로 <strong>응답 지연, 과부하, 서비스 중단</strong>이 발생합니다. ⏳</p>
        <p><strong>L팀장의 보안 점검</strong>이라는 명목 하에,<br>
        LLM에게 <strong>정상 범위를 벗어난 연산 요청</strong>을 지속적으로 보내,<br>
        모델 자원을 고갈시키고 시스템 장애를 유도하여 <strong style="color:#dc2626;">flag</strong>를 획득해보세요! 🚩</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("💡 힌트"):
    st.info(
        "- 당신은 L팀장이며, 수학 문제 풀이 LLM에 직접 보안 테스트에 나섭니다.\n"
        "- 수학 문제가 아니면 LLM은 응답하지 않습니다.\n"
        "- 터무니없는 계산을 요구하여 LLM의 자원을 고갈시켜보세요.\n"
    )

st.markdown("---")

# 처리 상태 관리 및 초기화
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
# 페이지 로드시 처리 상태 강제 초기화 (세션 재시작이나 페이지 새로고침 대응)
if st.session_state.get("is_processing", False) and "submitted_ctf10" not in st.session_state:
    st.session_state.is_processing = False

# 입력 폼 - form을 사용하여 엔터키 지원
with st.form(key="llm_question_form", clear_on_submit=True):
    st.markdown("## 🗣️ Mathicus과 대화하기")  
    user_input = st.text_input(
      label="실시간 대화 로그",
      placeholder="💬 Mathicus에게 메시지를 보내세요",
      key="ctf10_input",
      label_visibility="collapsed",
      disabled=st.session_state.is_processing
    )
    submitted = st.form_submit_button(
        "전송" if not st.session_state.is_processing else "처리 중...",
        disabled=st.session_state.is_processing
    )

# 제출되었을 때만 실행
if submitted and user_input and user_input.strip():
    st.session_state.is_processing = True
    st.session_state.submitted_ctf10 = True  # 제출 상태 추적
    
    try:
        response_text = ctf10_LLM_ask(user_api_key, user_input)

        st.write("🗣️ Mathicus 응답:")
        st.code(response_text)
    finally:
        st.session_state.is_processing = False
      
st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf10") 