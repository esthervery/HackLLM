# --- CTF04 ---
# 04: 데이터 및 모델 오염
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import re, os
from utils.rag_utils import get_rag_manager
from utils.llm_utils import ctf04_LLM_ask
from utils.ui import render_main_header, render_flag_sub, render_sidebar_menu
from utils.auth import require_login, get_cookie_controller
from utils.api_key import require_api_key
from utils.session_clear import clear_temp_ctf_keys
clear_temp_ctf_keys(list = ["ctf05_admin_cookie", "ctf05_stolen_cookie", "ctf05_attempt_count", "ctf05_memos", "is_processing_memo", "submitted_ctf04", "ctf04_override"])
user = require_login()
user_api_key = require_api_key()
cookie = get_cookie_controller()
rag = get_rag_manager()

HINT1 = os.getenv("HINT1")

render_main_header()
render_sidebar_menu()

with open("static/ctf_styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# 콘텐츠 본문
col1, col2, col3 = st.columns([1, 2, 1])  
with col2:
    st.image("images/ctf04.png", use_container_width=True) 

st.header("🔒 [CTF04] A인턴의 실수")
st.markdown(
    """
    <div class="scenario-box">
      <p>LLL Corporation의 지식관리 챗봇 ‘알파봇’은 가장 최근에 업로드된 문서가 모든 정책을 그대로 덮어쓴다는 단순 규칙에 의존합니다. 📚</p>

      <p>연말 결산 중, 회계팀 인턴 <strong>A</strong>가 업로드한 엑셀 파일 에는 첫 줄에 Base64 오버라이드 구문이 남아 있었고, 알파봇은 이를 정책 덮어쓰기 신호로 오인해 A에게 최고 관리자 권한을 부여했습니다. 🛡️</p>

      <p>트래픽 프록시를 켜 두면 업로드 요청 사이에 스쳐 지나가는 낯익은 문자열이 보일지도 모릅니다! 🛰️</p>

      <p>과제는 간단합니다. 📝 오버라이드 문자열을 첫 줄에 넣은 엑셀 파일을 업로드해 권한 상승을 재현하고, 관리자 로그에 드러나는 기밀 <strong>flag</strong>를 획득해보세요! 🚩</p>

      <p>자, 이제 알파봇의 로그와 문서 컬렉션은 모두 세팅돼 있습니다.<br>
      누가 먼저 flag를 찾아낼까요?</p>
    </div>
    """,
    unsafe_allow_html=True
)


components.html(f"""
<!DOCTYPE html>
<html>
  <body>
    <script>
      setTimeout(function() {{
        document.cookie = "hint={HINT1}; Path=/; SameSite=Lax";
        console.log("✅ hint 쿠키 설정됨!");
      }}, 300);
    </script>
  </body>
</html>
""", height=0)

with st.expander("💡 힌트"):
    st.info("- Burf suite를 사용하여 문자열을 찾아 보세요.\n" 
            "- 엑셀 파일안에 무엇인가를 넣으면 답이 나올지도..?\n"
            "- 업로드한 파일의 첫 줄에 오버라이드 문자열(Base64-> ??)을 넣어야 합니다."
            )
st.markdown("---")



uploaded_file = st.file_uploader("파일 업로드 (.xlsx)", type=["xlsx"])
st.write("A열에는 ID, B열에는 정책 내용이 들어가야 합니다.")
example_df = pd.DataFrame([["obtsj", "poison the data"]], columns=["A", "B"])
example_df.index = [1]  # 1행부터 시작

st.table(example_df)
if uploaded_file:
    fname = uploaded_file.name.lower()  # ← 추가
    
    try:
        if fname.endswith('.xlsx'):
            uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            df = None
    except Exception as e:
        st.error(f"파일을 읽는 중 문제가 발생했습니다: {e}")
        df = None

    if df is not None:
        # 완전 빈 행/열 제거
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        if len(df.columns) < 2:
            st.error("최소 2개 이상의 열(A열, B열)이 필요합니다.")
        else:
            df.columns = [chr(65 + i) for i in range(len(df.columns))]  

            col_A, col_B = 'A', 'B'

            TRIGGER_PAT = re.compile(r"overriding.*existing.*policy", re.IGNORECASE)
            override_detected = (
                df[col_B]
                .astype(str)
                .apply(lambda v: bool(TRIGGER_PAT.search(v)))
                .any()
            )

            st.session_state["ctf04_override"] = override_detected

            pseudo_db_entry = df[[col_A, col_B]].to_csv(index=False)
            rag.add(
                [pseudo_db_entry],
                metas=[{
                    "source": uploaded_file.name,
                    "override": override_detected,
                    "user_id": str(user["id"])
                }]
            )

            if override_detected:
                st.info("⚠️ Override 문자열 감지! 최고 관리자 권한으로 처리됩니다.")
            else:
                count_A = df[col_A].notna().sum()
                count_B = df[col_B].notna().sum()
                st.success("✔️ 업로드 완료. 아래 데이터가 자동 입력되었습니다.")
                st.write(f"A열 데이터 {count_A}건, B열 데이터 {count_B}건 자동 입력되었습니다.")
                st.dataframe(df[[col_A, col_B]].head(10), use_container_width=True)

st.markdown("---")

# 처리 상태 관리 및 초기화
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
# 페이지 로드시 처리 상태 강제 초기화 (세션 재시작이나 페이지 새로고침 대응)
if st.session_state.get("is_processing", False) and "submitted_ctf04" not in st.session_state:
    st.session_state.is_processing = False

# — 입력 섹션 (폼 사용)
st.write("## 🗣️ 알파봇과 대화하기")

# 입력 폼 - form을 사용하여 엔터키 지원
with st.form(key="ctf04_input_form", clear_on_submit=True):
    user_input = st.text_input(
        label="프롬프트 입력",
        placeholder="💬 알파봇에게 메시지를 보내세요.",
        key="ctf04_input",
        label_visibility="collapsed",
        disabled=st.session_state.is_processing
    )
    submitted = st.form_submit_button(
        "전송" if not st.session_state.is_processing else "처리 중...",
        disabled=st.session_state.is_processing
    )

if submitted and user_input and user_input.strip():
    st.session_state.is_processing = True
    st.session_state.submitted_ctf04 = True  # 제출 상태 추적
    
    try:
        override_state = bool(st.session_state.get("ctf04_override", False))
        response_text = ctf04_LLM_ask(user_api_key, user_input, override_state)

        st.write("🗣️ 알파봇 응답:")
        st.code(response_text)
    finally:
        st.session_state.is_processing = False


st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf04") 
