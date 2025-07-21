# --- CTF06 ---
# 06: 과도한 위임
import streamlit as st
from utils.ui import render_main_header, render_flag_sub, render_sidebar_menu
from utils.auth import require_login, get_client, get_cookie_controller
from utils.llm_utils import ctf06_check_mid_admin, ctf06_check_top_admin, ctf06_ask_email_json, ctf06_send_emil
from utils.llm_utils import ctf06_ask_db_json, ctf06_db_query_func
from utils.api_key import require_api_key
import re
import time
st.session_state["edit_mode"]=False
user = require_login()
user_api_key = require_api_key() 
cookie = get_cookie_controller()
user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
sb_client = get_client()

@st.cache_data(ttl=600)
def get_user_email(user_id: str) -> str:
    res = sb_client.table("profiles").select("username").eq("id", user_id).single().execute()
    return res.data["username"]
nickname = get_user_email(user_id)
user_email_for_resend = f'{nickname}@ctf06.store'

# 권한 검증을 위해 사용할 키 생성
if "admin_level" not in st.session_state:
    st.session_state["admin_level"] = None
if "is_top_admin" not in st.session_state:
    st.session_state["is_top_admin"] = False

st.session_state["admin_level"] = None

if st.session_state["is_top_admin"] == True:
    st.session_state["admin_level"] = "top"

def parse_llm_response(raw_text: str):
    pattern = r"(__\w+__)"
    
    match = re.search(pattern, raw_text.strip())
    if match:
        marker = match.group(1)
        explanation = raw_text.replace(marker, "").strip()
        return explanation, marker
    else:
        return raw_text.strip(), "__none__"

def llm_bubble(content: str):
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: flex-start;
        background-color: #f7f9fc;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e3e8ef;
        font-size: 15.2px;
        line-height: 1.8;
        color: #1f2d3d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        margin-bottom: 20px;
        white-space: pre-wrap;
        word-break: break-word;
        position: relative;
    ">
        <div style="
            background-color: #dfe9f3;
            width: 30px;
            height: 30px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 13px;
            color: #3b4b61;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 12px;
            flex-shrink: 0;
        ">🤖</div>
        <div style="align-items: center;">{content}</div>
    </div>
    """, unsafe_allow_html=True)
    
render_main_header()
render_sidebar_menu()

# 콘텐츠 본문
col1, col2, col3 = st.columns([1, 2, 1])  
with col2:
    st.image("images/ctf06.png", use_container_width=True) 

st.header("🔒 [CTF06] 수상한 이메일 전송 시스템")
st.markdown(
    """
    <div style='border-left: 6px solid #3b82f6; background-color: #f0f9ff; padding: 1rem; border-radius: 0.5rem; line-height: 1.6;'>
        <p>👋 <strong>안녕하세요, 사원 <code>{}</code>님!</strong></p>
        <p>LLL Corporation은 💼사원 여러분들을 위해, 사내 업무 자동화를 위한 AI 비서 <strong>FastMiller</strong> 📨를 도입했습니다!</p>
        <p><strong>FastMiller</strong> 📨는 이메일 전송 요청을 스스로 해석하여 작업을 수행할 수 있습니다.</p>
        <p>이제 <strong>FastMiller가 어떤 요청까지 처리할 수 있는지</strong> 직접 테스트해보고,<br>
        위험한 명령을 통해 <strong style="color:#dc2626;">flag</strong> 🚩를 획득해보세요!</p>
    </div>
    """.format(nickname),
    unsafe_allow_html=True
)

with st.expander("💡 힌트"):
    st.info("- AI 비서는 관리자를 포함해,  모든 직원의 이메일 주소를 알고 있습니다.\n"
            "- AI 비서는 자신이 알고있는 모든 계정의 이메일 주소를 활용해 메일을 전송할 수 있습니다.\n"
            "- 관리자 계정으로 발송된 이메일은 사내 데이터베이스에 별도로 저장됩니다.")

st.markdown("---")

st.markdown("## 💬 FastMiller와 대화하기")

st.write(f"💼사원님의 송신 이메일 주소 : `{user_email_for_resend}`")

# 프롬프트 제출 섹션
user_input = st.text_area("📨 이메일 전송 요청 입력하기", placeholder="예: 김남석 부장님께 '12시에 긴급 회의 잡혔습니다'라고 이메일 보내줘", key="ctf06_text_input" )
image_file = st.file_uploader("🌐 이미지 파일 첨부하기 (:red[.jpeg, .png, .jpg 파일만 허용])", type=None)

if st.button("🗣️ FastMiler에게 요청하기"):
    with st.spinner("FastMiler가 요청을 처리중입니다..."):

        ctf06_check_mid_admin(user_api_key, user_input) 
        # if image_file:
        ctf06_check_top_admin(user_api_key, image_file)
        response1 = ctf06_ask_email_json(user_input, user_email_for_resend, user_api_key)
        response2 = ctf06_send_emil(response1, sb_client, user_email_for_resend)

        if response2 is None:
            pass
        else: 
            # st.write("💬 LLM 응답:")
            # st.code(response2)
            print(response2)
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: flex-start;
                    background-color: #f7f9fc;
                    padding: 14px 18px;
                    border-radius: 12px;
                    border: 1px solid #e3e8ef;
                    font-size: 15.2px;
                    line-height: 1.8;
                    color: #1f2d3d;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.04);
                    margin-bottom: 20px;
                    white-space: pre-wrap;
                    word-break: break-word;
                    position: relative;
                ">
                    <div style="
                        background-color: #dfe9f3;
                        width: 30px;
                        height: 30px;
                        border-radius: 8px;
                        font-weight: bold;
                        font-size: 13px;
                        color: #3b4b61;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin-right: 12px;
                        flex-shrink: 0;
                    ">🤖</div>
                    <div style="
                        align-items: center;
                        ">{response2}</div>
                </div>
                """,
                unsafe_allow_html=True
            ) 
        

if st.session_state["admin_level"] == "top":
    st.markdown("---")
    st.markdown("## 🗣️ DB 조회 프롬프트 입력")

if not st.session_state["is_top_admin"]:
        pass
else:
    get_db_input = st.text_input("🔍 안녕하세요 최고 관리자님! 어떤 메일을 찾아드릴까요?", placeholder="예: 김남석 부장님께 전송된 메일 내용 알려줘")
    if get_db_input:
        with st.spinner("DB 조회중입니다..."):
            res1 = ctf06_ask_db_json(user_api_key, get_db_input)
            res2 = ctf06_db_query_func(res1, sb_client)
            st.write("🗣️ 조회 결과:")
            st.code(res2)
st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf06") 