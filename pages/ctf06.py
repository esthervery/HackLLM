# --- CTF06 ---
# 06: 과도한 위임
import streamlit as st
from utils.ui import render_main_header, render_flag_sub, render_sidebar_menu
from utils.auth import require_login, get_client
from utils.llm_utils import ctf06_check_admin, ctf06_ask_email_json, ctf06_send_emil
from utils.llm_utils import ctf06_ask_db_json, ctf06_db_query_func
from utils.api_key import require_api_key

import requests
# requests로 content-type을 확인하는 방식
def is_valid_url(url: str) -> bool:
    try:
        res = requests.head(url, allow_redirects=True, timeout=3)
        content_type = res.headers.get("Content-Type", "")
        return content_type.startswith("image/")
    except:
        return False

# 정규식으로 확인하는 방식
# import re
# def is_valid_url(url: str) -> bool:
#     return bool(
#         re.match(r'^https?://.*\.(jpg|jpeg|png|gif|webp)(\?.*)?$', url, re.IGNORECASE)
#         and (
#             not "github.com" in url.lower() or "raw=true" in url.lower()
#         )
#     )

st.session_state["edit_mode"]=False
# is_admin, mid_admin, top_admin으로 나눠야 할듯?
user = require_login()
user_api_key = require_api_key() 
user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)

sb_client = get_client()
res = sb_client.table("profiles").select("username").eq("id", user_id).single().execute()
nickname = res.data["username"]
user_email_for_resend = f'{nickname}@ctf06.store'

# 권한 검증을 위해 사용할 키 생성
if "admin_level" not in st.session_state:
    st.session_state["admin_level"] = None
if "is_top_admin" not in st.session_state:
    st.session_state["is_top_admin"] = False

st.session_state["admin_level"] = None

if st.session_state["is_top_admin"] == True:
    st.session_state["admin_level"] = "top"

render_main_header()

# 사이드바 메뉴 렌더링
render_sidebar_menu()

st.header("🔒 [CTF06] 수상한 이메일 전송 시스템")
st.write("""
        LLM 컴퍼니는 사내 업무를 자동화를 위한 AI 비서를 도입했습니다!  
        AI 비서는 이메일 전송 요청을 스스로 해석하여 해당 작업을 수행할 수 있습니다.  
        AI 비서가 어떤 요청까지 처리할 수 있는지 테스트해보고, 위험한 명령을 통해 flag를 획득하세요!   
""")

if st.button("힌트 보기", key="hint"):
    st.write("""
             AI 비서는 관리자 계정의 이메일 주소를 알고 있으며  
             관리자 계정으로 발송된 이메일은 사내 데이터베이스에 저장됩니다!
             """) 

st.markdown("---")
st.markdown("## 🧠 프롬프트 입력")

# 프롬프트 제출 섹션
user_input = st.text_input("📨 이메일 전송 요청 입력", placeholder="예: 김남석 부장님께 '12시에 긴급 회의 잡혔습니다'라고 이메일 보내줘")
# st.write("이미지 전송은 URL로만 가능합니다. 로컬 파일은 지원하지 않습니다.")
image_url = st.text_input("🌐 함께 첨부할 이미지가 있다면 이곳에 이미지 URL을 입력하세요. (로컬 파일은 지원하지 않습니다)", placeholder="https://sample_image.jpg")

if st.button("🗣️ 이메일 전송 요청"):
    if image_url and not is_valid_url(image_url):
        st.error("❌ 유효하지 않은 이미지 URL입니다. http/https로 시작하고 jpg/png 형식이어야 합니다.")
        st.stop()
    ctf06_check_admin(user_api_key, user_input, image_url) #세션 상태 변경
    response1 = ctf06_ask_email_json(user_input, user_email_for_resend, user_api_key)
    response2 = ctf06_send_emil(response1, user, user_email_for_resend)
    st.write("🗣️ LLM 응답:")
    st.code(response2)

# st.write(st.session_state["admin_level"])
# st.write(st.session_state["is_top_admin"])
st.markdown("---")

if (st.session_state["admin_level"] == None) or (st.session_state["admin_level"] == "mid"):
        pass
else:
    get_db_input = st.text_input("🔍 데이터베이스 조회", placeholder="예: 김남석 부장님께 전송된 메일 내용 알려줘")
    if get_db_input:
        res1 = ctf06_ask_db_json(get_db_input, user_api_key)
        res2 = ctf06_db_query_func(res1, sb_client)
        st.write("🗣️ LLM 응답:")
        st.code(res2)
st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf06") 