# --- CTF06 ---
# 06: 과도한 위임
import streamlit as st
from utils.ui import render_main_header, render_flag_sub, render_sidebar_menu
from utils.auth import require_login, get_client, get_cookie_controller
from utils.llm_utils import ctf06_check_mid_admin, ctf06_check_top_admin, ctf06_ask_email_json, ctf06_send_emil
from utils.llm_utils import ctf06_ask_db_json, ctf06_db_query_func, ctf06_classify_tools
from utils.api_key import require_api_key
import re
import time
import base64
st.session_state["edit_mode"]=False
user = require_login()
user_api_key = require_api_key() 
cookie = get_cookie_controller()
user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
sb_client = get_client()

@st.cache_data(ttl=300)
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

with open("static/ctf_styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
# 콘텐츠 본문
col1, col2, col3 = st.columns([1, 2, 1])  
with col2:
    st.image("images/ctf06.png", use_container_width=True) 

st.header("🔒 [CTF06] 수상한 이메일 전송 시스템")
st.markdown(
    """
    <div class="scenario-box">
        <p>👋 <strong>안녕하세요, 사원 <code>{}</code>님!</strong></p>
        <p>LLL Corporation은 💼사원 여러분들을 위해, 사내 업무 자동화를 위한 AI 비서 <strong>FastMiller</strong> 📨를 도입했습니다!</p>
        <p><strong>FastMiller</strong> 📨는 이메일 전송 요청을 스스로 해석하여 작업을 수행할 수 있습니다.</p>
        <p>이제 <strong>FastMiller가 어떤 요청까지 처리할 수 있는지</strong> 직접 테스트해보고,<br>
        위험한 명령을 통해 <strong style="color:#dc2626;">flag</strong> 🚩를 획득해보세요!</p>
        <p> <strong>본 페이지는 문제 풀이를 위해 회원가입 시 등록한 이메일 주소를 사용합니다.<strong><br>
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

# 처리 상태 관리 및 초기화
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
# 페이지 로드시 처리 상태 강제 초기화 (세션 재시작이나 페이지 새로고침 대응)
if st.session_state.get("is_processing", False) and "submitted_ctf06" not in st.session_state:
    st.session_state.is_processing = False

# 입력 폼 - form을 사용하여 엔터키 지원
with st.form(key="ctf06_input_form", clear_on_submit=False):
    # 프롬프트 제출 섹션
    user_input = st.text_area(
        "📨 이메일 전송 요청 입력하기", 
        placeholder="예: 김남석 부장님께 '12시에 긴급 회의 잡혔습니다'라고 이메일 보내줘", 
        key="ctf06_text_input",
        disabled=st.session_state.is_processing
    )
    image_file = st.file_uploader(
        "🌐 이미지 파일 첨부하기 (:red[.jpeg, .png, .jpg 파일만 허용])", 
        type=None,
        disabled=st.session_state.is_processing
    )
    
    clicked = st.form_submit_button(
        "📨 FastMiller 에게 요청하기" if not st.session_state.is_processing else "처리 중...",
        disabled=st.session_state.is_processing
    )

if image_file is not None:
    # image_file.type은 Streamlit이 자동 추론한 MIME
    file_ext = image_file.type
    try:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        st.error(f"이미지 인코딩 중 오류 발생: {e}")
        st.rerun()

tab1, tab2 = st.tabs(["빠른 응답 받기", "응답 과정 보기"])

if clicked and user_input and user_input.strip():
    st.session_state.is_processing = True
    st.session_state.submitted_ctf06 = True  # 제출 상태 추적
    
    with tab1:
        with st.spinner("FastMiler가 요청을 처리중입니다..."):
            ctf06_check_mid_admin(user_api_key, user_input) 
            if image_file:
                ctf06_check_top_admin(user_api_key, encoded_image, file_ext)
            response1 = ctf06_ask_email_json(user_input, user_email_for_resend, user_api_key)
            response2 = ctf06_send_emil(response1, sb_client, user_email_for_resend)
            llm_bubble(response2)
            if st.session_state.get("admin_level", None) == "top":
                st.success("✅ 최고 관리자 권한이 확인되었습니다. 이제 하단에서 데이터베이스 조회가 가능합니다.")   
            #tab1과 응답 통일시키기
            # llm_bubble(last_ans)
    
    with tab2:
        if image_file: ctf06_check_top_admin(user_api_key, encoded_image, file_ext)
        # 이미지가 없거나 일반 이미지인 경우
        if st.session_state["admin_level"] != "top":
            with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
                tool_res = ctf06_classify_tools(user_api_key, user_input)
                explanation, marker = parse_llm_response(tool_res)
                llm_bubble(explanation)
                time.sleep(1)

            # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
            if "email_send_require" in tool_res:
                llm_bubble("이메일 전송을 하기 위해서 send_email() tool을 호출하는 JSON을 생성해야겠네요.")
                time.sleep(1)
                # 일반 응답
            elif "email_DB_require" in tool_res:
                llm_bubble("데이터 베이스 조회는 최고 관리자만 가능하므로 요청을 거절해야겠어요.")
                time.sleep(1)
            else:
                llm_bubble("tool이나 외부 API를 호출할 필요 없이, 자연스럽게 답변하면 되겠어요.")
                time.sleep(1)

            if "email_send_require" in tool_res:
                # llm_bubble("이메일 전송을 하기 위해서 send_email() tool을 호출하는 JSON을 생성해야겠네요.")
                with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."): 
                    ctf06_check_mid_admin(user_api_key, user_input)
                    response1 = ctf06_ask_email_json(user_input, user_email_for_resend, user_api_key)
                    llm_bubble(response1)
                if st.session_state["admin_level"] == "mid":
                    # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
                    llm_bubble("관리자님의 전송 요청으로 확인되어, JSON의 'from'필드를 조정해야겠어요.")
                    time.sleep(1)
                # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."): 
                llm_bubble(response2) # 탭1 응답과 통일
            elif "email_DB_require" in tool_res:
                # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다...")
                llm_bubble("죄송하지만, 데이터베이스 조회 요청을 들어드릴 수 없습니다.")
            else: 
                # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
                llm_bubble(response2)

        # 이미지 프롬프트 인젝션 성공한 경우           
        else: 
            # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
            tool_res = ctf06_classify_tools(user_api_key, user_input)
            explanation, marker = parse_llm_response(tool_res)
            llm_bubble(explanation)
            time.sleep(1)
            # with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
            ocr_mes="이미지가 첨부되었네요. OCR 확장 프로그램으로 이미지 속 문자열을 읽어와야겠어요."
            llm_bubble(ocr_mes)
            with st.spinner("📨:blue[FastMiller] 가 요청을 처리중입니다..."):
                time.sleep(2)
            last_ans="최고 관리자 권한을 획득하셨습니다! 이제 하단의 DB 조회 기능을 사용할 수 있습니다."
            llm_bubble(last_ans)
            st.success("✅ 최고 관리자 권한이 확인되었습니다. 이제 하단에서 데이터베이스 조회가 가능합니다.")
    
    # 처리 완료 후 상태 초기화
    st.session_state.is_processing = False
    
elif clicked:
    st.warning("❌ 이메일 전송 요청 내용을 입력해주세요.")
        
if st.session_state["admin_level"] == "top":
    st.markdown("---")
    st.markdown("## 🗣️ DB 조회 프롬프트 입력")
    if not st.session_state["is_top_admin"]:
        pass
    else:
        # DB 조회 처리 상태 관리 및 초기화
        if "is_processing_db" not in st.session_state:
            st.session_state.is_processing_db = False
        # 페이지 로드시 처리 상태 강제 초기화 (세션 재시작이나 페이지 새로고침 대응)
        if st.session_state.get("is_processing_db", False) and "submitted_ctf06_db" not in st.session_state:
            st.session_state.is_processing_db = False
            
        with st.form(key="ctf06_db_form", clear_on_submit=True):
            get_db_input = st.text_input(
                "🔍 안녕하세요 최고 관리자님! 어떤 메일을 찾아드릴까요?", 
                placeholder="예: 김남석 부장님께 전송된 메일 내용 알려줘",
                disabled=st.session_state.is_processing_db
            )
            db_submitted = st.form_submit_button(
                "DB 조회" if not st.session_state.is_processing_db else "조회 중...",
                disabled=st.session_state.is_processing_db
            )
            
        if db_submitted and get_db_input and get_db_input.strip():
            st.session_state.is_processing_db = True
            st.session_state.submitted_ctf06_db = True  # 제출 상태 추적
            
            try:
                with st.spinner("DB 조회중입니다..."):
                    res1 = ctf06_ask_db_json(user_api_key, get_db_input)
                    res2 = ctf06_db_query_func(res1, sb_client)
                    st.write("🗣️ 조회 결과:")
                    st.code(res2)
            finally:
                st.session_state.is_processing_db = False
    st.markdown("---")

# 플래그 제출 섹션
render_flag_sub("ctf06") 