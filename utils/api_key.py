import os
from dotenv import load_dotenv
import streamlit as st
from utils.auth import get_client, require_login
from cryptography.fernet import Fernet
from openai import RateLimitError, AuthenticationError, BadRequestError

load_dotenv()

fernet_key = os.getenv("FERNET_KEY") # 암호화에 사용할 대칭키

cipher = Fernet(fernet_key.encode()) 

# 10분 정도 캐시 유지, uid마다 분리됨
@st.cache_data(ttl=300)  
def get_decrypted_api_key(uid: str) -> str | None:
    sb = get_client()
    try:
        res = sb.table("profiles").select("api_key").eq("id", uid).single().execute()
        encrypted_api_key = res.data.get("api_key")
        if not encrypted_api_key:
            return None
        return cipher.decrypt(encrypted_api_key.encode()).decode()
    except:
        return None
    
def require_api_key():
    # 먼저 사용자 로그인 확인
    user = require_login() 
    user_id = getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else None)
    
    if "api_key" not in st.session_state:
        decrypted_api_key = get_decrypted_api_key(user_id)

        if not decrypted_api_key:
            st.error("API 키를 제출한 뒤 이용해주세요.")
            st.page_link("pages/mypage.py", label="👉 API키 제출하러 가기")
            st.stop()

        st.session_state["api_key"] = decrypted_api_key

    return st.session_state["api_key"]


def handle_api_error(error):
    """API 오류 처리 및 마이페이지로 리다이렉트"""
    if isinstance(error, AuthenticationError):
        st.error("❌ 올바른 API 키로 수정해주세요.")
    elif isinstance(error, RateLimitError):
        st.error("❌ API 사용 한도가 초과되었습니다. 잠시 후 다시 시도해주세요.")
    elif isinstance(error, BadRequestError):
        st.error("❌ 잘못된 요청입니다. API 키를 확인해주세요.")
    else:
        st.error("❌ 올바른 API 키로 수정해주세요.")
    
    # 마이페이지로 리다이렉트
    st.page_link("pages/mypage.py", label="🔧 마이페이지에서 API 키 수정하기")
    
    st.stop()