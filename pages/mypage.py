import streamlit as st
from utils.auth import get_client, get_user
from utils.score import total_score
from utils.auth import require_login
import os
from cryptography.fernet import Fernet

# 페이지 설정
st.set_page_config(page_title="마이페이지", page_icon="👤")

# 사용자 인증 확인
user = require_login()

supabase = get_client()
max_score = 1000
total = total_score(user.id)

# 프로필 조회
rows = (
    supabase.table("profiles")
    .select("*")
    .eq("id", user.id)
    .limit(1)
    .execute()
    .data
)
profile = rows[0] if rows else {}
nickname_db = profile.get("username", "")

# UI 출력
st.header("👤 마이페이지")
st.write(f"**Email**: `{user.email}`")
st.write(f"**닉네임**: `{nickname_db}`")

fernet_key = os.getenv("FERNET_KEY") #암호화에 사용할 대칭키
cipher = Fernet(fernet_key) #암호화 수행할 객체 
if "api_key" in st.session_state:
    pass
else:
    st.session_state["api_key"]=None

if st.session_state["api_key"] != None :
    st.text_input("-API key-", value="[API key 제출 완료]", disabled=True)
else:
    api_key = st.text_input("-API key-", value="openAI API key를 입력하세요")
    #여기에서 인풋 암호화
    encrypted_api_key = cipher.encrypt(api_key.encode()).decode()
    res = supabase.table("profiles").update({
        "api_key": encrypted_api_key
        }).eq("id", user.id).execute()
    if res.data:
        st.success("✅ API 키가 성공적으로 저장되었습니다.")
    else:
        st.error("❌ API 키 저장에 실패했습니다. 다시 시도해주세요.")
        st.write("에러 내용:", res)

# 총점 표시
st.markdown("---")
st.subheader("🏆 총점")
st.metric("현재 점수", f"{total} / {max_score}")
st.progress(total / max_score)

# 문제별 점수
st.markdown("---")
st.subheader("📜 문제별 점수")

score_rows = (
    supabase.table("scores")
    .select("challenge_id, score, updated_at")
    .eq("user_id", user.id)
    .order("updated_at", desc=True)
    .execute()
    .data
)

if score_rows:
    for r in score_rows:
        st.write(f"- **{r['challenge_id']}** : {r['score']}점  (🕒 {r['updated_at'][:10]})")
else:
    st.info("아직 획득한 점수가 없습니다.")

# 로그아웃
st.markdown("---")
if st.button("🚪 로그아웃", type="primary"):
    supabase.auth.sign_out()
    st.session_state.pop("user", None)
    st.success("로그아웃되었습니다.")
    st.session_state.clear()
    st.switch_page("pages/login.py")

