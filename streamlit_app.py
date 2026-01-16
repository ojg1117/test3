import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import plotly.graph_objects as go
from PIL import Image
import json
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="📚 초등학생 일기 평가 챗봇",
    page_icon="📝",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# API 키 설정
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


@st.cache_data
def load_pdf():
    """PDF 텍스트 추출"""
    try:
        reader = PdfReader("test.pdf")
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return None


def get_chat_response(user_question, pdf_context):
    """PDF 기반 챗봇 응답 생성"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    # 대화 히스토리 구성
    history = ""
    for msg in st.session_state.messages[-6:]:  # 최근 6개 메시지만
        role = "사용자" if msg["role"] == "user" else "AI"
        history += f"{role}: {msg['content']}\n"

    prompt = f"""당신은 PDF 문서 내용을 기반으로 답변하는 친절한 AI 어시스턴트입니다.

[PDF 문서 내용]
{pdf_context[:8000]}

[이전 대화]
{history}

[현재 질문]
{user_question}

위 PDF 문서 내용을 참고하여 질문에 답변해주세요. 
문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.
한국어로 친절하게 답변해주세요."""

    response = model.generate_content(prompt)
    return response.text


def extract_text_from_image(image):
    """이미지에서 텍스트 추출"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = """이 이미지는 초등학생이 쓴 일기입니다. 
    이미지에서 모든 텍스트를 정확하게 추출해주세요.
    손글씨를 주의 깊게 읽고, 원문 그대로 추출해주세요.
    추출된 텍스트만 반환하세요."""

    response = model.generate_content([prompt, image])
    return response.text


def evaluate_diary(text, criteria):
    """일기 평가"""
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""당신은 초등학생 일기를 평가하는 전문 교사입니다.

[일기 내용]
{text}

[평가 기준]
{criteria}

다음 JSON 형식으로만 응답하세요:
{{
    "overall_score": 3,
    "categories": [
        {{"name": "맞춤법/문법", "score": 3, "feedback": "피드백 내용"}},
        {{"name": "내용 충실도", "score": 3, "feedback": "피드백 내용"}},
        {{"name": "표현력", "score": 3, "feedback": "피드백 내용"}},
        {{"name": "구성/흐름", "score": 3, "feedback": "피드백 내용"}},
        {{"name": "창의성", "score": 3, "feedback": "피드백 내용"}}
    ],
    "overall_feedback": "전체 피드백",
    "improvement_tips": ["팁1", "팁2", "팁3"]
}}

점수는 1-5 사이 정수입니다. JSON만 반환하세요."""

    response = model.generate_content(prompt)

    try:
        result_text = response.text.strip()
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        return json.loads(result_text.strip())
    except:
        return None


def create_pie_chart(evaluation_result):
    """원형 그래프"""
    categories = evaluation_result["categories"]
    names = [cat["name"] for cat in categories]
    scores = [cat["score"] for cat in categories]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    fig = go.Figure(data=[go.Pie(
        labels=names,
        values=scores,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+value',
        texttemplate='%{label}<br>%{value}점'
    )])

    fig.update_layout(
        title="🥧 항목별 점수 분포",
        height=400,
        annotations=[dict(
            text=f'종합<br>{evaluation_result["overall_score"]}점',
            x=0.5, y=0.5, font_size=18, showarrow=False
        )]
    )
    return fig


def create_radar_chart(evaluation_result):
    """레이더 차트"""
    categories = evaluation_result["categories"]
    names = [cat["name"] for cat in categories] + [categories[0]["name"]]
    scores = [cat["score"] for cat in categories] + [categories[0]["score"]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores, theta=names, fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(color='#1E88E5', width=2)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title="📊 평가 결과", height=400, showlegend=False
    )
    return fig


def create_bar_chart(evaluation_result):
    """막대 그래프"""
    categories = evaluation_result["categories"]
    names = [cat["name"] for cat in categories]
    scores = [cat["score"] for cat in categories]
    colors = ['#FF6B6B' if s < 3 else '#FFEAA7' if s < 4 else '#96CEB4' for s in scores]

    fig = go.Figure(data=[go.Bar(
        x=names, y=scores, marker_color=colors,
        text=scores, textposition='outside'
    )])

    fig.update_layout(
        title="📈 항목별 상세 점수",
        yaxis=dict(range=[0, 5.5]), height=400
    )
    return fig


# 메인 헤더
st.markdown('<h1 class="main-header">📚 초등학생 일기 평가 챗봇</h1>', unsafe_allow_html=True)

# 탭 구성
tab1, tab2 = st.tabs(["📝 일기 평가", "💬 PDF 기반 챗봇"])

# ============ 탭 1: 일기 평가 ============
with tab1:
    st.markdown("### 📷 일기 이미지 업로드")

    col1, col2 = st.columns([1, 1])

    with col1:
        input_method = st.radio(
            "이미지 입력 방식",
            ["📤 파일 업로드", "📸 카메라 촬영"],
            horizontal=True
        )

        image = None

        if input_method == "📤 파일 업로드":
            uploaded_file = st.file_uploader(
                "일기 이미지를 업로드하세요",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
        else:
            camera_image = st.camera_input("일기를 촬영하세요")
            if camera_image:
                image = Image.open(camera_image)

        if image:
            st.image(image, caption="업로드된 일기", use_container_width=True)

            if st.button("🔍 텍스트 추출", type="primary", use_container_width=True):
                with st.spinner("텍스트 추출 중..."):
                    st.session_state.extracted_text = extract_text_from_image(image)
                    st.success("완료!")

    with col2:
        st.markdown("### 📄 추출된 텍스트")
        extracted_text = st.text_area(
            "일기 내용 (수정 가능)",
            value=st.session_state.extracted_text,
            height=200,
            placeholder="텍스트가 여기에 표시됩니다..."
        )

        st.markdown("### 📋 평가 기준")
        default_criteria = """1. 맞춤법과 문법이 정확한가?
2. 하루의 일과가 구체적으로 작성되었는가?
3. 자신의 감정과 생각이 잘 표현되었는가?
4. 글의 흐름이 자연스러운가?
5. 독창적인 표현이 사용되었는가?"""

        criteria = st.text_area(
            "평가 기준 입력",
            value=default_criteria,
            height=150
        )

        if st.button("✨ 일기 평가하기", type="primary", use_container_width=True):
            if extracted_text.strip():
                with st.spinner("평가 중..."):
                    result = evaluate_diary(extracted_text, criteria)
                    if result:
                        st.session_state.evaluation_result = result
                        st.success("평가 완료!")
                    else:
                        st.error("평가 중 오류 발생")
            else:
                st.warning("텍스트를 입력해주세요.")

    # 평가 결과
    if st.session_state.evaluation_result:
        st.markdown("---")
        st.markdown("## 📊 평가 결과")

        result = st.session_state.evaluation_result

        # 종합 점수
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            overall = result["overall_score"]
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 15px; color: white;">
                <h2>종합 점수</h2>
                <h1 style="font-size: 4rem;">{overall}/5</h1>
                <p style="font-size: 2rem;">{"⭐" * overall}</p>
            </div>
            """, unsafe_allow_html=True)

        # 차트
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_pie_chart(result), use_container_width=True)
        with c2:
            st.plotly_chart(create_radar_chart(result), use_container_width=True)

        st.plotly_chart(create_bar_chart(result), use_container_width=True)

        # 피드백
        st.markdown("### 💬 상세 피드백")
        for cat in result["categories"]:
            score = cat["score"]
            color = "#96CEB4" if score >= 4 else "#FFEAA7" if score >= 3 else "#FF6B6B"
            with st.expander(f"{cat['name']} - {score}점 {'⭐' * score}"):
                st.markdown(f'<div style="padding:1rem; background:{color}20; border-left:4px solid {color}; border-radius:5px;">{cat["feedback"]}</div>', unsafe_allow_html=True)

        st.markdown("### 🌟 선생님의 한마디")
        st.info(result["overall_feedback"])

        st.markdown("### 💡 개선 팁")
        for i, tip in enumerate(result.get("improvement_tips", []), 1):
            st.markdown(f"**{i}.** {tip}")


# ============ 탭 2: PDF 챗봇 ============
with tab2:
    st.markdown("### 📖 PDF 문서 기반 Q&A")

    # PDF 로드
    if not st.session_state.pdf_text:
        pdf_text = load_pdf()
        if pdf_text:
            st.session_state.pdf_text = pdf_text
            st.success("✅ test.pdf 로드 완료!")
        else:
            st.warning("⚠️ test.pdf 파일을 찾을 수 없습니다.")

    # 채팅 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if st.session_state.pdf_text:
                with st.spinner("답변 생성 중..."):
                    answer = get_chat_response(prompt, st.session_state.pdf_text)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.warning("PDF 파일이 로드되지 않았습니다.")

    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 사이드바
with st.sidebar:
    st.markdown("## ℹ️ 사용 안내")
    st.markdown("""
    ### 📝 일기 평가
    1. 이미지 업로드/촬영
    2. 텍스트 추출
    3. 평가 기준 설정
    4. 평가 실행

    ### 💬 챗봇
    - test.pdf 기반 Q&A

    ---
    ⭐ **점수 기준**
    - 5점: 매우 우수
    - 4점: 우수  
    - 3점: 보통
    - 2점: 노력 필요
    - 1점: 많은 노력 필요
    """)
    st.markdown("---")
    st.markdown("🤖 **Gemini 2.5 Flash**")
