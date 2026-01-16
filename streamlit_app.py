import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import plotly.graph_objects as go
from PIL import Image
import io
import json
import os
import tempfile

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
    .sub-header {
        font-size: 1.3rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .evaluation-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stChat message {
        border-radius: 15px;
    }
</style>
""", unsafe_allow_html=True)

# API 키 설정
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


@st.cache_resource
def load_pdf_and_create_vectorstore():
    """PDF 로드 및 벡터스토어 생성"""
    try:
        # PDF 파일 로드
        loader = PyPDFLoader("test.pdf")
        documents = loader.load()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # 임베딩 및 벡터스토어 생성
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"PDF 로드 오류: {e}")
        return None


def create_qa_chain(vectorstore):
    """QA 체인 생성"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.3
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return qa_chain


def extract_text_from_image(image):
    """Gemini Vision을 사용하여 이미지에서 텍스트 추출"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = """이 이미지는 초등학생이 쓴 일기입니다. 
    이미지에서 모든 텍스트를 정확하게 추출해주세요.
    손글씨를 주의 깊게 읽고, 원문 그대로 추출해주세요.
    추출된 텍스트만 반환하세요."""
    
    response = model.generate_content([prompt, image])
    return response.text


def evaluate_diary(text, criteria):
    """일기 평가 수행"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""당신은 초등학생 일기를 평가하는 전문 교사입니다.
    
다음 일기를 아래 평가 기준에 따라 평가해주세요.

[일기 내용]
{text}

[평가 기준]
{criteria}

다음 JSON 형식으로 정확하게 응답해주세요:
{{
    "overall_score": 1-5 사이의 숫자 (종합 점수),
    "categories": [
        {{"name": "맞춤법/문법", "score": 1-5, "feedback": "피드백"}},
        {{"name": "내용 충실도", "score": 1-5, "feedback": "피드백"}},
        {{"name": "표현력", "score": 1-5, "feedback": "피드백"}},
        {{"name": "구성/흐름", "score": 1-5, "feedback": "피드백"}},
        {{"name": "창의성", "score": 1-5, "feedback": "피드백"}}
    ],
    "overall_feedback": "전체적인 피드백과 격려의 말",
    "improvement_tips": ["개선 제안 1", "개선 제안 2", "개선 제안 3"]
}}

JSON만 반환하고 다른 텍스트는 포함하지 마세요."""

    response = model.generate_content(prompt)
    
    # JSON 파싱
    try:
        result_text = response.text.strip()
        # JSON 블록 추출
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        return json.loads(result_text)
    except json.JSONDecodeError:
        return None


def create_radar_chart(evaluation_result):
    """평가 결과를 레이더 차트로 시각화"""
    categories = evaluation_result["categories"]
    
    names = [cat["name"] for cat in categories]
    scores = [cat["score"] for cat in categories]
    
    # 레이더 차트 닫기 위해 첫 번째 값 추가
    names.append(names[0])
    scores.append(scores[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=names,
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.3)',
        line=dict(color='#1E88E5', width=2),
        name='평가 점수'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['1점', '2점', '3점', '4점', '5점']
            )
        ),
        showlegend=False,
        title=dict(
            text="📊 일기 평가 결과",
            font=dict(size=20)
        ),
        height=400
    )
    
    return fig


def create_pie_chart(evaluation_result):
    """평가 결과를 원형 그래프로 시각화"""
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
        texttemplate='%{label}<br>%{value}점',
        hovertemplate='%{label}: %{value}점<extra></extra>'
    )])
    
    fig.update_layout(
        title=dict(
            text="🥧 항목별 점수 분포",
            font=dict(size=20)
        ),
        height=400,
        annotations=[dict(
            text=f'종합<br>{evaluation_result["overall_score"]}점',
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )]
    )
    
    return fig


def create_bar_chart(evaluation_result):
    """평가 결과를 막대 그래프로 시각화"""
    categories = evaluation_result["categories"]
    
    names = [cat["name"] for cat in categories]
    scores = [cat["score"] for cat in categories]
    
    colors = ['#FF6B6B' if s < 3 else '#FFEAA7' if s < 4 else '#96CEB4' for s in scores]
    
    fig = go.Figure(data=[go.Bar(
        x=names,
        y=scores,
        marker_color=colors,
        text=scores,
        textposition='outside'
    )])
    
    fig.update_layout(
        title=dict(
            text="📈 항목별 상세 점수",
            font=dict(size=20)
        ),
        yaxis=dict(range=[0, 5.5], title="점수"),
        xaxis=dict(title="평가 항목"),
        height=400
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
        # 이미지 입력 방식 선택
        input_method = st.radio(
            "이미지 입력 방식 선택",
            ["📤 파일 업로드", "📸 카메라 촬영"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "📤 파일 업로드":
            uploaded_file = st.file_uploader(
                "일기 이미지를 업로드하세요",
                type=["png", "jpg", "jpeg"],
                help="초등학생이 작성한 일기 사진을 업로드해주세요."
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                
        else:  # 카메라 촬영
            camera_image = st.camera_input("일기를 촬영하세요")
            if camera_image:
                image = Image.open(camera_image)
        
        # 이미지 미리보기
        if image:
            st.image(image, caption="업로드된 일기", use_container_width=True)
            
            # 텍스트 추출 버튼
            if st.button("🔍 텍스트 추출", type="primary", use_container_width=True):
                with st.spinner("텍스트를 추출하는 중..."):
                    extracted = extract_text_from_image(image)
                    st.session_state.extracted_text = extracted
                    st.success("텍스트 추출 완료!")
    
    with col2:
        # 추출된 텍스트 표시 및 수정
        st.markdown("### 📄 추출된 텍스트")
        extracted_text = st.text_area(
            "추출된 일기 내용 (수정 가능)",
            value=st.session_state.extracted_text,
            height=200,
            placeholder="이미지에서 추출된 텍스트가 여기에 표시됩니다..."
        )
        
        # 평가 기준 입력
        st.markdown("### 📋 평가 기준 설정")
        default_criteria = """1. 맞춤법과 문법이 정확한가?
2. 하루의 일과가 구체적으로 작성되었는가?
3. 자신의 감정과 생각이 잘 표현되었는가?
4. 글의 시작, 중간, 끝이 자연스럽게 연결되는가?
5. 독창적인 표현이나 비유가 사용되었는가?"""
        
        criteria = st.text_area(
            "평가 기준을 입력하세요",
            value=default_criteria,
            height=150,
            help="평가할 기준을 자유롭게 수정할 수 있습니다."
        )
        
        # 평가 실행 버튼
        if st.button("✨ 일기 평가하기", type="primary", use_container_width=True):
            if extracted_text.strip():
                with st.spinner("일기를 평가하는 중..."):
                    result = evaluate_diary(extracted_text, criteria)
                    if result:
                        st.session_state.evaluation_result = result
                        st.success("평가 완료!")
                    else:
                        st.error("평가 중 오류가 발생했습니다.")
            else:
                st.warning("먼저 텍스트를 추출하거나 입력해주세요.")
    
    # 평가 결과 표시
    if st.session_state.evaluation_result:
        st.markdown("---")
        st.markdown("## 📊 평가 결과")
        
        result = st.session_state.evaluation_result
        
        # 종합 점수 표시
        score_col1, score_col2, score_col3 = st.columns([1, 2, 1])
        with score_col2:
            overall = result["overall_score"]
            stars = "⭐" * overall
            st.markdown(f"""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h2>종합 점수</h2>
                <h1 style="font-size: 4rem;">{overall}/5</h1>
                <p style="font-size: 2rem;">{stars}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # 차트 표시
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            pie_chart = create_pie_chart(result)
            st.plotly_chart(pie_chart, use_container_width=True)
        
        with chart_col2:
            radar_chart = create_radar_chart(result)
            st.plotly_chart(radar_chart, use_container_width=True)
        
        # 막대 그래프
        bar_chart = create_bar_chart(result)
        st.plotly_chart(bar_chart, use_container_width=True)
        
        # 상세 피드백
        st.markdown("### 💬 상세 피드백")
        
        for cat in result["categories"]:
            score = cat["score"]
            color = "#96CEB4" if score >= 4 else "#FFEAA7" if score >= 3 else "#FF6B6B"
            
            with st.expander(f"{cat['name']} - {score}점 {'⭐' * score}", expanded=True):
                st.markdown(f"""
                <div style="padding: 1rem; background-color: {color}20; border-left: 4px solid {color}; border-radius: 5px;">
                    {cat['feedback']}
                </div>
                """, unsafe_allow_html=True)
        
        # 전체 피드백
        st.markdown("### 🌟 선생님의 한마디")
        st.info(result["overall_feedback"])
        
        # 개선 제안
        st.markdown("### 💡 더 좋은 일기를 위한 팁")
        for i, tip in enumerate(result.get("improvement_tips", []), 1):
            st.markdown(f"**{i}.** {tip}")


# ============ 탭 2: PDF 기반 챗봇 ============
with tab2:
    st.markdown("### 📖 PDF 문서 기반 Q&A 챗봇")
    
    # 벡터스토어 초기화
    if st.session_state.vectorstore is None:
        with st.spinner("PDF 문서를 로드하는 중..."):
            st.session_state.vectorstore = load_pdf_and_create_vectorstore()
            if st.session_state.vectorstore:
                st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
                st.success("PDF 문서 로드 완료!")
            else:
                st.warning("test.pdf 파일을 찾을 수 없습니다. 파일을 확인해주세요.")
    
    # 채팅 인터페이스
    chat_container = st.container()
    
    with chat_container:
        # 이전 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("PDF 내용에 대해 질문하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            if st.session_state.qa_chain:
                with st.spinner("답변을 생성하는 중..."):
                    try:
                        response = st.session_state.qa_chain({
                            "question": prompt,
                            "chat_history": st.session_state.chat_history
                        })
                        
                        answer = response["answer"]
                        st.markdown(answer)
                        
                        # 출처 표시
                        if response.get("source_documents"):
                            with st.expander("📚 참고 문서"):
                                for i, doc in enumerate(response["source_documents"], 1):
                                    st.markdown(f"**출처 {i}:** {doc.page_content[:200]}...")
                        
                        # 히스토리 업데이트
                        st.session_state.chat_history.append((prompt, answer))
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                msg = "PDF 파일이 로드되지 않았습니다. test.pdf 파일을 확인해주세요."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
    
    # 채팅 초기화 버튼
    if st.button("🔄 대화 초기화"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 사이드바
with st.sidebar:
    st.markdown("## ℹ️ 사용 안내")
    
    st.markdown("""
    ### 📝 일기 평가 탭
    1. **이미지 업로드** 또는 **카메라 촬영**으로 일기 이미지 입력
    2. **텍스트 추출** 버튼으로 글씨 인식
    3. 필요시 추출된 텍스트 수정
    4. **평가 기준** 설정 (기본값 제공)
    5. **일기 평가하기** 버튼으로 평가 실행
    
    ### 💬 챗봇 탭
    - test.pdf 문서 내용을 기반으로 질문에 답변
    - RAG 기술을 활용한 정확한 답변 제공
    
    ---
    
    ### ⭐ 평가 점수 기준
    - **5점**: 매우 우수
    - **4점**: 우수
    - **3점**: 보통
    - **2점**: 노력 필요
    - **1점**: 많은 노력 필요
    """)
    
    st.markdown("---")
    st.markdown("🤖 Powered by **Gemini 2.5 Flash**")
