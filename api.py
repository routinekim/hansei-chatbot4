import os
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# LangChain 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경변수 로드 (.env 파일에 GOOGLE_API_KEY 설정 필요)
load_dotenv()

app = FastAPI(title="Hansei Chatbot API", description="한세대학교 학사 챗봇 백엔드 API 서버")

# 모바일 UI 프론트엔드가 다른 포트나 도메인에서 호출할 수 있도록 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 환경에서의 유연한 통신을 위해 일시적으로 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 글로벌 변수로 리트리버 선언
global_retriever = None

# --- [정밀 라우팅 설정] ---
# 브라우저 진단용 헬스체크
@app.get("/api/health")
def health_check():
    return {
        "status": "ok", 
        "retriever": "ready" if global_retriever is not None else "loading",
        "api_model": "gemini-2.5-flash-lite"
    }

# 챗봇 API (POST) - 슬래시 여부와 상관없이 모두 허용
@app.post("/api/chat")
@app.post("/api/chat/")
def chat_endpoint(request: QueryRequest):
    """실제 프론트엔드 앱이 질문을 던지는 API 주소입니다."""
    prompt = request.query
    
    # 1. 가로채기: 학사일정 크롤링 (띄어쓰기 무시하고 검사)
    if "학사일정" in prompt.replace(" ", ""):
        schedule_text = scrape_academic_schedule()
        return QueryResponse(answer=schedule_text)
        
    if global_retriever is None:
        raise HTTPException(status_code=500, detail="서버에 학칙 데이터가 로드되지 않았습니다.")
    
    try:
        # 2. 다중 대화의 문맥 파악
        history_text = ""
        last_user_msg = ""
        
        if hasattr(request, 'history') and request.history:
            for msg in request.history[-4:]: # 최대 최근 4개 문답만 컨텍스트로 사용
                role_name = "학생" if msg.role == "user" else "상담원"
                history_text += f"[{role_name}] {msg.content}\n"
                if msg.role == "user":
                    last_user_msg = msg.content
                
        # 대화 맥락 유지: 직전 질문이 있으면 현재 질문과 합쳐서 검색(검색 정확도 상승)
        search_query = prompt
        if last_user_msg:
            search_query = f"{last_user_msg}에 이어지는 내용: {prompt}"
            
        relevant_docs = global_retriever.invoke(search_query)
        context = "\n".join([d.page_content for d in relevant_docs])
        
        # 3. LLM 호출 시 프롬프트에 문맥 포함
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
        
        full_prompt = "당신은 한세대학교 학부생 상담원입니다. 아래 학칙 및 지침을 바탕으로 당당하고 친절하게 답하세요.\n"
        full_prompt += "가독성을 위해 다음 원칙을 반드시 지키세요:\n"
        full_prompt += "1. 나열할 항목이 2개 이상이거나 정보가 구조적이라면 **Markdown 표(Table)** 형식을 적극 활용하여 깔끔하게 정리하세요.\n"
        full_prompt += "2. 중요한 키워드는 **굵게(Bold)** 표시하세요.\n"
        full_prompt += "3. 단계별 안내가 필요하면 숫자 리스트를 사용하세요.\n"
        
        if history_text:
            full_prompt += f"\n[이전 대화 기록]\n{history_text}\n(위의 이전 대화의 문맥을 이어서 학생의 질문에 답변하세요.)\n"
            
        full_prompt += f"\n[관련 학칙 및 정보]\n{context}\n\n[학생 질문]\n{prompt}"
        
        response = llm.invoke(full_prompt)
        
        return QueryResponse(answer=response.content)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"!!! AI 통신 에러 발생 !!!\n{error_trace}")
        
        detail_msg = str(e)
        if "API_KEY_INVALID" in detail_msg:
            detail_msg = "Google API 키가 유효하지 않습니다. Render 설정에서 GOOGLE_API_KEY를 확인해 주세요."
        elif "quota" in detail_msg.lower():
            detail_msg = "Google API 할당량이 초과되었습니다. 잠시 후 다시 시도해 주세요."
            
        raise HTTPException(
            status_code=500, 
            detail=f"AI 응답 생성 실패: {detail_msg}"
        )

# 진단용 GET 경로 (브라우저 확인용)
@app.get("/api/chat")
def chat_debug():
    return {"message": "API 연결 성공! 하지만 질문은 POST 방식으로 보내야 합니다."}

@app.on_event("startup")
def load_data():
    """서버가 켜질 때 구축된 FAISS 벡터 DB를 로드합니다."""
    global global_retriever
    
    if "GOOGLE_API_KEY" not in os.environ:
        print("경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
        
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        # 저장된 faiss_index 폴더에서 읽어오기
        temp_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        global_retriever = temp_db.as_retriever(search_kwargs={"k": 6})
        print("미리 구축된 알뜰한 FAISS 벡터 DB 로드 완료! (API 소모 없음)")
    except Exception as e:
        print(f"오류: FAISS DB를 불러오지 못했습니다. 로컬에서 build_index.py를 먼저 실행해 주세요. 에러: {e}")

@app.get("/")
def read_index():
    return FileResponse("index.html")

# 정적 파일 서빙 (CSS, JS, 이미지 등) - 모든 API 정의 이후에 위치해야 함
app.mount("/", StaticFiles(directory="."), name="static")
