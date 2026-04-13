import os
from fastapi import FastAPI, HTTPException
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

from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

# 프론트엔드에서 보낼 질문 데이터 구조
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

class QueryResponse(BaseModel):
    answer: str

def scrape_academic_schedule():
    """사용자 요청에 따라 실시간 크롤링 대신 고정된 1~12월 전체 학사일정을 바로 반환합니다."""
    return """📅 **[2026학년도 학사일정]**

| 월 | 일정 및 상세 내용 |
| :--- | :--- |
| **01월** | • [01-06] 2학기 성적 확정<br>• [01-14 ~ 01-16] 동계 계절학기 성적입력<br>• [01-17 ~ 01-19] 동계 계절학기 성적정정 및 확정<br>• [01-26 ~ 02-06] 1학기 복학 신청(2차) |
| **02월** | • [02-10 ~ 02-12] 1학기 예비수강신청<br>• [02-12] 2025학년도 전기 학위수여식<br>• [02-19 ~ 02-27] 1학기 등록금 수납기간<br>• [02-23 ~ 02-25] 1학기 재·복학생 수강신청<br>• [02-24 ~ 02-25] 신입생 오리엔테이션<br>• [02-26 ~ 02-27] 1학기 신·편입생 수강신청 |
| **03월** | • [03-03] 1학기 개강 및 입학식<br>• [03-03 ~ 03-09] 수강신청 정정<br>• [03-11 ~ 03-17] 연한초과자 등록금 수납<br>• [03-17 ~ 03-20] 수강신청 철회신청 |
| **04월** | • [04-21 ~ 04-27] 중간고사<br>• [04-27 ~ 05-01] 다전공 신청<br>• [04-28] 전공진로박람회<br>• [04-29] 한세 JOB FAIR |
| **05월** | • [05-11 ~ 05-15] 재입학, 전부(과) 신청<br>• [05-12 ~ 05-13] 오순절 축제<br>• [05-22 ~ 05-29] 하계 계절학기 수강신청 |
| **06월** | • [06-02] 종강예배<br>• [06-09 ~ 06-15] 보강주<br>• [06-16 ~ 06-22] 기말고사<br>• [06-22] 1학기 종강<br>• [06-23 ~ 07-13] 하계 계절학기 수업 |
| **07월** | • [07-03] 1학기 성적 확정<br>• [07-20 ~ 07-31] 2학기 복학 신청(2차) |
| **08월** | • [08-18] 2025학년도 후기 졸업<br>• [08-18 ~ 08-31] 2학기 휴학 신청<br>• [08-19 ~ 08-21] 2학기 수강신청<br>• [08-24 ~ 08-31] 2학기 등록금 수납기간 |
| **09월** | • [09-01] 2학기 개강 및 개강예배<br>• [09-01 ~ 09-07] 수강신청 정정<br>• [09-15 ~ 09-18] 수강신청 철회신청 |
| **10월** | • [10-06] 한세체육대회<br>• [10-20 ~ 10-26] 중간고사<br>• [10-26 ~ 10-30] 다전공 신청 |
| **11월** | • [11-09 ~ 11-13] 재입학, 전부(과) 신청<br>• [11-23 ~ 11-27] 동계 계절학기 신청 |
| **12월** | • [12-01] 2학기 종강예배<br>• [12-15 ~ 12-21] 기말고사<br>• [12-21] 2학기 종강<br>• [12-22 ~ 01-13] 동계 계절학기 수업 |"""

@app.post("/chat", response_model=QueryResponse)
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
        print(f"AI 통신 에러: {e}")
        # 진짜 에러 원인을 추적하기 위해 e의 내용을 함께 보냅니다.
        raise HTTPException(status_code=500, detail=f"AI가 응답을 생성하는 중 오류가 발생했습니다. (상세 에러: {str(e)})")

# 정적 파일 서빙 (CSS, JS, 이미지 등) - 모든 API 정의 이후에 위치해야 함
app.mount("/", StaticFiles(directory="."), name="static")
