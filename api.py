import os
import asyncio
import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import urllib3

# LangChain 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 환경 설정 및 로깅
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(
    title="Hansei Chatbot API",
    description="한세대학교 학사 챗봇 백엔드 API 서버",
    redirect_slashes=True
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 데이터 모델 및 글로벌 변수
global_retriever = None

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

class QueryResponse(BaseModel):
    answer: str

# 3. 유틸리티 함수
def scrape_academic_schedule():
    """고정된 학사일정 반환"""
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

# 4. API 라우터 설정
api_router = APIRouter(prefix="/api")

@api_router.get("/health")
@api_router.head("/health")
async def health_check():
    """상태 확인용 엔드포인트"""
    return {
        "status": "ok",
        "retriever": "ready" if global_retriever is not None else "loading"
    }

@api_router.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """실시간 스트리밍 답변 엔드포인트"""
    start_time = time.time()
    logger.info(f"[API 호출 수신] /api/chat - 질문: {request.query}")
    
    prompt = request.query
    
    # 학사일정 즉시 처리
    if "학사일정" in prompt.replace(" ", ""):
        schedule_text = scrape_academic_schedule()
        async def schedule_generator():
            yield schedule_text
        return StreamingResponse(schedule_generator(), media_type="text/plain")

    if global_retriever is None:
        logger.error("글로벌 리트리버가 로드되지 않은 상태에서 요청이 들어왔습니다.")
        raise HTTPException(status_code=500, detail="서버 데이터 로딩 중입니다. 잠시 후에 다시 시도해 주세요.")

    async def response_generator():
        try:
            # 대화 문맥 구성
            history_text = ""
            last_user_msg = ""
            if request.history:
                for msg in request.history[-4:]:
                    history_text += f"[{'학생' if msg.role == 'user' else '상담원'}] {msg.content}\n"
                    if msg.role == "user":
                        last_user_msg = msg.content
            
            search_query = f"{last_user_msg} {prompt}" if last_user_msg else prompt
            
            # 검색 및 컨텍스트 생성
            search_start = time.time()
            relevant_docs = await asyncio.to_thread(global_retriever.invoke, search_query)
            logger.info(f"🔍 [검색 완료] 소요 시간: {time.time() - search_start:.2f}초")
            
            context = "\n".join([d.page_content for d in relevant_docs])
            
            # LLM 스트리밍 호출
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            system_instruction = (
                "당신은 한세대학교 학부생 상담원입니다. 아래 학칙을 바탕으로 친절하게 답하세요.\n"
                "1. 리스트나 표를 활용하여 가독성을 높이세요.\n"
                "2. 중요한 키워드는 굵게(**) 표시하세요.\n"
            )
            
            full_prompt = f"{system_instruction}\n[이전 대화]\n{history_text}\n[학칙 정보]\n{context}\n[질문]\n{prompt}"
            
            ai_start = time.time()
            first_chunk = True
            async for chunk in llm.astream(full_prompt):
                if first_chunk:
                    logger.info(f"🤖 [AI 응답 시작] 첫 토큰까지: {time.time() - ai_start:.2f}초")
                    first_chunk = False
                yield chunk.content
            
            logger.info(f"✅ [처리 완료] 총 소요 시간: {time.time() - start_time:.2f}초")
            
        except Exception as e:
            logger.error(f"❌ [AI 생성 에러] {str(e)}", exc_info=True)
            yield f"⚠️ 오류 발생: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

# 5. 앱 초기화 및 라우팅 등록
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    """서버 기동 시 데이터 로드"""
    global global_retriever
    try:
        logger.info("FAISS 벡터 데이터베이스 로딩 시작...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        temp_db = await asyncio.to_thread(FAISS.load_local, "faiss_index", embeddings, allow_dangerous_deserialization=True)
        global_retriever = temp_db.as_retriever(search_kwargs={"k": 6})
        logger.info("✅ 데이터베이스 로드 성공!")
    except Exception as e:
        logger.error(f"❌ 데이터베이스 로드 실패: {str(e)}", exc_info=True)

@app.get("/")
@app.head("/")
async def root():
    """루트 경로 처리"""
    return FileResponse("index.html")

# 마지막에 정적 파일 마운트
app.mount("/", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
