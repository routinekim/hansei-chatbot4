import os
import asyncio
import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 환경 설정 및 로깅
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hansei Chatbot API",
    description="한세대학교 학사 챗봇 백엔드 API 서버"
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

# 3. 유틸리티 함수
def scrape_academic_schedule():
    """고정된 2026학년도 전체 학사일정 반환"""
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

# 4. 명시적 엔드포인트 정의

@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "retriever": "ready" if global_retriever is not None else "loading"}

@app.post("/chat")
async def chat(request: QueryRequest):
    """실시간 스트리밍 답변 엔드포인트"""
    start_time = time.time()
    logger.info(f"🔥 [요청 수신] /chat - 질문: {request.query}")
    
    if global_retriever is None:
        raise HTTPException(status_code=500, detail="서버 데이터 로딩 중입니다.")

    async def response_generator():
        try:
            prompt = request.query
            # 학사일정 가로채기
            if "학사일정" in prompt.replace(" ", ""):
                 yield scrape_academic_schedule()
                 return

            # 대화 문맥 구성
            history_text = ""
            if request.history:
                for msg in request.history[-4:]:
                    history_text += f"[{'학생' if msg.role == 'user' else '상담원'}] {msg.content}\n"
            
            # 검색 및 컨텍스트 생성
            search_query = prompt
            relevant_docs = await asyncio.to_thread(global_retriever.invoke, search_query)
            context = "\n".join([d.page_content for d in relevant_docs])
            
            # AI 모델 호출 (사용자 요청: gemini-2.5-flash 모델 적용)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            
            full_prompt = (
                "당신은 한세대학교 학부생 상담원입니다. 아래 학칙 및 지침을 바탕으로 당당하고 친절하게 답하세요.\n"
                "가독성을 위해 다음 원칙을 반드시 지키세요:\n"
                "1. 나열할 항목이 2개 이상이면 Markdown 표(Table) 형식을 적극 활용하세요.\n"
                "2. 중요한 키워드는 **굵게** 표시하세요.\n\n"
                f"[관련 정보]\n{context}\n\n[이전 대화]\n{history_text}\n\n[학생 질문]\n{prompt}"
            )
            
            async for chunk in llm.astream(full_prompt):
                if chunk.content:
                    yield chunk.content
            
            logger.info(f"✅ [답변 완료] 소요시간: {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ [에러 발생] {str(e)}", exc_info=True)
            yield f"⚠️ 오류 발생: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
@app.head("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/app.js")
async def serve_js():
    return FileResponse("app.js")

@app.get("/style.css")
async def serve_css():
    return FileResponse("style.css")

@app.get("/hanbi.gif")
async def serve_gif():
    if os.path.exists("hanbi.gif"):
        return FileResponse("hanbi.gif")
    return {"error": "file not found"}

# 6. 서버 시작 설정
@app.on_event("startup")
async def startup_event():
    global global_retriever
    try:
        logger.info("📡 벡터 DB 로드 중...")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        temp_db = await asyncio.to_thread(FAISS.load_local, "faiss_index", embeddings, allow_dangerous_deserialization=True)
        global_retriever = temp_db.as_retriever(search_kwargs={"k": 6})
        logger.info("✅ 로드 완료")
    except Exception as e:
        logger.error(f"❌ 로드 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
