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
    """고정된 학사일정 반환"""
    return """📅 **[2026학년도 학사일정]**

| 월 | 일정 및 상세 내용 |
| :--- | :--- |
| **01월** | • [01-06] 2학기 성적 확정<br>• [01-14 ~ 01-16] 동계 계절학기 성적입력... (생략) |
| **02월** | • [02-10 ~ 02-12] 1학기 예비수강신청... (생략) |
| **03월** | • [03-03] 1학기 개강 및 입학식... (생략) |
| **이하 생략...** | (중량) |"""

# 4. 명시적 엔드포인트 정의 (404 방지를 위해 APIRouter와 Mount 제거)

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
            if "학사일정" in prompt.replace(" ", ""):
                 yield "잠시만 기다려주세요. 학사일정을 불러오고 있습니다...\n\n"
                 yield scrape_academic_schedule()
                 return

            # 대화 문맥 구성
            history_text = ""
            if request.history:
                for msg in request.history[-4:]:
                    history_text += f"[{'학생' if msg.role == 'user' else '상담원'}] {msg.content}\n"
            
            search_query = f"{request.history[-1].content if request.history else ''} {prompt}"
            relevant_docs = await asyncio.to_thread(global_retriever.invoke, search_query)
            context = "\n".join([d.page_content for d in relevant_docs])
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            full_prompt = f"한세대학교 상담원으로서 학칙을 바탕으로 답하세요.\n[정보]\n{context}\n[대화기록]\n{history_text}\n[질문]\n{prompt}"
            
            async for chunk in llm.astream(full_prompt):
                yield chunk.content
            
            logger.info(f"✅ [답변 완료] 소요시간: {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ [에러] {str(e)}", exc_info=True)
            yield f"⚠️ 오류 발생: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

# 5. 정적 파일 직접 서빙 (마운트 없이 하나씩 지정)

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
