import os
import asyncio
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain 관련 모듈
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. 환경 설정 및 로깅
load_dotenv()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hansei Chatbot API",
    description="한세대학교 학사 챗봇 백엔드 API 서버 (유료 인스턴스 최적화 버전)"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 데이터 모델
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

# 3. 챗봇 엔진 클래스 (싱글톤 패턴)
class HanseiBot:
    def __init__(self):
        self.retriever = None
        self.llm = None
        self.is_ready = False

    async def initialize(self):
        """서버 시작 시 한 번만 실행되어 리소스를 로드합니다."""
        try:
            start_time = time.time()
            logger.info("📡 [초기화] 벡터 DB 및 AI 모델 로드 시작...")

            # 3-1. 벡터 DB 로드
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_db = await asyncio.to_thread(
                FAISS.load_local, 
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 6})
            
            # 3-2. LLM 초기화 (Flash 모델 고정, Pro 제외)
            # 유료 서버의 안정성을 위해 객체를 미리 생성하여 재사용합니다.
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite", 
                temperature=0,
                max_retries=3
            )
            
            self.is_ready = True
            logger.info(f"✅ [초기화 완료] 소요 시간: {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"❌ [초기화 실패] {str(e)}")
            self.is_ready = False

bot = HanseiBot()

# 4. 유틸리티 함수
def scrape_academic_schedule():
    return """📅 **[2026학년도 학사일정]**

| 월 | 일정 및 상세 내용 |
| :--- | :--- |
| **01월** | • [01-06] 2학기 성적 확정<br>• [01-14 ~ 01-16] 동계 계절학기 성적입력... |
| **02월** | • [02-10 ~ 02-12] 1학기 예비수강신청... |
| **03월** | • [03-03] 1학기 개강 및 입학식... |
| **이하 생략...** | (검색 기능을 통해 상세 정보를 확인하세요) |"""

# 5. API 엔드포인트

@app.on_event("startup")
async def startup_event():
    await bot.initialize()

@app.get("/health")
async def health():
    """서버 상태 확인 (경량화 버전)"""
    return {
        "status": "online" if bot.is_ready else "initializing",
        "engine": "gemini-2.5-flash-lite"
    }

@app.get("/debug/models")
async def list_available_models():
    """사용 가능한 구글 AI 모델 리스트 조회 (디버그 전용)"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"available_models": models}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat")
async def chat(request: QueryRequest):
    """실시간 스트리밍 답변 엔드포인트 (성능 로깅 포함)"""
    request_start = time.time()
    logger.info(f"🔥 [요청 수신] 질문: {request.query}")
    
    if not bot.is_ready:
        raise HTTPException(status_code=503, detail="서버가 아직 준비 중입니다. 잠시 후 다시 시도해 주세요.")

    async def response_generator():
        try:
            prompt = request.query
            
            # 1. 고정 정보 가로채기
            if "학사일정" in prompt.replace(" ", ""):
                 yield scrape_academic_schedule()
                 return

            # 2. 관련 정보 검색 (Profiling)
            search_start = time.time()
            relevant_docs = await asyncio.to_thread(bot.retriever.invoke, prompt)
            context = "\n".join([d.page_content for d in relevant_docs])
            search_duration = time.time() - search_start
            logger.info(f"🔍 [데이터 검색 완료] 소요 시간: {search_duration:.2f}s")
            
            # 3. 대화 문맥 구성
            history_text = ""
            if request.history:
                for msg in request.history[-4:]:
                    history_text += f"[{'User' if msg.role == 'user' else 'AI'}] {msg.content}\n"
            
            full_prompt = (
                "당신은 한세대학교 학부생 상담원입니다. 아래 학칙 및 지침을 바탕으로 당당하고 친절하게 답하세요.\n"
                "가독성을 위해 Markdown 표(Table)나 굵게 표시를 적극 활용하세요.\n\n"
                f"[관련 정보]\n{context}\n\n[이전 대화]\n{history_text}\n\n[학생 질문]\n{prompt}"
            )
            
            # 4. AI 생성 및 스트리밍
            gen_start = time.time()
            first_chunk = True
            async for chunk in bot.llm.astream(full_prompt):
                if first_chunk:
                    logger.info(f"⚡ [AI 생성 시작] 첫 토큰 소요 시간: {time.time() - gen_start:.2f}s")
                    first_chunk = False
                if chunk.content:
                    yield chunk.content
            
            total_duration = time.time() - request_start
            logger.info(f"✅ [답변 완료] 총 소요 시간: {total_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ [에러 발생] {str(e)}", exc_info=True)
            yield f"⚠️ 오류 발생: {str(e)}"

    return StreamingResponse(response_generator(), media_type="text/plain")

# 6. 정적 파일 서빙
@app.get("/")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
