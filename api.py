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
    description="한세대학교 학사 챗봇 백엔드 API 서버 (503 폴백 강화 버전)"
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

# 3. 챗봇 엔진 클래스 (실시간 모델 풀 관리)
class HanseiBot:
    def __init__(self):
        self.retriever = None
        self.models = [] # 사용할 수 있는 LLM 객체들의 리스트
        self.is_ready = False

    async def initialize(self):
        """서버 시작 시 모든 후보 모델을 미리 초기화하여 풀(Pool)을 생성합니다."""
        try:
            start_time = time.time()
            logger.info("📡 [초기화] 벡터 DB 및 AI 모델 풀 로드 시작...")

            # 3-1. 벡터 DB 로드
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_db = await asyncio.to_thread(
                FAISS.load_local, 
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 6})
            
            # 3-2. LLM 모델 풀 구성 (우선순위 순)
            model_names = [
                "gemini-2.5-flash", 
                "gemini-2.5-flash-lite", 
                "gemini-2.0-flash", 
                "gemini-1.5-flash"
            ]
            
            for name in model_names:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=name, 
                        temperature=0,
                        max_retries=1 # 실시간 폴백을 위해 내부 재시도는 최소화
                    )
                    self.models.append({"name": name, "obj": llm})
                    logger.info(f"✅ [모델 로드 완료] {name}")
                except Exception as e:
                    logger.warning(f"⚠️ [모델 로드 실패] {name}: {str(e)}")

            if not self.models:
                raise Exception("사용 가능한 AI 모델을 하나도 로드하지 못했습니다.")

            self.is_ready = True
            logger.info(f"✅ [전체 초기화 완료] 소요 시간: {time.time() - start_time:.2f}s")
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
    return {
        "status": "online" if bot.is_ready else "initializing",
        "loaded_models": [m["name"] for m in bot.models],
        "is_ready": bot.is_ready
    }

@app.post("/chat")
async def chat(request: QueryRequest):
    """실시간 스트리밍 답변 엔드포인트 (실시간 503 폴백 로직 포함)"""
    request_start = time.time()
    
    if not bot.is_ready:
        raise HTTPException(status_code=503, detail="서버 준비 중입니다.")

    async def response_generator():
        prompt = request.query
        
        # 1. 고정 정보 가로채기
        if "학사일정" in prompt.replace(" ", ""):
             yield scrape_academic_schedule()
             return

        # 2. 관련 정보 검색
        relevant_docs = await asyncio.to_thread(bot.retriever.invoke, prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        
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

        # 4. 실시간 모델 폴백 시도
        success = False
        last_error = ""

        for model_entry in bot.models:
            model_name = model_entry["name"]
            llm = model_entry["obj"]
            
            try:
                logger.info(f"🚀 [답변 생성 시도] 모델: {model_name}")
                gen_start = time.time()
                first_chunk = True
                
                async for chunk in llm.astream(full_prompt):
                    if first_chunk:
                        logger.info(f"⚡ [스트리밍 시작] {model_name} (응답시간: {time.time() - gen_start:.2f}s)")
                        first_chunk = False
                    if chunk.content:
                        yield chunk.content
                
                success = True
                logger.info(f"✅ [답변 완료] 성공 모델: {model_name} (총 소요: {time.time() - request_start:.2f}s)")
                break # 성공 시 루프 탈출
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                # 503(인기 폭주) 또는 일시적인 사용 불가 에러인 경우 다음 모델 시도
                if "503" in error_str or "demand" in error_str.lower() or "Resource exhausted" in error_str:
                    logger.warning(f"⚠️ [503/과부하 감지] {model_name} 실패, 다음 모델로 전환합니다. (사유: {error_str[:100]}...)")
                    continue
                else:
                    # 그 외의 치명적 에러(401, 403 등)는 즉시 중단
                    logger.error(f"❌ [치명적 에러] {model_name}: {error_str}")
                    yield f"⚠️ 챗봇 엔진 오류가 발생했습니다: {error_str}"
                    return

        if not success:
            logger.error(f"💀 [최종 실패] 모든 모델이 응답할 수 없습니다. 마지막 에러: {last_error}")
            yield f"⚠️ 현재 모든 AI 모델의 사용량이 많아 답변을 드릴 수 없습니다. 잠시 후 다시 시도해 주세요. (503 Fallback Failed)"

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
