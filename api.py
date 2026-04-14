import os
import asyncio
# 1. 최우선 환경 설정 (임포트 전 실행 필수)
os.environ["GOOGLE_API_VERSION"] = "v1"

import json
import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# 2. AI 라이브러리 임포트 (v1 설정 이후 실행)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hansei Chatbot API",
    description="한세대학교 학업 상담 챗봇"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 데이터 모델
class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    history: Optional[List[Message]] = []

# 4. 챗봇 엔진 클래스
class HanseiBot:
    def __init__(self):
        self.retriever = None
        self.models = [] 
        self.is_ready = False

    async def initialize(self):
        try:
            start_time = time.time()
            logger.info("📡 [초기화] v1 API 안정화 버전 로드 시작...")

            # 4-1. 벡터 DB 로드
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_db = await asyncio.to_thread(
                FAISS.load_local, 
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 6})
            
            # 4-2. 모델 풀 구성 (안정 버전 v1 우선)
            model_names = [
                "gemini-1.5-flash", 
                "gemini-2.5-flash-lite", 
                "gemini-2.0-flash"
            ]
            
            for name in model_names:
                try:
                    llm = ChatGoogleGenerativeAI(
                        model=name, 
                        temperature=0,
                        timeout=20.0,
                        max_retries=0
                    )
                    self.models.append({"name": name, "obj": llm})
                    logger.info(f"✅ [모델 로드 성공] {name}")
                except Exception as e:
                    logger.warning(f"⚠️ [모델 로드 제외] {name}: {str(e)}")

            if not self.models:
                raise Exception("사용 가능한 모델이 없습니다.")

            self.is_ready = True
            logger.info(f"✅ 초기화 완료 ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ 초기화 실패: {str(e)}")
            self.is_ready = False

bot = HanseiBot()

def scrape_academic_schedule():
    return """📅 **[2026학년도 학사일정]** (상세 정보는 검색을 통해 확인하세요)"""

@app.on_event("startup")
async def startup_event():
    await bot.initialize()

@app.get("/health")
async def health():
    return {
        "status": "online" if bot.is_ready else "initializing",
        "current_api_version": os.environ.get("GOOGLE_API_VERSION", "not set"),
        "loaded_models": [m["name"] for m in bot.models]
    }

@app.post("/chat")
async def chat(request: QueryRequest):
    request_start = time.time()
    if not bot.is_ready:
        raise HTTPException(status_code=503, detail="서버 준비 중")

    async def response_generator():
        prompt = request.query
        if "학사일정" in prompt.replace(" ", ""):
             yield scrape_academic_schedule()
             return

        relevant_docs = await asyncio.to_thread(bot.retriever.invoke, prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        
        yield "최적의 엔진으로 답변을 생성하는 중입니다... ⏳\n\n"

        history_text = ""
        if request.history:
            for msg in request.history[-4:]:
                history_text += f"[{'User' if msg.role == 'user' else 'AI'}] {msg.content}\n"
        
        full_prompt = (
            "한세대학교 상담원으로서 답변하세요.\n\n"
            f"[참고정보]\n{context}\n\n[학생질문]\n{prompt}"
        )

        success = False
        for model_entry in bot.models:
            model_name = model_entry["name"]
            llm = model_entry["obj"]
            try:
                logger.info(f"🚀 [시도] 모델: {model_name}")
                async for chunk in llm.astream(full_prompt):
                    if chunk.content:
                        yield chunk.content
                success = True
                break
            except Exception as e:
                logger.warning(f"⚠️ [실패 및 전환] {model_name}: {str(e)}")
                continue

        if not success:
            yield f"\n\n⚠️ 모든 서비스가 현재 지연 중입니다. 잠시 후 시도해 주세요."

    return StreamingResponse(response_generator(), media_type="text/plain")

@app.get("/")
async def serve_index():
    path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(path)

@app.get("/app.js")
async def serve_js():
    path = os.path.join(os.path.dirname(__file__), "app.js")
    return FileResponse(path)

@app.get("/style.css")
async def serve_css():
    path = os.path.join(os.path.dirname(__file__), "style.css")
    return FileResponse(path)

@app.get("/hanbi.gif")
async def serve_gif():
    path = os.path.join(os.path.dirname(__file__), "hanbi.gif")
    if os.path.exists(path):
        return FileResponse(path)
    return {"error": "file not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
