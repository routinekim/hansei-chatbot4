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
    description="한세대학교 학업 상담 챗봇 (표 형식 및 문맥 유지 강화)"
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
        self.schedule_data = "" # 학사일정 텍스트 원본 보관
        self.is_ready = False

    async def initialize(self):
        try:
            start_time = time.time()
            logger.info("📡 [초기화] AI 엔진 및 데이터 로드 시작...")

            # 4-1. 벡터 DB 로드
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_db = await asyncio.to_thread(
                FAISS.load_local, 
                "faiss_index", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self.retriever = vector_db.as_retriever(search_kwargs={"k": 6})
            
            # 4-2. 학사일정 데이터 로드
            haksa_path = os.path.join(os.path.dirname(__file__), "2026haksa.txt")
            if os.path.exists(haksa_path):
                with open(haksa_path, "r", encoding="utf-8") as f:
                    self.schedule_data = f.read()
                logger.info("📅 [데이터 로드] 2026학년도 학사일정 로드 완료")
            
            # 4-3. 모델 풀 구성 (최신 모델 위주)
            model_names = [
                # "gemini-2.0-flash-lite",
                # "gemini-1.5-flash", 
                "gemini-2.5-flash-lite", 
                # "gemini-2.0-flash"
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
                    logger.info(f"✅ [모델 구성 성공] {name}")
                except Exception as e:
                    logger.warning(f"⚠️ [모델 구성 제외] {name}: {str(e)}")

            if not self.models:
                raise Exception("사용 가능한 모델이 없습니다.")

            self.is_ready = True
            logger.info(f"✅ [초기화 완료] ({time.time() - start_time:.2f}s)")
        except Exception as e:
            logger.error(f"❌ [초기화 실패] {str(e)}")
            self.is_ready = False

bot = HanseiBot()

# 5. API 엔드포인트

@app.on_event("startup")
async def startup_event():
    await bot.initialize()

@app.get("/health")
async def health():
    return {
        "status": "online" if bot.is_ready else "initializing",
        "current_api_version": os.environ.get("GOOGLE_API_VERSION", "not set"),
        "loaded_models": [m["name"] for m in bot.models],
        "schedule_loaded": bool(bot.schedule_data)
    }

@app.post("/chat")
async def chat(request: QueryRequest):
    request_start = time.time()
    
    if not bot.is_ready:
        raise HTTPException(status_code=503, detail="서버 준비 중")

    async def response_generator():
        prompt = request.query
        
        # 1. 관련 정보 검색 (RAG)
        relevant_docs = await asyncio.to_thread(bot.retriever.invoke, prompt)
        context = "\n".join([d.page_content for d in relevant_docs])
        
        # 🔔 [Keep-alive] 연결 유지를 위한 즉시 스트리밍
        yield "데이터를 분석하고 최적의 엔진으로 답변을 구성하고 있습니다... ⏳\n\n"

        # 2. 대화 이력 구성 (문맥 유지 핵심)
        history_text = ""
        if request.history:
            # 최근 5개 대화 정도를 문맥으로 제공
            for msg in request.history[-5:]:
                history_text += f"{'학생' if msg.role == 'user' else '한비'}: {msg.content}\n"
        
        # 3. 강화된 시스템 프롬프트
        system_instructions = (
            "당신은 한세대학교의 공식 챗봇 '한비'입니다. 친절하고 당당하며 예의바른 학생 상담원 톤으로 답변하세요.\n\n"
            "**[핵심 답변 지침]**\n"
            "1. 제공된 [학사일정] 및 [검색결과] 내용을 바탕으로만 답변하세요.\n"
            "2. **학사일정이나 반복되는 날짜 정보는 반드시 Markdown 표(Table) 형식**을 사용하여 가독성을 높이세요.\n"
            "3. 이전 대화 흐름([대화 이력])을 기억하고 질문이 모호하더라도 문맥에 맞게 답변하세요.\n"
            "4. 불필요한 서론은 생략하고 바로 본론부터 친절하게 답하세요.\n\n"
            f"**[참초 데이터 - 2026 학사일정]**\n{bot.schedule_data}\n\n"
            f"**[참조 데이터 - 검색결과]**\n{context}\n\n"
            f"**[최근 대화 이력]**\n{history_text}\n"
        )
        
        full_prompt = f"{system_instructions}\n**질문**: {prompt}"

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
                logger.info(f"✅ [답변 완료] 사용 엔진: {model_name} (총 소요: {time.time() - request_start:.2f}s)")
                break
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                logger.warning(f"⚠️ [전환 시도] {model_name} 실패: {error_str[:60]}...")
                continue

        if not success:
            logger.error(f"💀 [최종 실패] 모든 모델 응답 불가. 에러: {last_error}")
            yield f"\n\n⚠️ 현재 모든 AI 엔진의 사용량이 많아 답변을 완성하지 못했습니다. 잠시 후 새로고침(F5)하여 다시 시도해 주세요."

    return StreamingResponse(response_generator(), media_type="text/plain")

# 6. 정적 파일 서빙 (절대 경로 보장)
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
