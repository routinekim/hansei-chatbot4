import os
import time
os.environ["GOOGLE_API_VERSION"] = "v1"

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import requests

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_split(path, source):
    loader = PyPDFLoader(path) if path.endswith(".pdf") else TextLoader(path, encoding="utf-8")
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    return [{"content": c.page_content, "metadata": {"source": source}} for c in chunks]


def embed_and_insert(chunks):
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk['content'][:60]}...")
        embedding = embeddings.embed_query(chunk["content"])
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/chunks",
            headers=HEADERS,
            json={
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": embedding,
            },
        )
        if resp.status_code not in (200, 201):
            print(f"  ⚠️  삽입 실패: {resp.status_code} {resp.text[:100]}")
        if i < len(chunks) - 1:
            time.sleep(4.5)  # Google API rate limit (15 req/min)


files = [
    ("2026haksa.txt", "schedule"),
    ("scholarship.txt", "scholarship"),
    ("phone_directory.txt", "phone"),
    ("학부학칙.pdf", "regulations"),
]

base_dir = os.path.dirname(__file__)

for filename, source in files:
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️  파일 없음: {filename} (건너뜀)")
        continue
    print(f"\n📄 [{source}] {filename} 처리 중...")
    chunks = load_and_split(path, source)
    print(f"  → {len(chunks)}개 청크 생성")
    embed_and_insert(chunks)
    print(f"  ✅ 완료")

print("\n🎉 마이그레이션 완료!")
