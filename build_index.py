import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def build_vector_db():
    print("=== 벡터 데이터베이스(FAISS) 구축 시작 ===")
    
    docs = []
    
    # 1. 파일 로드
    files = [
        ("학부학칙.pdf", PyPDFLoader),
        ("scholarship.txt", lambda f: TextLoader(f, encoding='utf-8')),
        ("phone_directory.txt", lambda f: TextLoader(f, encoding='utf-8'))
    ]
    
    for filename, LoaderClass in files:
        if os.path.exists(filename):
            print(f"- {filename} 로드 중...")
            loader = LoaderClass(filename)
            docs.extend(loader.load())
        else:
            print(f"경고: {filename} 파일이 없습니다.")
            
    if not docs:
        print("문서가 없습니다. 종료합니다.")
        return

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    total_chunks = len(split_docs)
    print(f"총 {total_chunks}개의 문서 조각을 생성했습니다.")

    # 3. 임베딩 모델 설정
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    print("AI 임베딩 API 호출 시작... (429 에러 방지를 위해 천천히 진행합니다.)")
    
    # 4. 빈 FAISS 데이터베이스 준비
    # 꼼수 방지: 첫 번째 문서를 먼저 넣어서 초기화
    vector_db = FAISS.from_documents([split_docs[0]], embeddings)
    
    # 5. Rate Limit 에러 방지를 위해 1개씩 삽입
    # 무료 버전 한도(1분에 15번 API 요청)를 절대 넘지 않도록 4.5초에 1번씩만 요청합니다
    for i in range(1, total_chunks):
        print(f"  -> {i}/{total_chunks} 조각 임베딩 중...")
        vector_db.add_documents([split_docs[i]])
        time.sleep(4.5)
        
    # 6. 로컬에 저장
    vector_db.save_local("faiss_index")
    print("=== 구축 완료! 'faiss_index' 폴더에 저장되었습니다. ===")

if __name__ == "__main__":
    build_vector_db()
