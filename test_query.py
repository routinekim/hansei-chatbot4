import os
from dotenv import load_dotenv
load_dotenv()

from api import load_data, chat_endpoint, QueryRequest

# 데이타 로드 실행
load_data()

# 테스트 쿼리
query = "직전 학기 성적 3.0 이상을 받아야 하는 장학금들을 모두 찾아서 표로 나열해줘."
req = QueryRequest(query=query)

print("질문:", query)
print("="*50)
print("답변 중...")
try:
    resp = chat_endpoint(req)
    print(resp.answer)
except Exception as e:
    print(f"에러: {e}")
