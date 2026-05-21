-- 1. pgvector 확장 활성화
create extension if not exists vector;

-- 2. chunks 테이블에 embedding 컬럼 추가
alter table chunks add column if not exists embedding vector(768);

-- 3. hnsw 벡터 검색 인덱스
create index if not exists chunks_embedding_idx
  on chunks using hnsw (embedding vector_cosine_ops);

-- 4. 유사도 검색 함수
create or replace function match_chunks(
  query_embedding vector(768),
  match_count int default 6
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    id,
    content,
    metadata,
    1 - (embedding <=> query_embedding) as similarity
  from chunks
  where embedding is not null
  order by embedding <=> query_embedding
  limit match_count;
$$;
