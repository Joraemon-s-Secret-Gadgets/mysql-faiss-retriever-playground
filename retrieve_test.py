import os
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import get_mysql_db_config
from src.retrieval.hybrid_retriever import HybridRetriever

if __name__ == "__main__":
    # 캐시 경로 설정 (드라이브 내 원하는 경로)
    cache_dir = "./hf_cache"

    # 환경 변수 설정 (Hugging Face 라이브러리가 이 경로를 참조하게 함)
    os.environ['HF_HOME'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(cache_dir, "sentence_transformers")

    print("✅ 모든 AI 모델 및 데이터셋 캐시 경로가 고정되었습니다.")
    
    print("⏳ 임베딩 모델 로딩 중...")
    
    DB_CONFIG = get_mysql_db_config()
    
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'}, # GPU 없으면 'cpu'
        encode_kwargs={'normalize_embeddings': True},
        cache_folder = os.environ['SENTENCE_TRANSFORMERS_HOME']
    )
    
    retriever = HybridRetriever(
        db_config=DB_CONFIG,
        embeddings=hf_embeddings,
        top_n=3,       
        initial_k=5,
        index_folder="data/faiss_index_high" #faiss 인덱스 저장 경로로 지정
    )    
    test_queries = [
    """
**경력 및 경험:**
- 대학 내 다양한 팀 프로젝트 참여
- React Native를 활용한 모바일 애플리케이션 개발 경험 (개인 프로젝트)
- Expo를 이용한 간단한 애플리케이션 빌드 및 배포 경험
- HTML/CSS로 간단한 웹 페이지 제작

**기술 및 역량:**
- React Native: 기본적인 이해 및 사용 경험
- JavaScript/TypeScript: 학습 후 간단한 프로젝트에서 사용
- Git: 버전 관리 기본 능숙
- HTML/CSS: 기초적인 이해 및 활용 능력
- 커뮤니케이션 스킬: 팀 프로젝트 진행 시 협업 경험

**기타:**
- GitHub에 개인 프로젝트 저장소 운영 중
- 디자인 시스템 활용 또는 모바일 애니메이션 구현 경험 없음

    """
    ]

    print("\n" + "="*50)
    print("🚀 Hybrid Retriever 테스트 시작")
    print("="*50)

    searched_selfintro = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 [테스트 {i}] 질문: {query}\n")

        try:
            # 리트리버 실행 (invoke 사용)
            results = retriever.invoke(query)

            if not results:
                print("⚠️ 검색 결과가 없습니다.")
                continue

            # 결과 출력
            for idx, doc in enumerate(results, 1):
                selfintro = doc.page_content
                print(f'{idx} 번째 이력서 유사도: {doc.metadata.get("relevance_score")}')
                print(f'{idx} 번째 자소서\n{selfintro}')
                print(f'{idx} 번째 자소서 평가 점수\n{doc.metadata.get("selfintro_score")}\n')
        except Exception as e:
            print(f"❌ 에러 발생: {e}")

    print("\n" + "="*50)
    print("✨ 모든 테스트가 종료되었습니다.")    