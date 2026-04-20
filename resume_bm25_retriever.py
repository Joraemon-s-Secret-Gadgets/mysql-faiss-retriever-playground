# resume_bm25_retriever.py
import os
import sys
import io
import ssl
import pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.append('.')

import numpy as np
import pymysql
from pymysql import Error
from typing import List, Dict, Annotated
from collections import Counter
from pathlib import Path
from rank_bm25 import BM25Okapi
from kiwipiepy import Kiwi

from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langsmith import traceable
from pydantic import ConfigDict, PrivateAttr, SkipValidation
import time


# =============================================
# [보류] TF-IDF 직무별 키워드 추출기
# 생성 품질 평가 목적으로 추후 활용 예정
# =============================================
# class JobKeywordExtractor:
#     def __init__(self, top_n=20):
#         self.top_n = top_n
#         self.keywords_by_position = {}
#
#         self.stopwords = {
#             '경험', '개발', '활용한', '활용', '대한', '사용', '관리', '운영',
#             '이해', '능력', '역량', '설계', '구현', '기반', '수행', '처리',
#             '환경', '서비스', '시스템', '도구', '기술', '업무', '및', '등',
#             '위한', '통한', '이상', '보유', '가능', '관련', '필요', '사용한',
#             '있는', '있음', '통해', '대해', '하여', '으로', '에서', '에대한',
#             '또는', '이용한', '를', '을', '이', '가', '은', '는',
#         }
#
#         self.tfidf = TfidfVectorizer(
#             max_features=5000,
#             ngram_range=(1, 2),
#             min_df=5,
#             max_df=0.8,
#             sublinear_tf=True
#         )
#
#     def _combine_job_text(self, row):
#         return " ".join([
#             row['qualifications'],
#             row['responsibilities'],
#             row['preferred']
#         ])
#
#     def _filter_keywords(self, keywords):
#         filtered = []
#         for kw in keywords:
#             tokens = kw.split()
#             if not any(token in self.stopwords for token in tokens):
#                 filtered.append(kw)
#         return filtered
#
#     def fit(self, df):
#         corpus = df.apply(self._combine_job_text, axis=1).tolist()
#         tfidf_matrix = self.tfidf.fit_transform(corpus)
#         feature_names = self.tfidf.get_feature_names_out()
#
#         for position in df['position_type'].unique():
#             mask = df['position_type'] == position
#             position_matrix = tfidf_matrix[mask.values]
#             mean_tfidf = position_matrix.mean(axis=0).A1
#
#             top_indices = mean_tfidf.argsort()[::-1][:self.top_n * 3]
#             candidates = [feature_names[i] for i in top_indices]
#             filtered = self._filter_keywords(candidates)[:self.top_n]
#
#             self.keywords_by_position[position] = filtered
#             print(f"[{position}] 키워드 추출 완료: {filtered}")
#
#         return self
#
#     def get_keywords(self, position):
#         return self.keywords_by_position.get(position, [])


# =============================================
# 1. SSL 설정 (윈도우 환경용)
# =============================================
def get_db_config_with_ssl():
    """
    윈도우 환경에서 TiDB Cloud SSL 연결 설정
    ca 인증서 없이 SSL 컨텍스트를 직접 생성해서 연결
    """
    from src.config import get_mysql_db_config
    config = get_mysql_db_config()
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    config['ssl'] = ssl_ctx
    config.pop('ssl_ca', None)
    config.pop('ssl_verify_cert', None)
    config.pop('use_pure', None)
    return config


# =============================================
# 2. BM25 이력서 인덱서
# =============================================
class ResumeBM25Index:
    """
    DB의 resume_cleaned를 BM25로 인덱싱
    사용자 이력서(쿼리) vs DB 이력서들 유사도 계산

    사용 방법:
        # DB 기반 빌드 (운영)
        bm25_index = ResumeBM25Index()
        if not bm25_index.load():
            bm25_index.build_from_db(db_config)
            bm25_index.save()

        # train_df 기반 빌드 (테스트)
        bm25_index.build(
            db_ids=list(train_df.index),
            resumes=train_df['resume_cleaned'].tolist(),
            positions=train_df['position_type'].tolist()
        )
    """
    def __init__(self, cache_path: str = "bm25_index.pkl"):
        self.bm25_by_position = {}
        # {position: {'bm25': BM25Okapi, 'db_ids': [id1, id2, ...]}}
        self.kiwi = Kiwi()
        self.cache_path = cache_path

    def _tokenize(self, text):
        """
        Kiwi 형태소 분석기로 토크나이징
        명사(NN), 동사(VV), 형용사(VA), 영어(SL) 위주로 추출
        조사, 어미 등 불필요한 품사 제거
        """
        target_pos = {
            'NNG', 'NNP',  # 일반명사, 고유명사
            'VV', 'VA',    # 동사, 형용사
            'SL',          # 영어 (Python, FastAPI 등 기술스택)
            'SN',          # 숫자
        }
        tokens = []
        for token in self.kiwi.tokenize(text):
            if token.tag in target_pos:
                tokens.append(token.form.lower())
        return tokens

    def build(self, db_ids: List[int], resumes: List[str], positions: List[str]):
        """
        train_df 기반 BM25 인덱스 빌드 (테스트용)
        실제 운영에서는 build_from_db() 사용 권장
        """
        position_data = {}
        for db_id, resume, position in zip(db_ids, resumes, positions):
            if position not in position_data:
                position_data[position] = {'db_ids': [], 'resumes': []}
            position_data[position]['db_ids'].append(db_id)
            position_data[position]['resumes'].append(resume)

        for position, data in position_data.items():
            tokenized = [self._tokenize(r) for r in data['resumes']]
            self.bm25_by_position[position] = {
                'bm25': BM25Okapi(tokenized),
                'db_ids': data['db_ids']
            }
            print(f"[{position}] BM25 인덱스 빌드 완료: {len(data['db_ids'])}건")

        return self

    def build_from_db(self, db_config: dict):
        """
        DB에서 직접 resume_cleaned, position_type을 가져와 BM25 인덱스 빌드 (운영용)
        train_df의 임시 index 대신 실제 DB id를 사용해 id 불일치 문제 해결
        """
        conn = pymysql.connect(**db_config, cursorclass=pymysql.cursors.DictCursor)
        try:
            with conn.cursor() as cursor:
                sql = """SELECT
                        j.position_type,
                        r.id,
                        r.resume_cleaned,
                        vec_as_text(sv.embedding)
                        FROM job_posts j
                        JOIN applicant_records r ON j.id = r.jobpost_id
                        JOIN resume_vectors sv ON r.id = sv.record_id;"""
                cursor.execute(sql)
                rows = cursor.fetchall()

            if not rows:
                print("DB에 데이터가 없습니다.")
                return self

            position_counts = Counter(r['position_type'] for r in rows)
            print(f"DB position_type 분포: {position_counts}")

            db_ids = [r['id'] for r in rows]
            resumes = [r['resume_cleaned'] for r in rows]
            positions = [r['position_type'] for r in rows]

            print(f"DB에서 {len(rows)}건 로드 완료")
            return self.build(db_ids, resumes, positions)

        except Error as e:
            print(f"DB 조회 에러: {e}")
            raise e
        finally:
            conn.close()

    def save(self):
        """BM25 인덱스를 pickle로 저장"""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.bm25_by_position, f)
        print(f"BM25 인덱스 저장 완료: {self.cache_path}")

    def load(self) -> bool:
        """
        저장된 BM25 인덱스 로드
        캐시 파일이 없으면 False 반환 → build_from_db() 실행 필요
        캐시 파일이 있으면 로드 후 True 반환
        """
        if not Path(self.cache_path).exists():
            return False
        with open(self.cache_path, 'rb') as f:
            self.bm25_by_position = pickle.load(f)
        print(f"BM25 인덱스 로드 완료: {self.cache_path}")
        for position, data in self.bm25_by_position.items():
            print(f"  [{position}] {len(data['db_ids'])}건")
        return True

    def get_scores(self, query_resume: str, position: str) -> Dict[int, float]:
        """
        사용자 이력서(쿼리) vs 직무별 DB 이력서 BM25 점수 계산

        Returns:
            {db_id: normalized_score} 딕셔너리
        """
        entry = self.bm25_by_position.get(position)
        if entry is None:
            return {}

        bm25 = entry['bm25']
        db_ids = entry['db_ids']
        tokenized_query = self._tokenize(query_resume)

        raw_scores = bm25.get_scores(tokenized_query)

        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        min_score = float(raw_scores.min())

        if max_score == min_score:
            return {db_id: 0.0 for db_id in db_ids}

        normalized = (raw_scores - min_score) / (max_score - min_score)
        normalized = np.clip(normalized, 0.0, 1.0)

        return {db_id: round(float(score), 4)
                for db_id, score in zip(db_ids, normalized)}


# =============================================
# 3. ResumeBM25Retriever
# =============================================
class ResumeBM25Retriever(BaseRetriever):
    """
    기존 HybridRetriever에 BM25 reranking 추가
    DB에는 high grade 데이터만 적재되므로 별도 grade 필터 없음

    검색 흐름:
    1. FAISS 임베딩 유사도로 initial_k개 후보 추출
    2. BM25로 후보들 reranking
    3. hybrid_score = embedding * w_embedding + bm25 * w_bm25
    4. hybrid_score 내림차순 정렬 후 top_n개 반환

    position은 invoke() 호출 시 query와 함께 "position||query" 형태로 전달
    예: retriever.invoke("ai engineer||LangChain 프로젝트 경험이 있습니다")
    """
    embeddings: Annotated[any, SkipValidation]
    top_n: int = 3
    initial_k: int = 50
    index_folder: str = "faiss_index"
    w_embedding: float = 0.6
    w_bm25: float = 0.4

    vectorstore: FAISS = None

    _conn = PrivateAttr()
    _bm25_index = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, db_config, bm25_index: ResumeBM25Index, **kwargs):
        super().__init__(**kwargs)
        self._conn = pymysql.connect(
            **db_config,
            cursorclass=pymysql.cursors.DictCursor
        )
        self._bm25_index = bm25_index
        self._get_vector_index()

    def __del__(self):
        """객체 소멸 시 DB 연결 종료 - 초기화 실패해도 에러 없이 처리"""
        try:
            if self._conn and self._conn.open:
                self._conn.close()
        except Exception:
            pass

    def _get_vector_index(self):
        """FAISS 인덱스 로드"""
        self.vectorstore = FAISS.load_local(
            folder_path=self.index_folder,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("FAISS 인덱스 로드 완료")

    def _parse_query(self, raw_query: str):
        """
        "position||query" 형태로 넘어온 쿼리를 분리
        구분자 없으면 position=None으로 처리
        """
        if "||" in raw_query:
            position, query = raw_query.split("||", 1)
            return position.strip(), query.strip()
        return None, raw_query.strip()

    @traceable(name="ResumeBM25Retriever", process_inputs=lambda x: {}, process_outputs=lambda x: {})
    def _get_relevant_documents(self, query: str) -> List[Document]:
        if self.vectorstore is None:
            self._get_vector_index()

        position, actual_query = self._parse_query(query)

        # 1. FAISS 임베딩 유사도 검색
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            actual_query, k=self.initial_k
        )
        embedding_score_map = {
            int(doc.page_content): float(score)
            for doc, score in docs_and_scores
        }

        # 2. BM25 유사도 계산 (position이 없으면 BM25 스킵)
        bm25_score_map = {}
        if position:
            bm25_score_map = self._bm25_index.get_scores(actual_query, position)

        # 3. Hybrid Score 계산
        hybrid_scores = {}
        for db_id in embedding_score_map:
            emb_score = embedding_score_map.get(db_id, 0.0)
            bm25_score = bm25_score_map.get(db_id, 0.0)
            hybrid_scores[db_id] = round(
                self.w_embedding * emb_score + self.w_bm25 * bm25_score, 4
            )

        # 4. hybrid_score 내림차순 정렬 후 top_n개 선택
        candidates = [doc for doc, _ in docs_and_scores]
        candidates.sort(
            key=lambda doc: hybrid_scores.get(int(doc.page_content), 0.0),
            reverse=True
        )
        target_db_ids = [int(d.page_content) for d in candidates[:self.top_n]]

        # 5. MySQL에서 최종 문서 페치
        return self._fetch_final_documents(target_db_ids, hybrid_scores, bm25_score_map)

    def _fetch_final_documents(
        self,
        db_ids: List[int],
        hybrid_scores: Dict[int, float],
        bm25_scores: Dict[int, float]
    ) -> List[Document]:
        if not db_ids:
            return []

        self._conn.ping(reconnect=True)
        cursor = self._conn.cursor()

        try:
            format_strings = ','.join(['%s'] * len(db_ids))
            sql = f"""
                SELECT id, selfintro, resume_cleaned
                FROM applicant_records
                WHERE id IN ({format_strings})
            """
            cursor.execute(sql, tuple(db_ids))
            rows = cursor.fetchall()
            id_map = {r['id']: r for r in rows}

            final_docs = []
            for db_id in db_ids:
                if db_id in id_map:
                    record = id_map[db_id]
                    doc = Document(
                        page_content=record['selfintro'],
                        metadata={
                            "id": db_id,
                            "hybrid_score": hybrid_scores.get(db_id),
                            "bm25_score": bm25_scores.get(db_id),
                        }
                    )
                    final_docs.append(doc)

            return final_docs

        except Error as e:
            print(f"MySQL 에러: {e}")
            return []
        finally:
            cursor.close()
            # _conn은 ping(reconnect=True)로 재사용하는 구조라
            # 여기서 닫지 않고 __del__에서 닫음


# =============================================
# 4. 실행 - 전체 직무 검색 품질 검증
# =============================================
if __name__ == "__main__":
    from langchain_huggingface import HuggingFaceEmbeddings

    cache_dir = "./hf_cache"
    os.environ['HF_HOME'] = cache_dir
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.path.join(cache_dir, "sentence_transformers")
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = "1"

    # ── DB 설정 ──
    DB_CONFIG = get_db_config_with_ssl()

    # ── BM25 인덱스 빌드 또는 캐시 로드 ──
    print("BM25 인덱스 준비 중...")
    bm25_index = ResumeBM25Index(
        cache_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bm25_index.pkl")
    )

    if not bm25_index.load():
        # 캐시 없으면 DB에서 빌드 후 저장
        bm25_index.build_from_db(DB_CONFIG)
        bm25_index.save()

    # ── 임베딩 모델 로드 ──
    print("\n임베딩 모델 로드 중...")
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        cache_folder=os.environ['SENTENCE_TRANSFORMERS_HOME']
    )

    # ── Retriever 생성 ──
    retriever = ResumeBM25Retriever(
        db_config=DB_CONFIG,
        bm25_index=bm25_index,
        embeddings=hf_embeddings,
        top_n=5,
        initial_k=50,
        index_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "faiss_index"),
        w_embedding=0.6,
        w_bm25=0.4,
    )

    # ── 전체 직무 검색 품질 검증 ──
    test_cases = [
        # ai engineer
        "ai engineer||LangChain과 LangGraph를 활용한 프로젝트 경험이 있으며 Python을 주로 사용합니다.",
        # "ai engineer||RAG 시스템을 직접 구축한 경험이 있고 OpenAI API와 벡터DB를 활용해봤습니다.",
        # # backend engineer
        # "backend engineer||Spring Boot와 MySQL을 활용한 RESTful API 개발 경험이 있습니다.",
        # "backend engineer||Node.js와 PostgreSQL로 서버 개발 경험이 있으며 Docker도 사용해봤습니다.",
        # # frontend engineer
        # "frontend engineer||React와 TypeScript로 웹 애플리케이션을 개발한 경험이 있습니다.",
        # "frontend engineer||Vue.js와 Vuex를 활용한 SPA 개발 경험이 있습니다.",
    ]

    print("\n" + "=" * 60)
    print("전체 직무 검색 품질 검증")
    print("=" * 60)

    for test_query in test_cases:
        _, query = test_query.split("||", 1)
        print(f"\n{'─'*60}")
        print(f"쿼리: {query}")
        print(f"{'─'*60}")

        start = time.time()        
        results = retriever.invoke(test_query)
        elapsed = time.time() - start 
        print(f"  검색 시간: {elapsed:.2f}초")  

        if not results:
            print("  검색 결과 없음")
            continue

        for i, doc in enumerate(results, 1):
            print(f"\n  {i}번째 결과")
            print(f"    hybrid_score : {doc.metadata['hybrid_score']}")
            print(f"    bm25_score   : {doc.metadata['bm25_score']}")
            print(f"    자소서 앞 200자: {doc.page_content[:200]}")