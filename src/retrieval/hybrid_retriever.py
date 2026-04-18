import pymysql
from pymysql import Error
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langsmith import traceable # LangSmith 추적용
from pydantic import ConfigDict, PrivateAttr

class HybridRetriever(BaseRetriever):
    embeddings: any
    top_n: int = 3
    initial_k: int = 50
    vectorstore: FAISS = None
    index_folder: str = "faiss_index"
    
    _conn = PrivateAttr()
        
    def __init__(self, db_config, **kwargs):
        super().__init__(**kwargs)
        self._conn = pymysql.connect(
            **db_config,
            cursorclass=pymysql.cursors.DictCursor
        )
        self._get_vector_index()
    def __del__(self):
        if self._conn and self._conn.open: self._conn.close()

    model_config = ConfigDict(arbitrary_types_allowed=True, extra = "allow")

    def _get_vector_index(self):
        """서버 기동 시 빌드된 FAISS index를 불러옴"""
        self.vectorstore = FAISS.load_local(
            folder_path = self.index_folder,
            embeddings = self.embeddings,
            allow_dangerous_deserialization=True
        )

    @traceable(name="Vector Search", process_inputs=lambda x: {}, process_outputs=lambda x: {})
    def _get_relevant_documents(self, query: str) -> List[Document]:
        if self.vectorstore is None: self._get_vector_index()

        # 1. FAISS 유사도 검색 (상위 initial_k개 후보 추출)
        # LangChain의 similarity_search_with_score는 내부적으로 쿼리 벡터를 정규화하여 검색함
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=self.initial_k)
        
        # 이력서 유사도 점수와 자소서 평가 점수를 하나의 map으로 관리
        score_map = {int(doc.page_content): (doc.metadata.get('selfintro_score', 0) , float(score)) for doc, score in docs_and_scores}
        
        # 2. Peer-First (Grade 기반) 필터링: 상 -> 중 순서로 정렬
        # 우선 '상' 등급만 적재했으므로 제외
        candidates = [doc for doc,_ in docs_and_scores]
        #high_grade = [d for d in candidates if d.metadata['grade'] == 'high']
        #mid_grade = [d for d in candidates if d.metadata['grade'] == 'mid']
        
        # 최종적으로 우리가 원하는 개수(top_n)만큼 ID 선별
        #final_candidate_docs = (high_grade + mid_grade)[:self.top_n]
        final_candidate_docs = candidates[:self.top_n]
        target_db_ids = [int(d.page_content) for d in final_candidate_docs]
        # 3. MySQL에서 '진짜 자소서 본문' 페치
        return self._fetch_final_documents(target_db_ids, score_map)

    def _fetch_final_documents(self, db_ids: List[int], score_map: Dict[int, float]) -> List[Document]:
        if not db_ids: return []
        
        self._conn.ping(reconnect=True)
        cursor = self._conn.cursor()

        try:
            format_strings = ','.join(['%s'] * len(db_ids))
            # content_json(원본)과 grade를 가져옴
            sql = f"""
            SELECT id, selfintro
            FROM applicant_records
            WHERE id IN ({format_strings})
            """
            cursor.execute(sql, tuple(db_ids))
            rows = cursor.fetchall()

            id_map = {r['id']: r for r in rows}

            # 원래 유사도 순서(db_ids)를 유지하며 Document 객체 생성
            final_docs = []
            for db_id in db_ids:
                if db_id in id_map:
                    record = id_map[db_id]
                    
                    # Document 객체 생성
                    doc = Document(
                        # 실제 검색에 사용될 메인 텍스트 (자소서)
                        page_content=record['selfintro'], 
                        metadata={
                            "id": db_id,
                            "selfintro_score": score_map.get(db_id)[0],      # 자소서 평가 점수 (최대 60점) 
                            #"eval": record['selfintro_evaluation'], # 평가 의견
                            "relevance_score": score_map.get(db_id)[1]  # 리트리버가 계산한 유사도 점수
                        }
                    )
                    final_docs.append(doc)
                    
            return final_docs
        except Error as e:
            print(f"❌ MySQL 에러: {e}")
            return []
        finally:
            cursor.close()
            self._conn.close()
            