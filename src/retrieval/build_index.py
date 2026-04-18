import pymysql
import json
import numpy as np
import os
from tqdm import tqdm

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndexBuilder:
    def __init__(self, db_config, model_name="Qwen/Qwen3-Embedding-0.6B", save_path="faiss_index"):
        self.db_config = db_config
        self.save_path = save_path
        # 임베딩 모델 초기화 (GPU 사용)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={"normalize_embeddings": True}
        )

    def build_and_save(self):
        """DB에서 데이터를 가져와 FAISS 인덱스를 생성하고 파일로 저장"""
        # 1. pymysql 연결 (DictCursor 사용)
        conn = pymysql.connect(**self.db_config, cursorclass=pymysql.cursors.DictCursor)
        
        try:
            with conn.cursor() as cursor:
                print("🔍 [1/4] DB에서 벡터 데이터 로드 중...")
                # TiDB 혹은 VECTOR 타입을 지원하는 환경에 맞춘 SQL
                sql = """
                    SELECT r.id, r.grade, VECTOR_TO_STRING(v.embedding) as embedding_str
                    FROM applicant_records r
                    JOIN resume_vectors v ON r.id = v.record_id
                """
                cursor.execute(sql)
                rows = cursor.fetchall()

                if not rows:
                    print("⚠️ 적재된 데이터가 없습니다.")
                    return

                # 2. 데이터 가공 (메타데이터 및 벡터 분리)
                print(f"⚙️ [2/4] {len(rows)}건 데이터 가공 및 넘파이 변환 중...")
                metadatas = []
                vectors = []
                ids = []

                for r in tqdm(rows, desc="Parsing"):
                    metadatas.append({
                        "selfintro_score": r['selfintro_score']
                    })
                    ids.append(str(r['id']))
                    
                    # JSON 파싱 (실패 시 0 벡터 채움)
                    try:
                        vec = json.loads(r['embedding_str']) if r['embedding_str'] else [0.0]*1024
                    except json.JSONDecodeError:
                        vec = [0.0]*1024
                    vectors.append(vec)

                # float32 변환
                vectors_np = np.array(vectors).astype('float32')

                # 3. FAISS 인덱스 빌드
                print("🏗️ [3/4] FAISS 인덱스 생성 중 (Inner Product)...")
                # (텍스트, 임베딩) 쌍 생성 - 텍스트 자리에 ID를 넣어 나중에 역추적 가능하게 함
                text_embeddings = list(zip(ids, vectors_np)) # Pointer-to-index 기법

                vectorstore = FAISS.from_embeddings(
                    text_embeddings=text_embeddings,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                )

                # 4. 로컬 저장
                print(f"💾 [4/4] 인덱스 파일 저장 중... (경로: {self.save_path})")
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                
                vectorstore.save_local(self.save_path)
                print(f"✨ 성공! {len(rows)}건의 인덱스가 '{self.save_path}'에 저장되었습니다.")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            raise e
        finally:
            conn.close()

# --- 실행 예시 ---
if __name__ == "__main__":
    from src.config import get_mysql_db_config # 설정 함수가 있다고 가정
    
    config = get_mysql_db_config()
    builder = FAISSIndexBuilder(db_config=config, save_path="./faiss_idx")
    builder.build_and_save()