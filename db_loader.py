import os
import json
import pandas as pd
import mysql.connector
from tqdm import tqdm
from langsmith import traceable # LangSmith 추적용
from langchain_huggingface import HuggingFaceEmbeddings # LangChain 임베딩

from config import get_db_config



class DBLoader:
    def __init__(self, db_config, model_name="Qwen/Qwen3-Embedding-0.6B", batch_size=50):
        self.db_config = db_config # DB 접속정보
        self.batch_size = batch_size
        
        # 임베딩 모델
        self.embeddings = HuggingFaceEmbeddings( 
            model_name=model_name,
            model_kwargs={'device': 'cuda'}, #GPU 사용
            encode_kwargs={"normalize_embeddings": True} #저장할 때 미리 정규화
        )

    def prepare_samples(self, df, N):
        """데이터 로드 및 (포지션, 자소서 등급)별 N건 샘플링"""
        samples = df.groupby(['position_type', 'grade']).sample(n=N, random_state=42).reset_index(drop=True)
        print(f"✅ 샘플링 완료: 총 {len(samples)}건")
        return samples

    @traceable(name="Embedding_DB_Pipeline_batch_insert", process_inputs=lambda x: {}) # LangSmith에 기록될 수 있도록 설정
    def run_pipeline(self, df):
        """벡터 임베딩 계산 및 DB 적재"""
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()

        try:
            for i in tqdm(range(0, len(df), self.batch_size), desc="Batch Processing"):
                batch = df.iloc[i : i + self.batch_size] #배치 단위 데이터 로드

                texts = batch['resume_cleaned'].tolist() #임베딩할 정제된 이력서 데이터
                batch_embeddings = self.embeddings.embed_documents(texts) # 임베딩 생성

                # 메타데이터 적재
                record_values = [
                    (row['career_type'], row['position_type'], row['selfintro'], row['resume_cleaned'], row['grade'])
                    for _, row in batch.iterrows()
                ]
                sql_rec = """
                    INSERT INTO application_records (career_type, position_type, selfintro, resume_cleaned, grade)
                    VALUES (%s, %s, %s, %s, %s)
                """ 
                cursor.executemany(sql_rec, record_values)

                # DB 적재된 메타데이터의 제일 첫번째 index
                first_id = cursor.lastrowid
                row_count = cursor.rowcount

                vector_values = [
                    (first_id + idx, json.dumps(batch_embeddings[idx])) #'[0.1, 0.2, ...]' 와 같은 형식으로 변환
                    for idx in range(row_count)
                ]

                sql_vec = "INSERT INTO application_vectors (record_id, embedding) VALUES (%s, %s)" #알아서 vector type으로 잘 변환됨
                cursor.executemany(sql_vec, vector_values)

                conn.commit()
                    
            print(f"\n✨ 적재 완료!")
            return {'status':"success"}
        except Exception as e:
          conn.rollback()
          print(f"❌ Batch Error: {e}")
          raise e            
        finally:
            cursor.close()
            conn.close()

# --- 실행 예시 ---
if __name__ == "__main__":
    from datasets import load_dataset #local 실행시 datasets 설치 필요
    from data_processor import DataProcessor
    train_data = load_dataset("Youseff1987/resume-matching-dataset-v2", split='train')
    processor = DataProcessor()
    result = processor.run_preprocess_pipeline(train_data)
    loader = DBLoader(get_db_config(), batch_size = 16)
    test_samples = loader.prepare_samples(result, 200)
    loader.run_pipeline(test_samples)