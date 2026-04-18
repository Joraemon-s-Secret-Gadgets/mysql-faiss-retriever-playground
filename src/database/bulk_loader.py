import pymysql
import pandas as pd
from typing import List
import json

class JobPocketBulkLoader:
    def __init__(self, **db_config):
        self.conn = pymysql.connect(
            **db_config,
            cursorclass=pymysql.cursors.DictCursor
        )

    def __del__(self):
        if self.conn: self.conn.close()

    def _bulk_insert(self, sql: str, data: List[tuple], table_name: str):
        self.conn.ping(reconnect=True)
        with self.conn.cursor() as cursor:
            try:
                cursor.executemany(sql, data)
                self.conn.commit()
                print(f"✅ {table_name}: {cursor.rowcount}개 행 적재 완료")
            except Exception as e:
                self.conn.rollback()
                print(f"❌ {table_name} 적재 실패: {e}")
                raise e

    def upload_companies(self, df: pd.DataFrame):
        companies = df[['company_id', 'company']].drop_duplicates()
        data = list(companies.itertuples(index=False, name=None))
        sql = "INSERT IGNORE INTO companies (id, name) VALUES (%s, %s)"
        self._bulk_insert(sql, data, "Companies")

    def upload_jobposts(self, df: pd.DataFrame):
        jobposts = df[[
            'jobpost_id', 'company_id', 'career_type', 'position_type',
            'responsibilities', 'qualifications', 'preferred', 'description'
        ]].drop_duplicates()
        data = list(jobposts.itertuples(index=False, name=None))
        sql = """
            INSERT IGNORE INTO job_posts
            (id, company_id, career_type, position_type, responsibilities, qualifications, preferred, description)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        self._bulk_insert(sql, data, "JobPosts")

    def upload_applicants_and_vectors(self, df: pd.DataFrame):
        # 1. 지원 기록 준비
        applicants = df[[
            'applicant_id', 'jobpost_id', 'resume_cleaned', 'selfintro',
            'selfintro_evaluation', 'selfintro_score', 'selfintro_grade'
        ]]
        app_data = list(applicants.itertuples(index=False, name=None)) #원래 applicant_tables임
        app_sql = """
            INSERT IGNORE INTO applicant_records
            (id, jobpost_id, resume_cleaned, selfintro, selfintro_evaluation, selfintro_score, grade)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        self._bulk_insert(app_sql, app_data, "Applicants Records")

        # 2. 벡터 데이터 준비 (Pipeline이 실시간으로 만든 embedding 컬럼 사용)
        resume_vecs = [(row.applicant_id, json.dumps(row.resume_embedding)) for row in df.itertuples()]
        #intro_vecs = [(row.applicant_id, str(row.selfintro_embedding)) for row in df.itertuples()]
        self._bulk_insert("INSERT IGNORE INTO resume_vectors (record_id, embedding) VALUES (%s, STRING_TO_VECTOR(%s))", resume_vecs, "Resume Vectors")
        # self._bulk_insert("INSERT IGNORE INTO selfintro_vectors (id, embedding) VALUES (%s, %s)", intro_vecs, "Self-Intro Vectors")