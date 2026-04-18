import os
from dotenv import load_dotenv

# .env 파일 로드(로컬 환경)
load_dotenv()

def get_env(key: str, default: str = None) -> str:
    """
    OS 환경 변수를 불러옵니다.
    """
    return os.getenv(key, default)

# DB 접속 정보
def get_mysql_db_config() -> dict:
    """
    MySQL DB 접속 정보를 반환합니다.
    """
    return  {
    'host': get_env('HOST'),
    'port': int(get_env('PORT')),
    'user': get_env('USER'),
    'password': get_env('PASSWORD'),
    'db': get_env('DB'),
    # "ssl_ca": "./CERTIFICATE_NAME.crt", #only for TiDB
    #"ssl_verify_cert": True, #only for TiDB
    'charset': 'utf8mb4'
    }
    
# AI 모델 및 API 설정
def get_hf_token() -> str:
    return get_env("HF_TOKEN")

