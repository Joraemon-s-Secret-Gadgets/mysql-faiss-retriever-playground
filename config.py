import os
import sys
import platform
from dotenv import load_dotenv

# .env 파일 로드(로컬 환경)
load_dotenv()

# 실행 환경 감지
def check_colab() -> bool:
    return "google.colab" in sys.modules
def is_linux() -> bool:
    return platform.system() == "Linux"

def get_env(key: str, default: str = None) -> str:
    """
    Colab Secrets 혹은 OS 환경 변수를 불러옵니다.
    """
    if check_colab():
        from google.colab import userdata
        try:
            return userdata.get(key)
        except:
            return os.getenv(key, default)
    return os.getenv(key, default)

def get_ssl_ca_path() -> str:
    """
    환경별 표준 SSL CA 인증서 경로를 반환합니다.
    """
    # Colab, RunPod, WSL2(Ubuntu)는 대부분 이 경로를 공유합니다.
    if is_linux():
        standard_path = "/etc/ssl/certs/ca-certificates.crt"
        if os.path.exists(standard_path):
            return standard_path
    
    # 윈도우 네이티브 환경이거나 커스텀 경로가 필요한 경우 .env 활용
    return os.getenv("SSL_CA_PATH", "")

# DB 접속 정보
def get_db_config() -> dict:
    return  {
        "host": get_env("TIDB_HOST"),
        "user": get_env("TIDB_USER"),
        "password": get_env("TIDB_PW"),
        "database": get_env("TIDB_DB", "test"),
        "port": int(get_env("TIDB_PORT")),
        "ssl_ca": get_ssl_ca_path(),
        "ssl_verify_cert": True,
        "use_pure": True
}

# AI 모델 및 API 설정
def get_hf_token():
    return get_env("HF_TOKEN")

