# MySQL-FAISS Hybrid Retriever

MySQL 9.x의 `VECTOR` 타입과 in-memory `FAISS`를 사용해 vector search를 구현해보려고 합니다.

## DB Schema 1.0(Vertical Partitioning)
벡터 검색이라는 목적에 맞게 테이블을 2개로 분리하여 관리합니다.
- `application_records`: 정제된 이력서, 자소서 원문, 직무, 등급 등 메타데이터 저장
- `application_vectors`: 이력서 벡터 임베딩과 `application_records`의 `id`를 foreign key로 하는 벡터 전용 테이블

## Records

| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **id** | BIGINT | PRIMARY KEY, AUTO_INC | 고유 식별자 |
| **career_type** | ENUM | NOT NULL | 경력 구분 ('junior', 'senior') |
| **position_type** | ENUM | NOT NULL | 직무 분야 ('frontend engineer', 'backend engineer', 'ai engineer') |
| **selfintro** | TEXT | NOT NULL | 자기소개서 원문 |
| **resume_cleaned** | TEXT | NOT NULL | 전처리된 이력서 텍스트 (임베딩 대상) |
| **grade** | ENUM | NOT NULL, **INDEX** | 평가 등급 ('high', 'mid', 'low') |
| **created_at** | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

### Vectors
| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **record_id** | BIGINT | PRIMARY KEY, **FK** | `application_records.id` 참조 |
| **embedding** | VECTOR(1024) | NOT NULL | 이력서 텍스트의 벡터 표현 값 (1024차원) |

## Key Components
| **클래스 명** | **역할** | **주요 특징** |
| --- | --- | --- |
| **`config.py`** | 환경 변수 관리 | DB 접속 및 api key 등 통합 관리|
| **`DataProcessor`** | 전처리 파이프라인 | 섹션 파싱, 컬럼 정규화, 결측치 처리|
| **`DBSampleLoader`** | 벡터 임베딩 생성 및 DB 적재 | `mysql-connector` 트랜잭션을 활용해 배치 단위로 쿼리 실행 |
| **`MySQLFaissRetriever`** | 벡터 검색 | `LangChain` 기반 인터페이스, 코사인 유사도로 `N`개를 먼저 추출한 뒤 자소서 등급이 높은 `k`개 추출 |
