# MySQL-FAISS Hybrid Retriever

MySQL 9.x의 `VECTOR` 타입과 in-memory `FAISS`를 사용해 vector search를 구현해보려고 합니다.

## DB Schema 2.0
정규화를 통한 테이블 세분화
- `companies`: 회사 정보 저장.
- `jobposts`:  공고 정보(직무 분야, 경력 여부, 자격사항 등) 저장. `companies`의 `id`를 foreign key로 가짐. 
- `applicant_records`: 지원자 제출 서류 정보(정제된 이력서, 자소서, 평가 의견 등) 저장. `jobposts`의 `id`를 foreign key로 가짐.
- `resume_vectors`: 이력서 벡터 임베딩 저장. `application_records`의 `id`를 foreign key로 가지는 벡터 전용 테이블

### 테이블별 상세 정보

- companies
 
| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **id** | INT | PRIMARY KEY | 고유 식별자 |
| **name** | VARCHAR | NOT NULL | 회사명 |
| **updated_at** | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 마지막 수정 일시 |

- jobposts

| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **id** | INT | PRIMARY KEY | 고유 식별자 |
| **company_id** | INT | FOREIGN KEY | `companies.id` 참조 |
| **description** | TEXT | - | 기업 소개 |
| **position_type** | ENUM | NOT NULL | 직무 분야 ('frontend engineer', 'backend engineer', 'ai engineer') |
| **career_type** | ENUM | NOT NULL | 경력 구분 ('junior', 'senior') |
| **responsibilities** | TEXT | NOT NULL | 주요 업무 내용 |
| **qualifications** | TEXT | NOT NULL | 자격 요건 |
| **preferred** | TEXT | - | 우대 사항 |
| **updated_at** | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 마지막 수정 일시 |

- **applicant_records**

| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **id** | INT | PRIMARY KEY, AUTO_INC | 고유 식별자 |
| **jobpost_id** | INT | FOREIGN KEY | `jobposts.id` 참조 |
| **resume_cleaned** | TEXT | NOT NULL | 전처리된 이력서 텍스트 (임베딩 대상) |
| **selfintro** | TEXT | NOT NULL | 자기소개서 원문 |
| **selfintro_evaluation** | TEXT | NOT NULL | 자기소개서 평가 의견 |
| **selfintro_score** | INT | NOT NULL | 자기소개서 평가 점 |
| **grade** | ENUM | NOT NULL | 자소서 평가 등급 ('high', 'mid', 'low') |
| **created_at** | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | 생성 일시 |

- **resume_vectors**

| **컬럼명** | **타입** | **제약 조건** | **설명** |
| --- | --- | --- | --- |
| **record_id** | BIGINT | PRIMARY KEY, **FK** | `application_records.id` 참조 |
| **embedding** | VECTOR(1024) | NOT NULL | 이력서 텍스트의 벡터 표현 값 (1024차원) |

## Key Components
| **클래스(함수) 명** | **역할** | **주요 특징** |
| --- | --- | --- |
| **`get_dataset`** | 데이터셋 로드 | Huggingface에서 데이터셋 불러오기|
| **`CompanyNameCleaner`** | 회사 이름 전처리 | 예외 리스트(`cleaner_config`) 등을 고려해 정규화 수행|
| **`DataProcessor`** | 데이터셋 전처리 | feature extraction & normalization |
| **`JobPocketBulkLoader`** | DB 적재 | `pymysql`을 활용해 배치 단위로 쿼리 실행 |
| **`JobPocketPipeline`** | DB 적재 데이터 생성, 파이프라인 | 각 테이블의 id, 벡터 임베딩 등 생성, 순서에 맞게 `JobPocketBulkLoader` 메서드 실행|
