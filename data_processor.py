import re
import pandas as pd

class DataProcessor:
  def _normalize_job_title(self, title: str) -> str:
    if title is None:
        return None
    title = str(title)
    title = title.lower()

    if any(keyword in title for keyword in ['ai', 'llm']):
        return 'ai engineer'

    elif any(keyword in title for keyword in ['백엔드']):
        return 'backend engineer'

    elif any(keyword in title for keyword in ['프론트엔드']):
        return 'frontend engineer'

    return 'others'

  def _normalize_career_type(self, career: str) -> str:
      if career is None:
        return None
      career = career.strip()
      if any(keyword in career for keyword in ['신입']):
          return 'junior'
      elif any(keyword in career for keyword in ['경력', '경력직']):
          return 'senior'
      return 'other'

  def _extract_fields_from_jobpost(self, jobpost) -> pd.DataFrame:
    #company = re.search(r"\*\*기업명\*\*:\s*\[(.*?)\]", text)
    career_type = re.search(r"\*\*신입/경력\*\*:\s*\[(.*?)\]", jobpost)
    position = re.search(r"\*\*포지션명\*\*:\s*\[(.*?)\]", jobpost)
    return {
        "career_type": career_type.group(1) if career_type else None,
        "position_type": position.group(1) if position else None,
    }

  def _extract_jobpost(self, jobposts) -> pd.DataFrame:
    position_and_careers = [self._extract_fields_from_jobpost(jobpost) for jobpost in jobposts]
    df = pd.DataFrame(position_and_careers)
    df['career_type'] = df['career_type'].apply(self._normalize_career_type)
    df['position_type'] = df['position_type'].apply(self._normalize_job_title)
    return df

  def _extract_grade(self, df: pd.DataFrame, grades) -> pd.DataFrame:
    df['grade'] = grades
    df['grade'] = df['grade'].map({'상':'high','중':'mid','하':'low'})
    return df

  def _refine_resume(self, resume: str) -> str:
    edu_match = re.search(r"\*\*학력:\*\*", resume)
    exp_match = re.search(r"\*\*경력 및 경험:\*\*", resume)

    if not edu_match or not exp_match: # 형식에 맞지 않는경우
        return resume

    # 학력 섹션의 마지막 항목(최종학력) 확보
    edu_section = resume[edu_match.end():exp_match.start()].strip()
    edu_lines = [line.strip("- ").strip() for line in edu_section.split('\n') if line.strip()]
    final_edu = edu_lines[-1] if edu_lines else ""

    # 경력 및 경험 섹션부터 끝까지 추출
    experience_and_beyond = resume[exp_match.start():].strip()

    # 추가 정제
    summary = f"[최종학력] {final_edu}\n{experience_and_beyond}"
    refined_summary = re.sub(r"\*\*(.*?):\*\*", r"[\1]", summary)
    refined_summary = re.sub(r'\n{2,}', '\n', refined_summary)
    return refined_summary.strip()

  def _extract_resume(self, df, resumes):
    df['resume'] = resumes
    df['resume'] = df['resume'].apply(self._refine_resume)
    df = df.rename(columns={'resume': 'resume_cleaned'})
    return df

  def _extract_selfintro(self, df, selfintros):
    df['selfintro'] = selfintros
    return df

  def run_preprocess_pipeline(self, raw_datset):
    """전처리 파이프라인 통합 실행"""
    
    data_df = self._extract_jobpost(raw_datset['jobpost'])
    data_df = self._extract_grade(data_df, raw_datset['selfintro_grade'])
    data_df = self._extract_resume(data_df, raw_datset['resume'])
    data_df = self._extract_selfintro(data_df, raw_datset['selfintro'])
    data_df = data_df.dropna(how = 'any')
    return data_df

# --- 사용 예시 ---
if __name__ == "__main__":
    from datasets import load_dataset #local 실행시 datasets 설치 필요
    train_data = load_dataset("Youseff1987/resume-matching-dataset-v2", split='train')
    processor = DataProcessor()
    result = processor.run_preprocess_pipeline(train_data)
    print(len(result))