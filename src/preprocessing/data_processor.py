import re
import pandas as pd

class DataProcessor:
    def __init__(self, cleaner):
        self._cleaner = cleaner
        self._is_cleaner_fitted = False

    #=====# 1===========jobpost==============
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
        career = str(career)
        career = career.strip()
        if any(keyword in career for keyword in ['신입']):
            return 'junior'
        elif any(keyword in career for keyword in ['경력', '경력직']):
            return 'senior'
        return None

    def _extract_fields_from_jobpost(self, jobpost) -> pd.DataFrame:
        company = re.search(r"\*\*기업명\*\*:\s*\[(.*?)\]", jobpost)
        career_type = re.search(r"\*\*신입/경력\*\*:\s*\[(.*?)\]", jobpost)
        description = re.search(r"\*\*소개\*\*:\s*(.*?)(?=\n\n|$)", jobpost, re.DOTALL)
        position = re.search(r"\*\*포지션명\*\*:\s*\[(.*?)\]", jobpost)

        res =  {
            "company": company.group(1).strip() if company else None,
            "description": description.group(1).strip() if description else None,
            "position_type": position.group(1).strip() if position else None,
            "career_type": career_type.group(1).strip() if career_type else None,
            "responsibilities": "",
            "qualifications": "",
            "preferred": "",
        }

        lines = jobpost.split('\n')
        current_section = None
        section_map = {
            "주요업무": "responsibilities",
            "자격요건": "qualifications",
            "우대사항": "preferred"
        }

        for line in lines:
            line = line.strip()
            if not line: continue

            found_header = False
            for kor_name, eng_name in section_map.items():
                if kor_name in line:
                    current_section = eng_name
                    found_header = True
                    break

            if found_header:
                continue

            if current_section and line.startswith('-'):
                clean_item = line.lstrip('- ').strip()
                res[current_section] += clean_item
                res[current_section] += '\n'
        return res    

    def _extract_jobpost(self, jobposts) -> pd.DataFrame:
        refined_jobposts = [self._extract_fields_from_jobpost(jobpost) for jobpost in jobposts]
        df = pd.DataFrame(refined_jobposts)
        if not self._is_cleaner_fitted:
            self._cleaner.fit(df['company'])
            self._is_cleaner_fitted = True
        df.loc[:, 'company'] = df['company'].apply(self._cleaner.clean)
        df.loc[:, 'position_type'] = df['position_type'].apply(self._normalize_job_title)
        df.loc[:, 'career_type'] = df['career_type'].apply(self._normalize_career_type)
        
        return df

    #=====# 2===========selfintro grade==============
    def _extract_grade(self, df: pd.DataFrame, grades) -> pd.DataFrame:
        df['selfintro_grade'] = grades
        df['selfintro_grade'] = df['selfintro_grade'].map({'상':'high','중':'mid','하':'low'})
        return df

    #=====# 3===========refined resume==============
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
        df = df.rename(columns={'resume': 'resume_cleaned'}) #컬럼명 변경 주
        return df

    #=====# 4===========selfintro==============
    def _extract_selfintro(self, df, selfintros):
        df['selfintro'] = selfintros
        return df

    #=====# 5===========selfintro evaluation==============
    def _refine_selfintro_eval(self, evaluation):
        if not evaluation or not isinstance(evaluation, str):
            return ""

        # <eval_selfintro> 태그 사이의 내용을 추출 (DotAll 옵션으로 줄바꿈 포함)
        pattern = r"<eval_selfintro>(.*?)</eval_selfintro>"
        match = re.search(pattern, evaluation, re.DOTALL)

        if match:
            # 추출된 내용의 앞뒤 불필요한 공백/줄바꿈 제거
            return match.group(1).strip()
        return ""

    def _extract_selfintro_evaluation(self, df, evaluations):
        df['selfintro_evaluation'] = evaluations
        df['selfintro_evaluation'] = df['selfintro_evaluation'].apply(self._refine_selfintro_eval)
        return df

    #=====# 6===========selfintro score==============
    def _extract_selfintro_score(self, df, seltintro_scores):
        df['selfintro_score'] = seltintro_scores
        return df

    def run_preprocess_pipeline(self, raw_datset):
        """전처리 파이프라인 통합 실행"""

        data_df = self._extract_jobpost(raw_datset['jobpost'])
        data_df = self._extract_resume(data_df, raw_datset['resume'])
        data_df = self._extract_selfintro(data_df, raw_datset['selfintro'])
        data_df = self._extract_selfintro_evaluation(data_df, raw_datset['evaluation'])
        data_df = self._extract_selfintro_score(data_df, raw_datset['selfintro_score'])
        data_df = self._extract_grade(data_df, raw_datset['selfintro_grade'])
        data_df = data_df.dropna(how = 'any')
        data_df = data_df[data_df['selfintro_grade'] == 'high'] # high 등급만 적재  
        
        return data_df

