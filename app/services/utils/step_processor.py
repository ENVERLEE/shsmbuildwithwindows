from typing import List, Dict
import logging

mlx_service = MLXService()

class StepProcessor:
    def __init__(self, db, mlx_service, perplexity_service):
        self.db = db
        self.cohere = mlx_service
        self.perplexity = perplexity_service
        self.logger = logging.getLogger(__name__)
    
    def process_step(self, step, references: List[Dict] = None) -> Dict:
        '''단일 연구 단계 처리'''
        try:
            step.progress_percentage = 10
            self.db.commit()
            
            # 참고문헌 검색 및 분석
            if not references:
                step.progress_percentage = 20
                self.db.commit()
                references = self.perplexity.search_references(" ".join(step.keywords_list))
            
            step.progress_percentage = 40
            self.db.commit()
            
            # 단계별 분석 수행
            analysis_prompt = self._create_analysis_prompt(step, references)
            step.progress_percentage = 60
            self.db.commit()
            
            analysis = self.cohere.analyze_content(analysis_prompt)
            step.progress_percentage = 80
            self.db.commit()
            
            # 결과 정리
            result = self._format_step_result(analysis, references)
            step.progress_percentage = 100
            self.db.commit()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step.step_number} processing failed: {e}")
            raise
    
    def _create_analysis_prompt(self, step, references: List[Dict]) -> str:
        '''분석을 위한 프롬프트 생성'''
        reference_text = "\n".join([
            f"제목: {ref['title']}\n저자: {ref['authors']}\n요약: {ref['abstract']}\n"
            for ref in references
        ])
        
        return f'''
        당신은 한국의 안보 전문 연구원입니다. 다음 연구 단계를 수행해주세요:

        [연구 단계 정보]
        단계: {step.step_number}
        설명: {step.description}
        방법론: {step.methodology}

        [참고문헌]
        {reference_text}

        다음 형식으로 분석 결과를 제시해주세요:
        1. 핵심 발견사항
        2. 세부 분석
        3. 시사점
        4. 다음 단계를 위한 제언
        '''
    
    def _format_step_result(self, analysis: Dict, references: List[Dict]) -> Dict:
        '''단계 결과 포맷팅'''
        return {
            "analysis_result": analysis,
            "used_references": references,
            "keywords_used": analysis.get("main_topics", []),
            "confidence_score": analysis.get("confidence", 0)
        }

