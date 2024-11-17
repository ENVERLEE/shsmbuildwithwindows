#reference_service.py
from typing import List, Dict
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import Reference
from .mlx_service import MLXService
import re
from .perplexity_service import PerplexityService
from .research_service import ResearchService

mlx_service = MLXService()

class ReferenceService:
    def __init__(self,
                db: Session,
                perplexity_service: PerplexityService,
                mlx_service: MLXService):
        self.db = db
        self.perplexity = perplexity_service
        self.mlx = mlx_service
    
    def search_and_save_references(self,
                                     project_id: int,
                                     keywords: List[str]) -> List[Reference]:
        try:
            # Perplexity API로 참고문헌 검색
            query = " AND ".join(keywords)
            references = self.perplexity.search_references(query)
            
            # 검색된 참고문헌 저장
            saved_references = []
            for ref_data in references:
                try:
                    # 날짜 파싱 에러 처리
                    try:
                        pub_date = datetime.fromisoformat(ref_data["publication_date"].replace('Z', '+00:00'))
                    except (ValueError, TypeError):
                        pub_date = None

                    # 임베딩 생성
                    content = ref_data.get("abstract", "")
                    if content:
                        embedding = self.mlx.analyze_content(content)
                        embedding_vector = embedding.get("embeddings", [])
                    else:
                        embedding_vector = []

                    reference = Reference(
                        project_id=project_id,
                        title=ref_data.get("title", ""),
                        authors=ref_data.get("authors", []),
                        publication_date=pub_date,
                        content=content,
                        metadata1={
                            "journal": ref_data.get("journal"),
                            "doi": ref_data.get("doi"),
                            "url": ref_data.get("url"),
                            "citations": ref_data.get("citations", 0)
                        },
                        embedding=embedding_vector
                    )
                    
                    self.db.add(reference)
                    saved_references.append(reference)
                except Exception as e:
                    self.logger.error(f"Error saving reference: {str(e)}")
                    continue
            
            self.db.commit()
            return saved_references

        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Reference search and save failed: {str(e)}")
            raise
   
    def get_reference_summary(self, reference_id: int) -> Dict:
        reference = self.db.query(Reference).get(reference_id)
        if not reference:
            raise ValueError("참고문헌을 찾을 수 없습니다.")
       
       # Cohere를 사용하여 요약 생성
        analysis = self.cohere.analyze_content(reference.content)
       
        return {
           "title": reference.title,
           "authors": reference.authors,
           "summary": analysis["analysis"],
           "metadata": reference.metadata
        }
