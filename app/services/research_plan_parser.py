from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import torch
from typing import List, Dict, Tuple
import re

class ResearchPlanParser:
    def __init__(self):
        """연구 계획 파서 초기화"""
        # 섹션 분류를 위한 BERT 모델
        self.section_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.section_model = AutoModelForSequenceClassification.from_pretrained(
            "klue/bert-base",
            num_labels=4  # description, keywords, methodology, output_format
        )
        
        # 키워드 추출을 위한 NER 모델
        self.ner_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            "klue/bert-base",
            num_labels=3  # B-KW, I-KW, O
        )
        
        # MPS 가속 설정
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.section_model.to(self.device)
        self.ner_model.to(self.device)
        
        # 레이블 매핑
        self.section_labels = {
            0: "description",
            1: "keywords",
            2: "methodology",
            3: "output_format"
        }
        
        # 상태 유지
        self.current_step = 1

def _parse_research_plan(self, plan_text: str) -> List[Dict]:
    """연구 계획 텍스트를 구조화된 형식으로 파싱"""
    try:
        # 텍스트 전처리
        lines = [line.strip() for line in plan_text.split('\n') if line.strip()]
        steps = []
        current_step = None
        
        for line in lines:
            # 새로운 단계 시작 감지
            if self._is_new_step(line):
                if current_step:
                    steps.append(current_step)
                current_step = self._initialize_new_step()
                continue
            
            if not current_step:
                continue
                
            # 라인 분류 및 처리
            section_type = self._classify_section(line)
            if section_type == "keywords":
                keywords = self._extract_keywords(line)
                if keywords:
                    current_step["keywords"].extend(keywords)
            else:
                processed_content = self._process_section_content(line)
                if processed_content:
                    current_step[section_type] = processed_content
        
        # 마지막 단계 추가
        if current_step:
            steps.append(current_step)
        
        return self._post_process_steps(steps)
        
    except Exception as e:
        raise Exception(f"Research plan parsing failed: {str(e)}")

def _is_new_step(self, line: str) -> bool:
    """새로운 단계의 시작인지 확인"""
    step_patterns = [
        r"##?\s*\d+\s*단계",
        r"단계\s*\d+",
        r"step\s*\d+",
        r"\d+\.\s*단계"
    ]
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in step_patterns)

def _initialize_new_step(self) -> Dict:
    """새로운 단계 딕셔너리 초기화"""
    step = {
        "step_number": self.current_step,
        "description": "",
        "keywords": [],
        "methodology": "",
        "output_format": ""
    }
    self.current_step += 1
    return step

def _classify_section(self, text: str) -> str:
    """텍스트의 섹션 유형 분류"""
    # 토큰화
    inputs = self.section_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(self.device)
    
    # 모델 추론
    with torch.no_grad():
        outputs = self.section_model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(predictions, dim=1).item()
    
    return self.section_labels[predicted_label]

def _extract_keywords(self, text: str) -> List[str]:
    """텍스트에서 키워드 추출"""
    # 키워드 섹션 헤더 제거
    for header in ["키워드:", "키워드", "**키워드**:", "**키워드**"]:
        text = text.replace(header, "")
    
    # NER 모델을 사용한 키워드 추출
    inputs = self.ner_tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(self.device)
    
    with torch.no_grad():
        outputs = self.ner_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 토큰을 키워드로 변환
    tokens = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    keywords = []
    current_keyword = []
    
    for token, pred in zip(tokens, predictions[0]):
        if pred == 1:  # B-KW
            if current_keyword:
                keywords.append("".join(current_keyword))
                current_keyword = []
            current_keyword.append(token)
        elif pred == 2:  # I-KW
            current_keyword.append(token)
    
    if current_keyword:
        keywords.append("".join(current_keyword))
    
    # 후처리: 특수 토큰 제거 및 정제
    keywords = [
        kw.replace("#", "").replace("##", "").strip()
        for kw in keywords
        if kw not in ["[CLS]", "[SEP]", "[PAD]"]
    ]
    
    return [kw for kw in keywords if kw]

def _process_section_content(self, text: str) -> str:
    """섹션 내용 처리 및 정제"""
    # 특수 문자 및 마크다운 제거
    text = re.sub(r'\*\*|\[|\]|\#', '', text)
    return text.strip()

def _post_process_steps(self, steps: List[Dict]) -> List[Dict]:
    """단계 데이터 후처리"""
    # 빈 필드 제거 및 정제
    processed_steps = []
    for step in steps:
        # 키워드 중복 제거 및 정제
        step["keywords"] = list(set(step["keywords"]))
        
        # 빈 필드 기본값 설정
        if not step["description"]:
            step["description"] = f"{step['step_number']}단계"
        if not step["methodology"]:
            step["methodology"] = "방법론 미정의"
        if not step["output_format"]:
            step["output_format"] = "결과물 미정의"
        
        processed_steps.append(step)
    
    return processed_steps