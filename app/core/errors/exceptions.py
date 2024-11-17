class ResearchAutomationError(Exception):
    """기본 에러 클래스"""
    def __init__(self, message: str = None):
        self.message = message
        super().__init__(self.message)

class ResourceNotFoundError(ResearchAutomationError):
    """리소스를 찾을 수 없을 때 발생하는 에러"""
    pass

class APIError(ResearchAutomationError):
    """외부 API 호출 실패시 발생하는 에러"""
    pass

class ValidationError(ResearchAutomationError):
    """데이터 검증 실패시 발생하는 에러"""
    pass

class DatabaseError(ResearchAutomationError):
    """데이터베이스 작업 실패시 발생하는 에러"""
    pass
