from typing import List, Dict, Optional, Union
import aiohttp
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
from typing_extensions import Literal

# Define types for better type hinting
RecencyFilter = Literal["day", "week", "month", "year", "all"]
SearchDomain = Literal["academic", "web", "news"]

@dataclass
class SearchConfig:
    """Configuration for search parameters"""
    max_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 0.9
    return_images: bool = False
    return_related_questions: bool = False
    search_recency_filter: RecencyFilter = "month"
    top_k: int = 0
    stream: bool = False
    presence_penalty: float = 0
    frequency_penalty: float = 1
    search_domain: SearchDomain = "academic"

@dataclass
class AcademicReference:
    """Structure for academic reference data"""
    title: str
    authors: Optional[List[str]]
    abstract: Optional[str]
    publication_date: Optional[str]
    journal: Optional[str]
    doi: Optional[str]
    url: Optional[str]
    citations: Optional[int]

class PerplexityService:
    """Service for academic reference search using Perplexity AI API"""
    
    def __init__(self, api_key: str):
        """Initialize PerplexityService with API key and configuration"""
        self.api_key = api_key
        self.base_url = "https://api.perplexity.ai"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        
        # Default system prompt for academic search
        self.system_prompt = """You are an academic research assistant specializing in scientific literature search.
        Provide all responses in the following JSON format:
        {
            "references": [{
                "title": "Paper title",
                "authors": ["Author 1", "Author 2"],
                "abstract": "Paper abstract",
                "publication_date": "YYYY-MM-DD",
                "journal": "Journal name",
                "doi": "DOI number",
                "url": "Paper URL",
                "citations": integer_count
            }],
            "metadata": {
                "total_results": integer,
                "search_timestamp": "ISO datetime",
                "query_context": "search context"
            }
        }
        Use null for missing fields. Citations must be integers."""

    async def search_references(
        self,
        query: str,
        model: str = "llama-3.1-sonar-small-128k-online",
        config: Optional[SearchConfig] = None
    ) -> Dict:
        """
        Asynchronously search for academic references
        
        Args:
            query: Search query string
            model: Model identifier
            config: Search configuration parameters
            
        Returns:
            Dict containing search results and metadata
            
        Raises:
            PerplexityAPIError: For API-related errors
            ValidationError: For input validation errors
        """
        try:
            # Use default config if none provided
            config = config or SearchConfig()
            
            # Validate and prepare search query
            cleaned_query = self._sanitize_query(query)
            
            # Prepare API payload
            payload = self._prepare_payload(cleaned_query, model, config)
            
            # Execute search request
            async with aiohttp.ClientSession() as session:
                return await self._execute_search(session, payload)
                
        except aiohttp.ClientError as e:
            error_msg = f"Network error during API request: {str(e)}"
            self.logger.error(error_msg)
            raise PerplexityAPIError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing API response: {str(e)}"
            self.logger.error(error_msg)
            raise PerplexityAPIError(error_msg)
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    def _sanitize_query(self, query: str) -> str:
        """Sanitize and validate search query"""
        if not query or not query.strip():
            raise ValidationError("Search query cannot be empty")
        
        # Remove potentially harmful characters
        cleaned = re.sub(r'[^\w\s\-.,?!()]', '', query)
        
        # Ensure minimum query length
        if len(cleaned) < 3:
            raise ValidationError("Search query too short")
            
        return cleaned

    def _prepare_payload(
        self,
        query: str,
        model: str,
        config: SearchConfig
    ) -> Dict:
        """Prepare API request payload"""
        return {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "return_images": config.return_images,
            "return_related_questions": config.return_related_questions,
            "search_recency_filter": config.search_recency_filter,
            "top_k": config.top_k,
            "stream": config.stream,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
            "search_domain": config.search_domain
        }

    async def _execute_search(
        self,
        session: aiohttp.ClientSession,
        payload: Dict
    ) -> Dict:
        """Execute search request and process response"""
        async with session.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        ) as response:
            if response.status == 200:
                data = await response.json()
                return self._process_response(data)
            else:
                error_text = await response.text()
                raise PerplexityAPIError(
                    f"API Error {response.status}: {error_text}"
                )

    def _process_response(self, response_data: Dict) -> Dict:
        """Process and validate API response"""
        try:
            # Parse response content
            content = json.loads(response_data['choices'][0]['message']['content'])
            
            # Add metadata
            content['metadata'] = {
                'search_timestamp': datetime.utcnow().isoformat(),
                'total_results': len(content.get('references', [])),
                'response_id': response_data.get('id')
            }
            
            # Validate references
            content['references'] = [
                self._validate_reference(ref)
                for ref in content.get('references', [])
            ]
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error processing response: {str(e)}")
            raise PerplexityAPIError(f"Response processing error: {str(e)}")

    def _validate_reference(self, reference: Dict) -> Dict:
        """Validate and clean individual reference data"""
        try:
            # Convert to AcademicReference for validation
            ref = AcademicReference(
                title=reference.get('title'),
                authors=reference.get('authors'),
                abstract=reference.get('abstract'),
                publication_date=reference.get('publication_date'),
                journal=reference.get('journal'),
                doi=reference.get('doi'),
                url=reference.get('url'),
                citations=reference.get('citations')
            )
            
            # Ensure citations is integer
            if ref.citations is not None:
                ref.citations = int(ref.citations)
                
            return vars(ref)
            
        except Exception as e:
            self.logger.warning(f"Error validating reference: {str(e)}")
            return reference

class PerplexityAPIError(Exception):
    """Custom exception for Perplexity API errors"""
    pass

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass