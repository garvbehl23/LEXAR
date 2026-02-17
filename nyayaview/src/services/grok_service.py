"""
Grok API Service Module
========================
Handles integration with Grok (xAI) API for enhanced text generation.

When user provides Grok API key:
- Retrieval still uses LEXAR
- Generation uses Grok with evidence context
- Evidence grounding enforced via system prompt

Fallback:
- If no API key, use full LEXAR local generation
"""

import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time


@dataclass
class GrokResponse:
    """Response from Grok API"""
    text: str
    success: bool
    error_message: str = ""
    token_usage: int = 0
    processing_time: float = 0.0


class GrokService:
    """
    Grok API integration service.
    
    Implements evidence-constrained generation using Grok model
    while maintaining LEXAR architecture principles.
    """
    
    # Grok API endpoints
    API_ENDPOINT = "https://api.x.ai/v1/chat/completions"
    OPENAI_COMPATIBLE_ENDPOINT = "https://api.x.ai/openai/v1/chat/completions"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._validated = False
    
    def set_api_key(self, api_key: str) -> bool:
        """Set and validate API key"""
        self.api_key = api_key
        self._validated = False
        
        if api_key:
            self._validated = self.validate_api_key()
        
        return self._validated
    
    def validate_api_key(self) -> bool:
        """Validate the API key with a minimal request"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Minimal validation request
            payload = {
                "messages": [{"role": "user", "content": "test"}],
                "model": "grok-beta",
                "max_tokens": 5
            }
            
            response = requests.post(
                self.OPENAI_COMPATIBLE_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=10
            )
            
            # Check if response is valid (even error responses indicate valid key format)
            if response.status_code in [200, 400, 429]:
                self._validated = True
                return True
            elif response.status_code == 401:
                self._validated = False
                return False
            else:
                # Assume valid if we get any response
                self._validated = True
                return True
                
        except Exception as e:
            print(f"Grok API validation error: {e}")
            # Don't invalidate on network errors
            return True
    
    @property
    def is_available(self) -> bool:
        """Check if Grok API is available"""
        return bool(self.api_key)
    
    def generate(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        case_type: str = "constitutional",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> GrokResponse:
        """
        Generate text using Grok API with LEXAR evidence grounding.
        
        Args:
            query: User's legal question
            evidence_chunks: Retrieved evidence from LEXAR
            case_type: Type of legal case
            temperature: Generation temperature (lower = more deterministic)
            max_tokens: Maximum tokens to generate
            
        Returns:
            GrokResponse with generated text
        """
        if not self.api_key:
            return GrokResponse(
                text="",
                success=False,
                error_message="Grok API key not configured"
            )
        
        start_time = time.time()
        
        try:
            # Build evidence context
            evidence_context = self._build_evidence_context(evidence_chunks)
            
            # Build system prompt with LEXAR principles
            system_prompt = self._build_system_prompt(case_type, evidence_context)
            
            # Build user prompt
            user_prompt = self._build_user_prompt(query, case_type)
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "model": "grok-beta",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                self.OPENAI_COMPATIBLE_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=60
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                text = data['choices'][0]['message']['content']
                token_usage = data.get('usage', {}).get('total_tokens', 0)
                
                return GrokResponse(
                    text=text,
                    success=True,
                    token_usage=token_usage,
                    processing_time=processing_time
                )
            else:
                return GrokResponse(
                    text="",
                    success=False,
                    error_message=f"Grok API error: {response.status_code} - {response.text}",
                    processing_time=processing_time
                )
                
        except requests.exceptions.Timeout:
            return GrokResponse(
                text="",
                success=False,
                error_message="Grok API request timed out",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return GrokResponse(
                text="",
                success=False,
                error_message=f"Grok API error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _build_evidence_context(self, evidence_chunks: List[Dict[str, Any]]) -> str:
        """Build evidence context string from chunks"""
        if not evidence_chunks:
            return "No evidence retrieved. Generate response indicating insufficient evidence."
        
        context_parts = []
        for i, chunk in enumerate(evidence_chunks, 1):
            if isinstance(chunk, dict):
                text = chunk.get('text', str(chunk))
                source = chunk.get('source', 'Unknown source')
                section = chunk.get('section', '')
                score = chunk.get('score', 0.0)
            else:
                # Handle RetrievedChunk dataclass
                text = getattr(chunk, 'text', str(chunk))
                source = getattr(chunk, 'source', 'Unknown source')
                section = getattr(chunk, 'section', '')
                score = getattr(chunk, 'score', 0.0)
            
            context_parts.append(f"""
EVIDENCE {i}:
Source: {source}
Section: {section}
Relevance Score: {score:.3f}
Content: {text}
""")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, case_type: str, evidence_context: str) -> str:
        """Build system prompt with LEXAR principles"""
        
        case_type_instructions = {
            'constitutional': "Focus on fundamental rights, constitutional provisions, and landmark constitutional judgments.",
            'criminal': "Focus on IPC sections, CrPC procedures, sentencing guidelines, and criminal precedents.",
            'civil': "Focus on CPC procedures, contract law, property law, and civil remedies.",
            'writ': "Focus on writ jurisdiction, administrative law, and judicial review principles.",
            'slp': "Focus on substantial questions of law, special leave criteria, and appellate standards."
        }
        
        type_instruction = case_type_instructions.get(case_type, case_type_instructions['constitutional'])
        
        return f"""You are an expert legal AI assistant integrated with the LEXAR (Legal Expert Analysis and Reasoning) architecture, providing analysis in the style of the Supreme Court of India.

CRITICAL CONSTRAINTS (LEXAR Architecture Principles):
1. NO GENERATION WITHOUT EVIDENCE: Every statement must be grounded in the retrieved evidence below. If evidence is insufficient, explicitly state this.
2. HARD ATTENTION MASKING: Focus ONLY on the provided evidence. Do not use parametric knowledge for legal facts.
3. EVIDENCE METADATA PRESERVATION: Cite specific sources, sections, and relevance scores.
4. PROVABLE GROUNDING: All conclusions must trace back to specific evidence chunks.
5. CONFIDENCE SCORING: Indicate confidence levels based on evidence support.

CASE TYPE: {case_type.upper()}
{type_instruction}

RETRIEVED EVIDENCE (from LEXAR retrieval system):
{evidence_context}

OUTPUT FORMAT:
Structure your response in the formal Supreme Court style:
I. ISSUES FRAMED - List the legal issues identified
II. STATUTORY POSITION - Cite applicable laws and precedents from evidence
III. JUDICIAL REASONING - Analyze with explicit evidence references [Evidence 1], [Evidence 2], etc.
IV. CONCLUSION - Evidence-grounded conclusion with confidence
V. ORDER - Formal disposition

Remember: You can ONLY use information from the RETRIEVED EVIDENCE above. Any statement not supported by evidence must be flagged as inference or must be omitted."""
    
    def _build_user_prompt(self, query: str, case_type: str) -> str:
        """Build user prompt"""
        return f"""LEGAL QUERY:
{query}

Please provide a comprehensive legal analysis in the Supreme Court of India style, strictly grounded in the retrieved evidence. Include:
1. Clear framing of legal issues
2. Applicable statutory provisions with citations
3. Judicial reasoning with evidence references
4. Confidence-scored conclusions
5. Formal order/disposition

Ensure all statements are traceable to specific evidence chunks."""
    
    def generate_advocate_view(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        case_type: str = "constitutional"
    ) -> GrokResponse:
        """Generate analysis in Advocate mode (detailed citations)"""
        
        # Modify system prompt for advocate view
        advocate_system = """You are assisting an Advocate. Provide detailed analysis with:
- Full citation mapping (case names, sections, paragraph numbers)
- Highlighted supporting statutes
- Cross-references to related provisions
- Precedent links with distinguishing factors
- Strategic considerations

Format for courtroom presentation."""
        
        return self.generate(
            query=f"[ADVOCATE MODE] {query}",
            evidence_chunks=evidence_chunks,
            case_type=case_type
        )
    
    def generate_bench_view(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        case_type: str = "constitutional"
    ) -> GrokResponse:
        """Generate analysis in Bench mode (structured judgment)"""
        
        return self.generate(
            query=f"[BENCH MODE - JUDGMENT FORMAT] {query}",
            evidence_chunks=evidence_chunks,
            case_type=case_type
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'api_key_set': bool(self.api_key),
            'validated': self._validated,
            'endpoint': self.OPENAI_COMPATIBLE_ENDPOINT
        }


class OpenAIService:
    """
    OpenAI API integration service (alternative to Grok).
    Same interface as GrokService for easy swapping.
    """
    
    API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self._validated = False
    
    def set_api_key(self, api_key: str) -> bool:
        """Set and validate API key"""
        self.api_key = api_key
        self._validated = bool(api_key)
        return self._validated
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        case_type: str = "constitutional",
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> GrokResponse:
        """Generate using OpenAI API"""
        if not self.api_key:
            return GrokResponse(
                text="",
                success=False,
                error_message="OpenAI API key not configured"
            )
        
        start_time = time.time()
        
        try:
            import openai
            openai.api_key = self.api_key
            
            grok_service = GrokService()
            evidence_context = grok_service._build_evidence_context(evidence_chunks)
            system_prompt = grok_service._build_system_prompt(case_type, evidence_context)
            user_prompt = grok_service._build_user_prompt(query, case_type)
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return GrokResponse(
                text=response.choices[0].message.content,
                success=True,
                token_usage=response.usage.total_tokens,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return GrokResponse(
                text="",
                success=False,
                error_message=f"OpenAI API error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'api_key_set': bool(self.api_key),
            'validated': self._validated,
            'endpoint': self.API_ENDPOINT
        }
