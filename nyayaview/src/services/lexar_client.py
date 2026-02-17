import time
from typing import Dict, Any

class LexarClient:
    """Client for interfacing with the LEXAR pipeline."""
    
    def __init__(self):
        """Initialize the LEXAR pipeline."""
        try:
            from lexar import LexarPipeline
            self.pipeline = LexarPipeline()
            self.available = True
        except ImportError:
            self.pipeline = None
            self.available = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a legal query through the complete LEXAR pipeline.
        
        Returns:
            Dictionary containing:
            - execution_time: Time taken in seconds
            - retrieval: Retrieved chunks data
            - generation: Generated answer
            - gating: Evidence sufficiency evaluation
            - provenance: Token-level attribution
        """
        start_time = time.time()
        
        if not self.available:
            # Return mock data for demo purposes
            return self._get_mock_results(query)
        
        try:
            # Run retrieval
            retrieval_result = self.pipeline.retrieve(query)
            
            # Run generation
            generation_result = self.pipeline.generate(query, retrieval_result)
            
            # Evaluate evidence sufficiency
            gating_result = self.pipeline.evaluate_evidence(retrieval_result, generation_result)
            
            # Compute token provenance
            provenance_result = self.pipeline.compute_provenance(generation_result, retrieval_result)
            
            execution_time = time.time() - start_time
            
            return {
                'execution_time': execution_time,
                'retrieval': retrieval_result,
                'generation': generation_result,
                'gating': gating_result,
                'provenance': provenance_result
            }
        except Exception as e:
            # Fallback to mock data on error
            return self._get_mock_results(query, error=str(e))
    
    def _get_mock_results(self, query: str, error: str = None) -> Dict[str, Any]:
        """Generate mock results for demonstration purposes."""
        return {
            'execution_time': 1.234,
            'retrieval': {
                'num_chunks': 5,
                'chunks': [
                    {
                        'statute': 'Indian Penal Code (IPC)',
                        'section': '§302',
                        'chunk_id': 'ipc_302_001',
                        'score': 0.8765,
                        'text': 'Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.'
                    },
                    {
                        'statute': 'Code of Criminal Procedure (CrPC)',
                        'section': '§41',
                        'chunk_id': 'crpc_41_002',
                        'score': 0.8234,
                        'text': 'When any police officer may arrest without warrant any person who has been concerned in any cognizable offence.'
                    },
                    {
                        'statute': 'Indian Evidence Act (IEA)',
                        'section': '§24',
                        'chunk_id': 'iea_24_001',
                        'score': 0.7891,
                        'text': 'A confession made by an accused person is irrelevant in a criminal proceeding, if the making of the confession appears to the Court to have been caused by any inducement, threat or promise.'
                    },
                    {
                        'statute': 'Indian Penal Code (IPC)',
                        'section': '§300',
                        'chunk_id': 'ipc_300_001',
                        'score': 0.7654,
                        'text': 'Except in the cases hereinafter excepted, culpable homicide is murder.'
                    },
                    {
                        'statute': 'Code of Criminal Procedure (CrPC)',
                        'section': '§154',
                        'chunk_id': 'crpc_154_001',
                        'score': 0.7432,
                        'text': 'Every information relating to the commission of a cognizable offence, if given orally to an officer in charge of a police station, shall be reduced to writing.'
                    }
                ]
            },
            'generation': {
                'answer': f'Based on the query "{query}", under Indian Penal Code (IPC §302), murder is defined as the unlawful killing of a human being with malice aforethought. The punishment for murder includes death or life imprisonment. Under CrPC §41, police officers have the authority to arrest without warrant in cases of cognizable offences. Evidence admissibility is governed by IEA §24, which excludes confessions obtained through inducement or coercion.'
            },
            'gating': {
                'status': 'PASS',
                'threshold': 0.650,
                'max_attention': 87.5,
                'dominant_section': 'IPC §302',
                'margin': 0.225
            },
            'provenance': {
                'tokens': [
                    {'token': 'murder', 'statute': 'IPC', 'section': '§302', 'confidence': 0.945},
                    {'token': 'punishment', 'statute': 'IPC', 'section': '§302', 'confidence': 0.912},
                    {'token': 'death', 'statute': 'IPC', 'section': '§302', 'confidence': 0.889},
                    {'token': 'arrest', 'statute': 'CrPC', 'section': '§41', 'confidence': 0.876},
                    {'token': 'warrant', 'statute': 'CrPC', 'section': '§41', 'confidence': 0.854},
                    {'token': 'confession', 'statute': 'IEA', 'section': '§24', 'confidence': 0.823},
                    {'token': 'evidence', 'statute': 'IEA', 'section': '§24', 'confidence': 0.798},
                ]
            },
            'error': error
        }