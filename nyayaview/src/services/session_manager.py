from streamlit import session_state

class SessionManager:
    def __init__(self):
        if 'query' not in session_state:
            session_state.query = ""
        if 'response' not in session_state:
            session_state.response = None
        if 'evidence' not in session_state:
            session_state.evidence = []
        if 'provenance' not in session_state:
            session_state.provenance = []

    def set_query(self, query):
        session_state.query = query

    def set_response(self, response):
        session_state.response = response

    def set_evidence(self, evidence):
        session_state.evidence = evidence

    def set_provenance(self, provenance):
        session_state.provenance = provenance

    def clear_session(self):
        session_state.query = ""
        session_state.response = None
        session_state.evidence = []
        session_state.provenance = []