import re


class QueryRouter:
    """
    Routes queries to appropriate indices based on intent.
    """

    IPC_KEYWORDS = [
        "ipc", "section", "punishment", "offence", "offense ",
        "definition", "act", "code",
        # Expanded IPC-related terms
        "criminal", "penal", "culpable", "mens rea", "actus reus",
        "bailable", "non-bailable", "cognizable", "non-cognizable",
        "compoundable", "non-compoundable", "rigorous imprisonment",
        "simple imprisonment", "fine", "death penalty", "life imprisonment",
        "attempt", "abetment", "conspiracy", "common intention",
        "wrongful gain", "wrongful loss", "good faith", "negligence",
        "homicide", "murder", "culpable homicide", "theft", "robbery",
        "dacoity", "extortion", "cheating", "forgery", "defamation",
        "assault", "criminal force", "kidnapping", "abduction",
        "rape", "adultery", "mischief", "rioting", "unlawful assembly"
    ]

    JUDGMENT_KEYWORDS = [
        "case", "judgment", "precedent", "held",
        "supreme court", "high court", "appeal",
        # Expanded judgment/case law-related terms
        "ruling", "verdict", "order", "decree", "decision",
        "bench", "division bench", "constitutional bench",
        "plaintiff", "defendant", "petitioner", "respondent",
        "writ", "habeas corpus", "mandamus", "certiorari", "prohibition", "quo warranto",
        "original jurisdiction", "appellate jurisdiction",
        "ratio decidendi", "obiter dicta", "stare decisis",
        "overruled", "distinguished", "affirmed", "reversed",
        "stay", "injunction", "interim relief", "final order",
        "cited", "citation", "reported", "unreported",
        "justice", "hon'ble", "learned counsel", "submissions",
        "findings of fact", "question of law", "sub judice",
        "res judicata", "amicus curiae", "suo moto"
    ]

    SECTION_PATTERN = re.compile(r"\b\d{1,3}\b")

    def route(self, query: str, has_user_docs: bool = False):
        q = query.lower()

        use_ipc = False
        use_judgment = False
        use_user = has_user_docs

        # IPC signals
        if any(k in q for k in self.IPC_KEYWORDS):
            use_ipc = True

        if self.SECTION_PATTERN.search(q):
            use_ipc = True

        # Judgment signals
        if any(k in q for k in self.JUDGMENT_KEYWORDS):
            use_judgment = True

        # Default fallback
        if not use_ipc and not use_judgment:
            use_ipc = True
            use_judgment = True

        return {
            "ipc": use_ipc,
            "judgment": use_judgment,
            "user": use_user
        }
