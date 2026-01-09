from backend.app.services.retrieval.query_router import QueryRouter

router = QueryRouter()

queries = [
    # --- IPC / Statute focused ---
    "What is the punishment under section 302 IPC?",
    "Explain section 304A of IPC",
    "What does IPC say about culpable homicide?",
    "Define murder under Indian Penal Code",
    "Is intention necessary under section 300?",
    "Difference between section 299 and 300 IPC",
    "What is the punishment for attempt to murder?",
    "Explain offence of dacoity with murder",
    "What is the maximum punishment under IPC 376?",
    "Is death penalty mandatory under section 302?",

    # --- Judgment / Case law focused ---
    "What did the Supreme Court hold in the Nirbhaya case?",
    "Latest Supreme Court judgment on death penalty",
    "Explain ratio decidendi in Bachan Singh case",
    "What precedent applies to custodial death cases?",
    "High Court ruling on anticipatory bail in murder cases",
    "Summarize Supreme Court views on life imprisonment",
    "Case law related to section 302 IPC",
    "Judgment explaining rarest of rare doctrine",

    # --- Mixed IPC + Judgment ---
    "How has section 302 IPC been interpreted by Supreme Court?",
    "Judicial interpretation of culpable homicide not amounting to murder",
    "Important judgments explaining section 304 IPC",
    "What courts have said about intention vs knowledge under IPC?",
    "Explain punishment for murder with relevant case laws",

    # --- User document focused ---
    "Explain clause 5 of the uploaded contract",
    "What does this agreement say about termination?",
    "Summarize obligations of parties in this document",
    "Is there any penalty clause in the uploaded PDF?",
    "Explain liability mentioned in the uploaded agreement",

    # --- User + IPC mixed ---
    "Does this contract violate any IPC provision?",
    "Is clause 7 of the agreement legally valid under IPC?",
    "Does this document involve any criminal liability?",
    "Which IPC sections may apply to this agreement?",

    # --- Colloquial / messy real-world queries ---
    "What happens if someone kills another person in India?",
    "What law applies if a person dies because of negligence?",
    "Can someone get death penalty in India?",
    "What punishment will I get for murder?",
    "What does law say if killing was unintentional?",
    "Explain murder law simply",
    "Is life imprisonment same as 14 years?",

    # --- Student / exam style ---
    "Short note on culpable homicide",
    "Discuss murder under IPC",
    "Differentiate between murder and culpable homicide",
    "Explain section 299 with examples",
    "Explain section 300 with case laws",

    # --- Edge / ambiguous ---
    "Explain murder",
    "Explain homicide",
    "Explain Indian criminal law on killing",
    "Explain crime of murder in India"
]

for q in queries:
    route = router.route(q, has_user_docs=True)
    print(q)
    print(route)
    print("-" * 70)

