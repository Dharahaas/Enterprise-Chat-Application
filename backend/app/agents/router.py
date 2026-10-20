"""Lightweight query normalization, conversation intent, and agent routing."""

from __future__ import annotations

import difflib
import re

from app.agents.agent_specs import AGENTS


KEYWORDS: dict[str, list[str]] = {
    "hr": [
        "leave", "leaves", "payroll", "salary", "onboarding", "onboard",
        "benefits", "health plan", "insurance", "attendance", "reimbursement",
        "performance review", "appraisal", "exit", "resignation", "joining",
        "offer letter", "pf", "provident", "tax", "form 16",
    ],
    "it": [
        "password", "vpn", "laptop", "hardware", "software", "install",
        "ticket", "network", "email", "outlook", "wifi", "printer",
        "mfa", "2fa", "authenticator", "reset", "login", "access",
        "anyconnect", "antivirus", "phishing", "ransomware", "firewall",
    ],
    "policy": [
        "policy", "policies", "compliance", "code of conduct", "conduct",
        "ethics", "data classification", "classification", "remote work",
        "hybrid", "wfh", "gift", "conflict of interest",
        "information security", "incident", "breach", "gdpr", "pii",
        "confidential", "restricted", "internal use", "iso 27001", "audit",
    ],
}

_COMMON_CORRECTIONS = {
    "helo": "hello",
    "helllo": "hello",
    "thx": "thanks",
    "thnks": "thanks",
    "thankyou": "thank you",
    "thanku": "thank you",
    "ty": "thanks",
    "wht": "what",
    "wat": "what",
    "nt": "not",
    "wrking": "working",
    "wrkng": "working",
    "polcy": "policy",
    "vpm": "vpn",
}
_QUERY_VOCABULARY = {
    "hello", "hi", "hey", "thanks", "thank", "you", "welcome", "what",
    "can", "do", "how", "help", "about", "tell", "me", "my", "is", "not",
    "working", "company", "guidelines", "support", "assistant", "today",
}
_QUERY_VOCABULARY.update(
    word
    for phrases in KEYWORDS.values()
    for phrase in phrases
    for word in phrase.split()
)
_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")
_PHRASE_WEIGHTS = {
    "policy": 1,
    "policies": 1,
    "leave": 2,
    "leaves": 2,
}
_CONVERSATION_REPLIES = {
    "greeting": (
        "Hello! How can I help you today? You can ask me about HR policies, "
        "IT support, or company guidelines."
    ),
    "acknowledgement": "Okay! Let me know if you need anything else.",
    "positive_acknowledgement": "Glad to hear it! Let me know if you need anything else.",
    "thanks": "You're welcome! Let me know if you need help with anything else.",
    "farewell": "Goodbye! Feel free to come back if you need help.",
    "capabilities": (
        "I can help answer questions using the available HR documents "
        "(such as leave and payroll policies), IT support guides (such as VPN "
        "and password help), and company policies or compliance documents."
    ),
}


def normalize_query(question: str) -> str:
    """Correct high-confidence typos locally without changing original history."""
    def correct(match: re.Match[str]) -> str:
        word = match.group()
        normalized = word.casefold()
        if normalized in _COMMON_CORRECTIONS:
            return _COMMON_CORRECTIONS[normalized]
        if len(normalized) < 4 or normalized in _QUERY_VOCABULARY:
            return word

        candidates = difflib.get_close_matches(
            normalized,
            _QUERY_VOCABULARY,
            n=2,
            cutoff=0.84,
        )
        if not candidates:
            return word
        if len(candidates) == 2:
            best_score = difflib.SequenceMatcher(None, normalized, candidates[0]).ratio()
            next_score = difflib.SequenceMatcher(None, normalized, candidates[1]).ratio()
            if best_score - next_score < 0.06:
                return word
        return candidates[0]

    return _WORD_PATTERN.sub(correct, question)


def classify_conversation_intent(question: str) -> str | None:
    """Recognize brief social turns without matching knowledge questions."""
    q = normalize_query(question).casefold().strip()
    q = re.sub(r"[.!?,;:]+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    if re.fullmatch(
        r"(?:hello|hi|hey)(?: there)?|good (?:morning|afternoon|evening)",
        q,
    ):
        return "greeting"
    thank_you = r"(?:thanks?|thank you)(?: so much| a lot)?"
    acknowledgement = (
        r"(?:ok(?:ay)?|all right|alright|got it|understood|noted|sure|"
        r"sounds good|makes sense|cool)"
    )
    if re.fullmatch(thank_you, q) or re.fullmatch(
        rf"{acknowledgement} (?:and )?{thank_you}", q
    ):
        return "thanks"
    if re.fullmatch(
        rf"{acknowledgement}(?: (?:and )?got it)?",
        q,
    ):
        return "acknowledgement"
    if re.fullmatch(r"(?:good|great|perfect|nice)", q):
        return "positive_acknowledgement"
    if re.fullmatch(
        r"(?:bye(?: for now)?|goodbye|see you(?: later)?|take care)",
        q,
    ):
        return "farewell"
    if q in {
        "what can you do", "what do you do", "how can you help",
        "what are your capabilities", "who are you", "help",
    } or re.fullmatch(r"how can (?:you|u) help(?: me)?", q):
        return "capabilities"
    return None


def conversation_reply(intent: str) -> str:
    """Return the fixed, no-retrieval response for a recognized intent."""
    return _CONVERSATION_REPLIES[intent]


def classify_question(question: str, fallback: str = "hr") -> str:
    """Choose the most relevant knowledge domain using corrected query terms."""
    q = normalize_query(question).casefold()
    scores: dict[str, int] = {agent_id: 0 for agent_id in AGENTS}
    for agent_id, phrases in KEYWORDS.items():
        for phrase in phrases:
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
                scores[agent_id] += _PHRASE_WEIGHTS.get(phrase, 1)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return fallback
    return best
