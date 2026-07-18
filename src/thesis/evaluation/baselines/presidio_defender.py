"""Presidio baseline — NER-based PII redaction.

Replaces detected entities (names, addresses, phones, emails) with typed placeholders.
Catches explicit PII but not implicit/contextual attribute leakage.

Requires spacy model:
    uv run python -m spacy download en_core_web_lg
"""

from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PresidioDefender:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.operators = {
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        }

    def rewrite(self, text: str) -> str:
        results = self.analyzer.analyze(text=text, language=self.language)
        if not results:
            return text
        anonymized = self.anonymizer.anonymize(
            text=text, analyzer_results=results, operators=self.operators
        )
        return anonymized.text
