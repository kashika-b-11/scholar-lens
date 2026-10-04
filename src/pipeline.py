"""
ScholarLens Pipeline Orchestrator.
Manages the end-to-end execution of NLP stages across the paper text.
"""

import time
from typing import Optional, Tuple
from .config_loader import config
from .document_parser import DocumentParser
from .llm_client import LLMClient
from .models import (
    ClaimAuditResult,
    DocumentMetadata,
    ExecutiveSummary,
    FullAnalysisResult,
    JargonGlossary,
)
from prompts.prompts import (
    CLAIM_VERIFICATION_PROMPT,
    EXECUTIVE_DIGEST_PROMPT,
    GROUNDED_QA_PROMPT,
    JARGON_DECODER_PROMPT,
    SYSTEM_ACADEMIC_PEER_REVIEWER,
    SYSTEM_GROUNDED_QA,
    SYSTEM_SCIENCE_COMMUNICATOR,
    build_prompt,
)


class ScholarLensPipeline:
    """End-to-end NLP pipeline for research paper comprehension and claim verification."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def run_full_analysis(
        self,
        paper_text: str,
        metadata: DocumentMetadata,
        progress_callback=None,
    ) -> FullAnalysisResult:
        """
        Executes the 3 primary analytical stages over the parsed paper text.
        
        Args:
            paper_text: Normalized paper body text.
            metadata: Document stats and detected sections.
            progress_callback: Optional callable for UI progress updates (progress, status_message).
        """
        start_time = time.time()
        
        # Enforce context truncation guardrails if paper exceeds config limits
        truncated_text = paper_text[: config.max_char_limit]

        # Stage 1: Executive Digest
        if progress_callback:
            progress_callback(0.15, "Stage 1/3: Synthesizing Executive Digest & TL;DR...")
        exec_summary = self.generate_executive_summary(truncated_text)

        # Stage 2: Claim Verification Audit
        if progress_callback:
            progress_callback(0.50, "Stage 2/3: Auditing Empirical Claims & Methodology...")
        claim_audit = self.audit_claims(truncated_text)

        # Stage 3: Jargon Decoder
        if progress_callback:
            progress_callback(0.85, "Stage 3/3: Translating Domain Jargon & Analogies...")
        jargon_glossary = self.decode_jargon(truncated_text)

        if progress_callback:
            progress_callback(1.0, "Analysis complete!")

        elapsed = time.time() - start_time

        return FullAnalysisResult(
            metadata=metadata,
            executive_summary=exec_summary,
            claim_audit=claim_audit,
            jargon_glossary=jargon_glossary,
            execution_time_seconds=round(elapsed, 2),
            model_used=self.llm_client.model_name,
        )

    def generate_executive_summary(self, text: str) -> ExecutiveSummary:
        """Executes Stage 1: Generates structured TL;DR and contribution breakdown."""
        prompt = build_prompt(EXECUTIVE_DIGEST_PROMPT, paper_text=text)
        return self.llm_client.generate_structured(
            prompt=prompt,
            response_model=ExecutiveSummary,
            system_instruction=SYSTEM_ACADEMIC_PEER_REVIEWER,
        )

    def audit_claims(self, text: str) -> ClaimAuditResult:
        """Executes Stage 2: Audits empirical claims against supporting metrics."""
        prompt = build_prompt(CLAIM_VERIFICATION_PROMPT, paper_text=text)
        return self.llm_client.generate_structured(
            prompt=prompt,
            response_model=ClaimAuditResult,
            system_instruction=SYSTEM_ACADEMIC_PEER_REVIEWER,
        )

    def decode_jargon(self, text: str) -> JargonGlossary:
        """Executes Stage 3: Extracts complex terminology and generates analogies."""
        prompt = build_prompt(JARGON_DECODER_PROMPT, paper_text=text)
        return self.llm_client.generate_structured(
            prompt=prompt,
            response_model=JargonGlossary,
            system_instruction=SYSTEM_SCIENCE_COMMUNICATOR,
        )

    def ask_grounded_question(self, text: str, question: str) -> str:
        """Executes Grounded Q&A against the paper context with anti-hallucination constraints."""
        truncated_text = text[: config.max_char_limit]
        prompt = build_prompt(GROUNDED_QA_PROMPT, context=truncated_text, question=question)
        return self.llm_client.generate_text(
            prompt=prompt,
            system_instruction=SYSTEM_GROUNDED_QA,
            temperature=0.1,  # Ultra-low temperature for factual Q&A
        )
