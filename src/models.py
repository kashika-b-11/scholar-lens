"""
ScholarLens Pydantic Data Models
Defines structured schemas for type safety and API output validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExecutiveSummary(BaseModel):
    """Structured representation of the paper's core contributions."""
    paper_title: str = Field(
        default="Untitled Research Paper",
        description="Extracted or inferred title of the paper."
    )
    tldr: str = Field(
        ...,
        description="One high-impact sentence summarizing the core scientific breakthrough."
    )
    problem_statement: str = Field(
        ...,
        description="The fundamental limitation, gap, or research question addressed."
    )
    proposed_methodology: str = Field(
        ...,
        description="The novel architecture, algorithm, or methodology proposed."
    )
    key_findings: List[str] = Field(
        default_factory=list,
        description="Key quantitative benchmark results or empirical findings."
    )
    practical_applications: List[str] = Field(
        default_factory=list,
        description="Real-world implications, downstream tasks, or target industries."
    )


class ClaimItem(BaseModel):
    """Individual empirical claim and its verification audit."""
    claim: str = Field(
        ...,
        description="The specific assertion made by the authors."
    )
    metric_or_evidence: str = Field(
        ...,
        description="Experimental evidence, metric, or dataset cited."
    )
    status: str = Field(
        ...,
        description="Verification rating: 'Empirically Verified', 'Partially Supported', or 'Unsupported Claim / Author Speculation'."
    )
    limitations: Optional[str] = Field(
        default="None explicitly stated",
        description="Caveats, hardware constraints, or limitations."
    )


class ClaimAuditResult(BaseModel):
    """Complete collection of claims audited across the paper."""
    claims_audit: List[ClaimItem] = Field(
        default_factory=list,
        description="List of audited empirical claims."
    )
    overall_methodology_assessment: str = Field(
        default="Empirical evaluation follows standard benchmarks.",
        description="Peer-review evaluation of methodology rigor."
    )


class JargonItem(BaseModel):
    """Domain-specific technical term translated for intuitive understanding."""
    term: str = Field(
        ...,
        description="The technical term or domain acronym."
    )
    formal_definition: str = Field(
        ...,
        description="Rigorous academic definition."
    )
    intuitive_analogy: str = Field(
        ...,
        description="Everyday real-world metaphor illustrating how the concept works."
    )
    importance: str = Field(
        ...,
        description="Why this concept is crucial to the paper's mechanism."
    )


class JargonGlossary(BaseModel):
    """Glossary of simplified domain concepts."""
    jargon_list: List[JargonItem] = Field(
        default_factory=list,
        description="List of simplified technical concepts."
    )


class DocumentMetadata(BaseModel):
    """Metadata regarding the ingested document."""
    filename: str
    char_count: int
    word_count: int
    estimated_tokens: int
    detected_sections: List[str] = Field(default_factory=list)


class FullAnalysisResult(BaseModel):
    """Consolidated outcome of the multi-stage NLP analysis pipeline."""
    metadata: DocumentMetadata
    executive_summary: ExecutiveSummary
    claim_audit: ClaimAuditResult
    jargon_glossary: JargonGlossary
    execution_time_seconds: float
    model_used: str
