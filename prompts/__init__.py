"""
ScholarLens Prompt Engineering Module.
Exposes all prompt templates, system instructions, and formatting utilities.
"""

from .prompts import (
    SYSTEM_ACADEMIC_PEER_REVIEWER,
    SYSTEM_SCIENCE_COMMUNICATOR,
    SYSTEM_GROUNDED_QA,
    EXECUTIVE_DIGEST_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    JARGON_DECODER_PROMPT,
    GROUNDED_QA_PROMPT,
    build_prompt,
)

__all__ = [
    "SYSTEM_ACADEMIC_PEER_REVIEWER",
    "SYSTEM_SCIENCE_COMMUNICATOR",
    "SYSTEM_GROUNDED_QA",
    "EXECUTIVE_DIGEST_PROMPT",
    "CLAIM_VERIFICATION_PROMPT",
    "JARGON_DECODER_PROMPT",
    "GROUNDED_QA_PROMPT",
    "build_prompt",
]
