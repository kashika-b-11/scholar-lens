"""
ScholarLens Component & Integration Smoke Tests.
Verifies parsing, prompt building, schema validation, and config loading.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Windows stdout supports utf-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


from src.config_loader import config
from src.document_parser import DocumentParser
from src.models import ClaimItem, ExecutiveSummary, JargonItem
from prompts.prompts import (
    CLAIM_VERIFICATION_PROMPT,
    EXECUTIVE_DIGEST_PROMPT,
    GROUNDED_QA_PROMPT,
    JARGON_DECODER_PROMPT,
    build_prompt,
)


def test_config_loading():
    """Verify that configuration loads expected default values."""
    assert config.default_provider in ["gemini", "openai"]
    assert config.temperature >= 0.0
    assert config.max_output_tokens > 0
    assert config.max_char_limit > 1000
    print("✅ Configuration loading test passed.")


def test_document_parser():
    """Verify document normalization, section detection, and token estimation."""
    sample_text = """
    Abstract
    This is an abstract demonstrating sequence-to-sequence learning.
    
    1. Introduction
    Deep learning has revolutionized NLP. We intro-
    duce a novel mechanism.
    
    2. Experiments
    Results on WMT show an improvement of 2.5 BLEU.
    """
    cleaned, metadata = DocumentParser.parse_document(sample_text, "test.txt")
    
    # Assert hyphenation was resolved
    assert "introduce a novel mechanism" in cleaned
    assert metadata.char_count > 0
    assert metadata.word_count > 0
    assert metadata.estimated_tokens > 0
    assert "Abstract" in metadata.detected_sections
    assert "Introduction" in metadata.detected_sections
    assert "Experiments" in metadata.detected_sections
    print("✅ Document parser test passed.")


def test_prompt_formatting():
    """Verify prompt formatting safely injects few-shot exemplars and variables."""
    dummy_text = "Sample research paper text."
    
    # Test Executive Summary Prompt
    p1 = build_prompt(EXECUTIVE_DIGEST_PROMPT, paper_text=dummy_text)
    assert dummy_text in p1
    assert "OUTPUT FORMAT" in p1

    # Test Claim Verification Prompt with Few-Shot injection
    p2 = build_prompt(CLAIM_VERIFICATION_PROMPT, paper_text=dummy_text)
    assert dummy_text in p2
    assert "FEW-SHOT EXAMPLE" in p2 or "EXAMPLE OUTPUT" in p2

    # Test Jargon Decoder Prompt
    p3 = build_prompt(JARGON_DECODER_PROMPT, paper_text=dummy_text)
    assert dummy_text in p3

    # Test Grounded QA Prompt
    p4 = build_prompt(GROUNDED_QA_PROMPT, context=dummy_text, question="What is the BLEU score?")
    assert dummy_text in p4
    assert "What is the BLEU score?" in p4
    print("✅ Prompt formatting and exemplar injection test passed.")


def test_pydantic_schemas():
    """Verify Pydantic models validate valid and structured responses correctly."""
    summary = ExecutiveSummary(
        paper_title="Attention Is All You Need",
        tldr="Replaces recurrence with attention mechanism.",
        problem_statement="Recurrent models cannot be parallelized across sequence tokens.",
        proposed_methodology="Transformer architecture using multi-head self-attention.",
        key_findings=["28.4 BLEU on English-to-German", "41.8 BLEU on English-to-French"],
        practical_applications=["Machine Translation", "Text Summarization"]
    )
    assert summary.tldr is not None
    assert len(summary.key_findings) == 2

    claim = ClaimItem(
        claim="Transformer achieves state of the art in machine translation.",
        metric_or_evidence="28.4 BLEU score on WMT 2014 En-De.",
        status="Empirically Verified",
        limitations="Trained on 8 P100 GPUs for 3.5 days."
    )
    assert claim.status == "Empirically Verified"

    jargon = JargonItem(
        term="Scaled Dot-Product Attention",
        formal_definition="An attention function computing dot products of queries with keys divided by sqrt(d_k).",
        intuitive_analogy="Like using a search engine query where relevance weights are assigned to matched key phrases.",
        importance="Allows constant time parallel computation across attention matrix."
    )
    assert jargon.term == "Scaled Dot-Product Attention"
    print("✅ Pydantic schemas validation test passed.")


if __name__ == "__main__":
    print("Running ScholarLens Smoke Tests...")
    test_config_loading()
    test_document_parser()
    test_prompt_formatting()
    test_pydantic_schemas()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
