"""
================================================================================
ScholarLens: Research Paper Digest & Claim Verification Copilot
Main Streamlit Application
================================================================================
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config_loader import config
from src.document_parser import DocumentParser
from src.llm_client import LLMClient, LLMClientError
from src.pipeline import ScholarLensPipeline


# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="ScholarLens Copilot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Academic Theme CSS (Dark & Light Mode Adaptive)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3B82F6;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .tldr-box {
        background: rgba(99, 102, 241, 0.12);
        border-left: 5px solid #6366F1;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 1.05rem;
        font-weight: 500;
        color: inherit;
    }
    .status-badge-verified {
        background-color: rgba(16, 185, 129, 0.18);
        color: #10B981;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-badge-partial {
        background-color: rgba(245, 158, 11, 0.18);
        color: #F59E0B;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .status-badge-unsupported {
        background-color: rgba(239, 68, 68, 0.18);
        color: #EF4444;
        padding: 0.25rem 0.65rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .metric-card {
        background-color: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: inherit;
        padding: 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "current_paper_text" not in st.session_state:
    st.session_state.current_paper_text = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================

st.sidebar.title("⚙️ Configuration")

# Provider & API Key
provider_options = ["gemini", "openai"]
selected_provider = st.sidebar.selectbox(
    "LLM Provider",
    provider_options,
    index=0 if config.default_provider == "gemini" else 1,
    help="Google Gemini offers a free API key with 1M token context window."
)

env_api_key = config.get_api_key(selected_provider) or ""
api_key_input = st.sidebar.text_input(
    f"{selected_provider.upper()} API Key",
    value=env_api_key,
    type="password",
    help="Enter your API key or configure it in .env file."
)
effective_api_key = api_key_input.strip() or env_api_key
if env_api_key:
    st.sidebar.caption("🔒 Key loaded from `.env` file.")


# Model Selection
if selected_provider == "gemini":
    model_choices = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-flash-lite-latest"]
    default_model = config.gemini_model if config.gemini_model in model_choices else model_choices[0]
else:
    model_choices = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    default_model = config.openai_model if config.openai_model in model_choices else model_choices[0]



selected_model = st.sidebar.selectbox("Model", model_choices, index=model_choices.index(default_model))

temperature = st.sidebar.slider(
    "Temperature (Factual Precision)",
    min_value=0.0,
    max_value=1.0,
    value=config.temperature,
    step=0.05,
    help="Lower values (0.1 - 0.2) ensure strict factual accuracy and prevent hallucinations in academic papers."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Input Source")
input_mode = st.sidebar.radio(
    "Select Paper Input Mode",
    ["🧪 Load Benchmark Sample", "📤 Upload File (PDF / TXT)", "✍️ Paste Text"],
    index=0
)

paper_raw_text = ""
uploaded_file_name = "sample_paper.txt"

if input_mode == "🧪 Load Benchmark Sample":
    sample_path = BASE_DIR / "samples" / "attention_is_all_you_need_sample.txt"
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            paper_raw_text = f.read()
        uploaded_file_name = "Attention_Is_All_You_Need_Excerpt.txt"
        st.sidebar.success("Loaded 'Attention Is All You Need' benchmark excerpt.")
    else:
        st.sidebar.error("Sample file not found.")

elif input_mode == "📤 Upload File (PDF / TXT)":
    uploaded_file = st.sidebar.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        uploaded_file_name = uploaded_file.name
        if uploaded_file_name.lower().endswith(".pdf"):
            paper_raw_text = DocumentParser.extract_from_pdf(uploaded_file.read())
        else:
            paper_raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.sidebar.info(f"Loaded {uploaded_file_name} ({len(paper_raw_text):,} characters)")

elif input_mode == "✍️ Paste Text":
    paper_raw_text = st.sidebar.text_area("Paste research abstract or paper text here:", height=200)
    uploaded_file_name = "pasted_abstract.txt"

# Run Analysis CTA
analyze_btn = st.sidebar.button("🚀 Analyze Paper", type="primary", use_container_width=True)


# ==============================================================================
# MAIN PAGE HEADER & METRICS BAR
# ==============================================================================

st.markdown('<div class="main-header">🎓 ScholarLens</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Research Paper Digest, Claim Verification & Jargon Simplification Copilot</div>',
    unsafe_allow_html=True
)

# Execution Logic
if analyze_btn:
    if not effective_api_key:
        st.error(f"⚠️ Please provide an API key for {selected_provider.upper()} in the sidebar or in your `.env` file.")
    elif not paper_raw_text.strip():
        st.warning("⚠️ Please provide paper text via sample, file upload, or text paste.")
    else:
        try:
            # Step 1: Parse and Clean Document
            cleaned_text, doc_metadata = DocumentParser.parse_document(
                source=paper_raw_text,
                filename=uploaded_file_name
            )
            st.session_state.current_paper_text = cleaned_text

            # Step 2: Initialize LLM Client & Pipeline
            llm_client = LLMClient(
                provider=selected_provider,
                api_key=effective_api_key,
                model_name=selected_model,
                temperature=temperature,
            )

            pipeline = ScholarLensPipeline(llm_client=llm_client)

            # Step 3: Run Multi-Stage Pipeline with Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(ratio: float, msg: str):
                progress_bar.progress(ratio)
                status_text.text(msg)

            with st.spinner("Processing paper through NLP pipeline..."):
                analysis_result = pipeline.run_full_analysis(
                    paper_text=cleaned_text,
                    metadata=doc_metadata,
                    progress_callback=update_progress,
                )

            st.session_state.analysis_result = analysis_result
            st.session_state.chat_history = []  # Reset chat on new paper
            progress_bar.empty()
            status_text.empty()
            st.success("✅ Analysis completed successfully!")

        except LLMClientError as e:
            st.error(f"❌ LLM API Error: {e}")
        except Exception as e:
            st.error(f"❌ Unexpected Error: {e}")


# ==============================================================================
# DISPLAY ANALYSIS DASHBOARD (TABS)
# ==============================================================================

result = st.session_state.analysis_result

if result is not None:
    meta = result.metadata
    summary = result.executive_summary
    audit = result.claim_audit
    jargon = result.jargon_glossary

    # Top stats bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document Words", f"{meta.word_count:,}")
    with col2:
        st.metric("Estimated Tokens", f"~{meta.estimated_tokens:,}")
    with col3:
        st.metric("Pipeline Latency", f"{result.execution_time_seconds}s")
    with col4:
        st.metric("Model Deployed", result.model_used)

    if meta.detected_sections:
        sections_str = " • ".join([f"`{s}`" for s in meta.detected_sections])
        st.markdown(f"**Detected Paper Sections:** {sections_str}")

    st.markdown("---")

    # 4 Core Functional Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Executive Digest",
        "🔬 Claims & Methodology Audit",
        "💡 Jargon Decoder",
        "💬 Grounded Q&A Chat",
        "📥 Export Report & Text"
    ])

    # --------------------------------------------------------------------------
    # TAB 1: EXECUTIVE DIGEST
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader(f"📌 {summary.paper_title}")
        
        # TL;DR Highlight Card
        st.markdown(
            f'<div class="tldr-box"><strong>TL;DR:</strong> {summary.tldr}</div>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### ❓ Problem Statement & Gap")
            st.info(summary.problem_statement)

        with col_b:
            st.markdown("### 🛠️ Proposed Methodology")
            st.success(summary.proposed_methodology)

        st.markdown("### 📊 Key Empirical Findings")
        for finding in summary.key_findings:
            st.markdown(f"- 📈 **{finding}**")

        st.markdown("### 🌍 Real-World & Practical Applications")
        app_cols = st.columns(len(summary.practical_applications) or 1)
        for i, app_text in enumerate(summary.practical_applications):
            with app_cols[i % len(app_cols)]:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-weight: 700; color: #6366F1; font-size: 1.02rem; margin-bottom: 0.4rem;">💡 Application {i+1}</div>
                    <div style="line-height: 1.5;">{app_text}</div>
                </div>
                """, unsafe_allow_html=True)


    # --------------------------------------------------------------------------
    # TAB 2: CLAIMS & METHODOLOGY AUDIT
    # --------------------------------------------------------------------------
    with tab2:
        st.subheader("🔬 Empirical Claim Verification & Limitations")
        st.markdown(f"**Methodological Assessment:** {audit.overall_methodology_assessment}")
        st.markdown("---")

        for idx, item in enumerate(audit.claims_audit, 1):
            status = item.status.strip()
            if "verified" in status.lower():
                badge_class = "status-badge-verified"
                badge_icon = "🟢"
            elif "partial" in status.lower():
                badge_class = "status-badge-partial"
                badge_icon = "🟡"
            else:
                badge_class = "status-badge-unsupported"
                badge_icon = "🔴"

            with st.expander(f"{badge_icon} Claim {idx}: {item.claim[:90]}...", expanded=True):
                st.markdown(f"**Claim Assertion:** {item.claim}")
                st.markdown(
                    f'**Verification Rating:** <span class="{badge_class}">{item.status}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Reported Evidence / Metric:** `{item.metric_or_evidence}`")
                st.markdown(f"**Identified Limitations / Constraints:** ⚠️ {item.limitations}")

    # --------------------------------------------------------------------------
    # TAB 3: JARGON DECODER
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("💡 Technical Jargon & Analogies")
        st.markdown("Translates dense academic terminology into intuitive mental models.")

        for term_item in jargon.jargon_list:
            with st.container():
                st.markdown(f"### 🏷️ `{term_item.term}`")
                col_def, col_ana = st.columns([1, 1])
                with col_def:
                    st.markdown("**Formal Academic Definition:**")
                    st.write(term_item.formal_definition)
                    st.caption(f"**Architectural Role:** {term_item.importance}")
                with col_ana:
                    st.markdown("**🎈 Real-World Intuitive Analogy:**")
                    st.markdown(
                        f"""
                        <div style="background-color: rgba(245, 158, 11, 0.12); border-left: 4px solid #F59E0B; padding: 0.9rem; border-radius: 6px; line-height: 1.5;">
                            {term_item.intuitive_analogy}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("---")

    # --------------------------------------------------------------------------
    # TAB 4: GROUNDED Q&A CHAT
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("💬 Ask Grounded Questions About This Paper")
        st.caption("Responses are strictly bounded by the paper text to prevent hallucinations.")

        # Render conversation history
        for q, a in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)

        # Question input box
        user_query = st.chat_input("Ask anything about methodology, metrics, datasets, or limitations...")
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching paper text and citing answers..."):
                    llm_client = LLMClient(
                        provider=selected_provider,
                        api_key=effective_api_key,
                        model_name=selected_model,
                        temperature=0.1,
                    )

                    pipeline = ScholarLensPipeline(llm_client=llm_client)
                    response_text = pipeline.ask_grounded_question(
                        text=st.session_state.current_paper_text,
                        question=user_query
                    )
                    st.write(response_text)
                    st.session_state.chat_history.append((user_query, response_text))

    # --------------------------------------------------------------------------
    # TAB 5: EXPORT & RAW TEXT
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("📥 Export Comprehensive Research Report")
        
        # Build Markdown Report
        markdown_report = f"""# ScholarLens Research Synthesis Report: {summary.paper_title}

**Generated with:** {result.model_used} | **Processing Latency:** {result.execution_time_seconds}s
**Word Count:** {meta.word_count:,} | **Estimated Tokens:** ~{meta.estimated_tokens:,}

---

## 1. Executive Summary
- **TL;DR:** {summary.tldr}
- **Problem Statement:** {summary.problem_statement}
- **Proposed Methodology:** {summary.proposed_methodology}

### Key Benchmark Findings:
"""
        for kf in summary.key_findings:
            markdown_report += f"- {kf}\n"

        markdown_report += "\n### Practical Applications:\n"
        for pa in summary.practical_applications:
            markdown_report += f"- {pa}\n"

        markdown_report += f"\n---\n\n## 2. Claim Verification & Methodology Audit\n**Overall Verdict:** {audit.overall_methodology_assessment}\n\n"
        for idx, item in enumerate(audit.claims_audit, 1):
            markdown_report += f"### Claim {idx}: {item.claim}\n"
            markdown_report += f"- **Status:** {item.status}\n"
            markdown_report += f"- **Reported Metric / Evidence:** {item.metric_or_evidence}\n"
            markdown_report += f"- **Identified Limitations:** {item.limitations}\n\n"

        markdown_report += "---\n\n## 3. Jargon Glossary & Analogies\n"
        for item in jargon.jargon_list:
            markdown_report += f"### {item.term}\n"
            markdown_report += f"- **Formal Definition:** {item.formal_definition}\n"
            markdown_report += f"- **Everyday Analogy:** {item.intuitive_analogy}\n"
            markdown_report += f"- **Importance:** {item.importance}\n\n"

        st.download_button(
            label="💾 Download Synthesis Report (.md)",
            data=markdown_report,
            file_name=f"{meta.filename}_scholar_synthesis.md",
            mime="text/markdown",
            type="primary"
        )

        with st.expander("👁️ View Normalized Paper Text Extracted"):
            st.text_area("Normalized Text", st.session_state.current_paper_text, height=300)

else:
    # Empty State when no paper has been analyzed yet
    st.info(
        "👋 **Welcome to ScholarLens!**\n\n"
        "To get started:\n"
        "1. Select your LLM Provider and enter your API Key in the left sidebar.\n"
        "2. Choose an input source (or simply use the pre-loaded **'Attention Is All You Need'** sample).\n"
        "3. Click **'🚀 Analyze Paper'** to run the multi-stage NLP pipeline!"
    )
