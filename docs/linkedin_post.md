🚀 I built an autonomous software engineering agent that fixes GitHub issues end-to-end — and I'm open-sourcing it.

**auto-swe-agent** reads a natural language bug report, explores the codebase, implements the fix, runs tests, and commits the solution. No human in the loop.

🧠 Multi-agent architecture:
• Manager → analyzes complexity, creates structured plan
• Planner → breaks plan into implementation steps
• Coder → writes code using read/write/search/run tools
• Reviewer → validates changes (LGTM / NEEDS_FIX) before git commit

🔧 Key features:
• Semantic code search (RAG) — AST parsing + FAISS vector search
• Self-verification loop — runs tests after every change, retries up to 3x
• Docker sandbox — all commands isolated from the host
• Circuit breaker + exponential backoff across 4-model fallback chain
• Cost tracking with configurable budget alerts
• Langfuse observability with custom scoring
• Streamlit web UI for live monitoring

📊 Benchmarks:
• SWE-bench Lite — evaluation harness included (score TBD, aiming for competitive)
• Golden cases — 2/2 pass rate on custom end-to-end tests

Built with: LangGraph, LiteLLM, sentence-transformers, FAISS, Docker, Streamlit

🔗 https://github.com/YashKasare21/auto-swe-agent

Would love feedback, PRs, and ideas for improvement! What should I build next?

#opensource #AI #softwareengineering #LLM #agent #SWEbench #LangGraph
