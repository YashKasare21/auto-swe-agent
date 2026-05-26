1/ 🚀 I built an autonomous coding agent that fixes GitHub issues end-to-end — and it's open source!

auto-swe-agent reads a bug report, explores the codebase, writes the fix, runs tests, and commits. Zero human intervention.

GitHub: https://github.com/YashKasare21/auto-swe-agent

2/ 🧠 Multi-agent architecture:
Manager → Planner → Coder → Reviewer

Each agent uses a different model:
• Manager → flash-lite (complexity analysis)
• Coder → flash (implementation)
• Reviewer → 70B (code review)

3/ 🔎 Semantic code search (RAG):
AST parsing → sentence-transformers → FAISS vector search

The agent finds code by meaning, not just keywords. Auto-builds index on first run with staleness checks. No GPU needed.

4/ 🛡️ Resilience:
4-model fallback chain with exponential backoff + circuit breaker
gemini-2.0-flash → flash-lite → llama-3.3-70b → llama3-8b

If one model rate-limits, the next takes over seamlessly.

5/ 🧪 Self-verification loop:
Code → test → fix → verify (up to 3 retries)
Reviewer validates before git commit
Docker sandbox keeps everything isolated

6/ 📊 Observability with Langfuse:
• Agent spans for each role
• LLM generation traces with token usage
• Custom scoring: tests_passed, review_quality, search_efficiency
• Routing and tool execution traces

7/ 🖥️ Streamlit UI for live monitoring:
See the agent work in real-time — current agent, cost, circuit status, graph visualization.

8/ 📈 SWE-bench Lite harness included:
```bash
python -m swe_bench.run_swe_bench --num-tasks 10
```
Evaluates against 300 real GitHub issues from 12 Python repos. Report generator included.

9/ 🎯 Try it yourself:
```bash
git clone https://github.com/YashKasare21/auto-swe-agent.git
cd auto-swe-agent
pip install -r requirements.txt
python agent.py "Your bug description" --workspace ./
```

10/ This is an early release. PRs, issues, and ideas welcome! What should I build next?

#buildinpublic #opensource #AI #coding #softwareengineering #LLM
