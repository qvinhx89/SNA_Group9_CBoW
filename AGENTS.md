---
description: 
alwaysApply: true
---

# Code review graph (code-review-graph)

This repo uses **code-review-graph** (local graph under `.code-review-graph/`). **GitNexus MCP is disabled** — do not call GitNexus tools or `gitnexus://` resources.

> If tools fail or the graph is missing, run from repo root: `code-review-graph build` (or `code-review-graph update`). On **Cursor**, hooks do not refresh the graph; use **`crg-daemon`** or run `update` / `build` after large changes.

## Always do

- **Start token-light:** call **`get_minimal_context_tool`** first with a short `task` description; use minimal detail until you need more.
- **Before editing** a function/class/method: use **`query_graph_tool`** / **`get_impact_radius_tool`** (and **`get_review_context_tool`** if you need snippets) so callers and blast radius are known.
- **Before commit:** run **`detect_changes_tool`** and summarize risk and affected flows.
- **Warn the user** on high-risk `detect_changes` output before large edits.

## Never do

- Do not rely on **GitNexus** MCP, `gitnexus_*` calls, or `gitnexus://` URIs.
- Do not **rename** across files with blind find-and-replace — use **`refactor_tool`** (preview) / graph-aware steps.
- Do not explore only with Grep/Read when the graph can answer structure, impact, or tests.

## MCP tools (code-review-graph)

Use the names your MCP server lists (often `*_tool`). Common set:

| Tool | Use when |
|------|----------|
| `get_minimal_context_tool` | First call for any task — tiny context |
| `build_or_update_graph_tool` | No graph yet or after major refactors |
| `list_graph_stats_tool` | Health / size of index |
| `query_graph_tool` | Callers, callees, imports, tests |
| `semantic_search_nodes_tool` | Find symbols by name / keyword |
| `get_impact_radius_tool` | Blast radius |
| `get_review_context_tool` | Token-efficient review snippets |
| `get_affected_flows_tool` | Execution paths impacted by changes |
| `detect_changes_tool` | Risk-scored diff analysis |
| `get_architecture_overview_tool` | High-level structure |
| `list_communities_tool` / `get_community_tool` | Module-style clusters |
| `refactor_tool` | Rename preview, dead code hints |

Fall back to Grep/Glob/Read only when the graph does not cover the need.

If a call fails with “unknown tool”, use the **exact tool names** listed under Cursor → MCP for the `code-review-graph` server (some builds use a `_tool` suffix, others may not).

## Skills (workflows)

| Task | File |
|------|------|
| Explore / architecture | `.claude/skills/explore-codebase.md` |
| Review / PR mindset | `.claude/skills/review-changes.md` |
| Debug | `.claude/skills/debug-issue.md` |
| Refactor safely | `.claude/skills/refactor-safely.md` |

Optional prompts from CRG (if exposed): `review_changes`, `architecture_map`, `debug_issue`, `onboard_developer`, `pre_merge_check`.
