"""
MCP (Model Context Protocol) server that exposes the cricket stats/rules
tools over the FastMCP protocol, so any MCP-compatible client (e.g. Claude
Desktop) can call them directly instead of going through the FastAPI /
LangGraph agent in main.py / services/agent.py.
"""

import sys
from pathlib import Path

# Simple/low-complexity: makes sure the project root (one level up from
# this mcp/ package) is importable, so `from services...` below works
# whether this script is run directly or via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastmcp import FastMCP
from services.ingest import get_hybrid_retrieval_results
from services.text2sql import text2sql

mcp = FastMCP("cricket")

@mcp.tool
def get_cricket_stats(question:str)->str:
    """
        Answer statistical questions about the ICC Men's T20 World Cup 2024 -
        counts, totals, averages, records, player/team numbers. Pass player names
        exactly as written (keep 'Kohli' as 'Kohli')

    """
    # Delegates to the text-to-SQL pipeline; we only return the query
    # results here, not the generated SQL itself.
    sql,result=text2sql(question)
    return str(result)

@mcp.tool
def get_cricket_rules(question:str)->str:
    """
    Answers cricket rules/laws/regulations questions per the Laws of Cricket
    (players, umpires, equipment, ground, dimentions etc.).
    """
    # Delegates to the hybrid (vector + BM25) retrieval pipeline over the
    # ingested Laws of Cricket document.
    return get_hybrid_retrieval_results(question)

if __name__=="__main__":
    # Starts the MCP server over stdio (FastMCP's default transport),
    # blocking until the client disconnects.
    mcp.run()
