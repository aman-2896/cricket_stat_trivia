"""
LangGraph-based cricket assistant agent.

Builds a small ReAct-style graph (llm -> tools -> llm -> ... -> END) that
lets an LLM decide whether to call the stats tool (text2sql), the rules
tool (hybrid retrieval), both, or neither, then loops back until it has a
final answer or hits the recursion limit.
"""

from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph,START,END
from services.text2sql import text2sql
from services.ingest import get_hybrid_retrieval_results
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langgraph.errors import GraphRecursionError

# System prompt that keeps the LLM "grounded": it must only answer using the
# two tools below (T20 WC 2024 stats + cricket rules), and must decline
# (rather than guess from its own training knowledge) anything else,
# including partially-answerable multi-part questions.
GROUNDING_PROMPT="""
You are a cricket assistant who is capable of answering rules, regulations, decorum and concepts of cricket game, also you can provide assistance for the T20 WC 2024. Answer ONLY using the provided tools. If a question cannot be answered by your tools (e.g., it's about cricket administration, other tournaments, or general knowledge), politely say you can only help with T20 WC 2024 stats and rules — do NOT answer from your own knowledge about anything.If a question (or part of one) isn't answerable by the tools, say so — don't fill the gap from memory.For a multi-part question where one part is answerable and one isn't (like 'who had the lowest economy [answerable] and is it good [not]'), answer the answerable part from the tool and decline the rest — rather than answering the good part from memory.
"""
class AgentState(TypedDict):
    # The graph's shared state: just a running message list. `add_messages`
    # is a LangGraph reducer that appends new messages instead of
    # overwriting the list on each node update.
    messages:Annotated[list[AnyMessage],add_messages]

@tool
def get_t20_worldcup_2024_cricket_stats(question:str)->str:
    """Answers statistical questions about the ICC Men's T20 World Cup 2024 —
    counts, totals, averages, records, and player/team numbers — using the
    ball-by-ball match database. Use for questions about numbers and statistics.
    Pass player names exactly as the user wrote them — do not expand names
    (keep 'Kohli' as 'Kohli', not 'Virat Kohli')."""

    sql, result = text2sql(question)
    return str(result)

@tool
def get_cricket_rules_regulations(question:str)->str:
    """
    Answers rules regulations cricket questions as per the Laws of cricket. Covers rules about players, umpires, referees, cricket equipment , ground etc.
    """

    result=get_hybrid_retrieval_results(question)
    return str(result)
tools=[get_t20_worldcup_2024_cricket_stats,get_cricket_rules_regulations]
llm=ChatOpenAI(model="gpt-4o-mini")
# bind_tools lets the LLM emit structured tool-call requests instead of
# just free-text; ToolNode below is what actually executes those calls.
llm_with_tools=llm.bind_tools(tools)

tool_node=ToolNode(tools)

def llm_node(state: AgentState)->dict:
    # Prepends the grounding/system prompt on every turn (it's not stored
    # in state) and asks the LLM for the next message, which may be a
    # final answer or a request to call one/both tools.
    messages=[SystemMessage(content=GROUNDING_PROMPT)]+state["messages"]
    response=llm_with_tools.invoke(messages)
    return {"messages":[response]}

def should_continue(state:AgentState)->str:
    # Routing function for the conditional edge below: if the LLM's last
    # message requested tool calls, go run them; otherwise we have a final
    # answer and the graph can end.
    last_message=state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Graph shape: START -> llm -> (tools -> llm)* -> END
# i.e. a standard ReAct loop: the LLM can keep calling tools and re-reading
# their output until it's ready to answer directly.
graph=StateGraph(AgentState)
graph.add_node("llm",llm_node)
graph.add_node("tools",tool_node)
graph.add_edge(START,"llm")
graph.add_conditional_edges("llm",should_continue)
graph.add_edge("tools","llm")

app=graph.compile()

def run_agent(question: str) -> dict:
    """
    Run one user question through the agent graph to completion.

    Returns a dict with the final answer text, the list of tool names that
    were invoked along the way (used for eval/routing checks — see
    services/agent_eval.py), and the raw message history.
    """
    try:
        # recursion_limit caps how many llm<->tools round-trips can happen,
        # guarding against the LLM getting stuck in a tool-calling loop.
        result = app.invoke({"messages": [{"role": "user", "content": question}]},config={"recursion_limit":8})
        answer = result["messages"][-1].content
        # Flatten every tool_call across every message into a simple list
        # of tool names actually used for this question.
        called_tools = [tc["name"] for m in result["messages"]
                        if getattr(m, "tool_calls", None) for tc in m.tool_calls]
        return {"answer": answer, "called_tools": called_tools, "messages": result["messages"]}
    except GraphRecursionError as e:
        # Hit the recursion_limit above without reaching a final answer —
        # fail gracefully instead of propagating the exception to callers.
        return {"answer": "Sorry, I couldn't complete that request.", "called_tools": [], "messages": []}


if __name__=="__main__":
   query1="how many sixes did Kohli hit in the tournament? what is a no ball"
   query2="who is the chairman of icc?"
   query3="Can you tell me whether the next delivery will be a no-ball?"
   result = run_agent(query1)
   print(result["answer"])
#    print(result["messages"][-1].content)