"""
Tool-routing eval for the LangGraph agent (services/agent.py).

Checks that the agent calls the *right* tool(s) for a fixed set of
questions — the stats tool, the rules tool, both, or neither — rather than
checking the quality of the final answer text (that's what services/eval.py
does for text2sql specifically).
"""

from services.agent import run_agent

# Each case pairs a question with the set of tool names the agent is
# expected to call. `may_use_then_abort` marks a case where calling a tool
# and then declining to use its result is also acceptable.
routing_cases = [
    # 1. Use get_t20_worldcup_2024_cricket_stats — 3
    {
        "question": "How many sixes did Virat Kohli hit in the T20 World Cup 2024?",
        "expected_tools": {"get_t20_worldcup_2024_cricket_stats"}
    },
    {
        "question": "How many wickets did Arshdeep Singh take in the T20 World Cup 2024?",
        "expected_tools": {"get_t20_worldcup_2024_cricket_stats"}
    },
    {
        "question": "What was Rohit Sharma's total score in the T20 World Cup 2024?",
        "expected_tools": {"get_t20_worldcup_2024_cricket_stats"}
    },

    # 2. Use get_cricket_rules_regulations — 3
    {
        "question": "What is the LBW law in cricket?",
        "expected_tools": {"get_cricket_rules_regulations"}
    },
    {
        "question": "What are the official dimensions of cricket stumps?",
        "expected_tools": {"get_cricket_rules_regulations"}
    },
    {
        "question": "What are the rules for a no-ball in cricket?",
        "expected_tools": {"get_cricket_rules_regulations"}
    },

    # 3. Use both tools — 2
    {
        "question": "How many wickets did Arshdeep Singh take in the T20 World Cup 2024, and what are the official dimensions of the stumps?",
        "expected_tools": {
            "get_t20_worldcup_2024_cricket_stats",
            "get_cricket_rules_regulations"
        }
    },
    {
        "question": "How many runs did Virat Kohli score in the T20 World Cup 2024, and what are the rules for an LBW dismissal?",
        "expected_tools": {
            "get_t20_worldcup_2024_cricket_stats",
            "get_cricket_rules_regulations"
        }
    },

    # 4. Use neither tool — 1
    {
        "question": "Who is the chairman of the ICC?",
        "expected_tools": set()
    },

    # 5. May or may not use a tool — 1
    {
        "question": "Can you tell me whether the next delivery will be a no-ball?",
        "expected_tools": set(),
        "may_use_then_abort": True
    },
]

if __name__=="__main__":
    # Simple/low-complexity: runs every case through the agent and compares
    # the set of tools actually called against the expected set, printing
    # a mismatch line for each miss and a final score summary.
    correct_hits=0
    total_hits=len(routing_cases)
    for case in routing_cases:
        result=run_agent(case["question"])

        if set(case["expected_tools"])==set(result["called_tools"]):
            correct_hits+=1
    
        else:
            print(f"for question {case['question']} expected tool calls were \n\n {case['expected_tools']} \n\n agent tool return {result['called_tools']}")

    print(f"total correct results => {correct_hits}/{total_hits}")