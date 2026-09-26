"""
Alternate implementation of the cricket assistant agent, built directly on
the raw OpenAI SDK function-calling loop instead of LangGraph
(compare with services/agent.py, which is the LangGraph version used by
main.py). Kept for comparison/experimentation.
"""

from openai import OpenAI
import json
from services.text2sql import text2sql
from services.ingest import get_hybrid_retrieval_results

# Tool definitions in raw OpenAI function-calling schema: a list of
# {"type": "function", "function": {name, description, parameters}}
# objects, mirroring the two @tool-decorated functions in services/agent.py
# but described manually since we're not using LangChain here.
tools=[
        {
            "type":"function",
            "function":{
                "name":"get_t20_worldcup_2024_cricket_stats",
                "description":"Answers statistical questions about the ICC Men's T20 World Cup 2024 — counts, totals, averages, records, and player/team numbers — using the ball-by-ball match database. Use for questions about numbers and statistics.",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "question":{
                            "type":"string",
                            "description":"A statistics-related question about the ICC Men's T20 World Cup 2024. Pass player names exactly as the user wrote them — do not expand or complete names (e.g. keep 'Kohli' as 'Kohli', do not change it to 'Virat Kohli')."
                        },
                    },
                    "required":["question"]
                }
            }
        },
        {
            "type":"function",
            "function":{
                "name":"get_cricket_rules_regulations",
                "description":"Answers rules regulations cricket questions as per the Laws of cricket. Covers rules about players, umpires, referees, cricket equipment , ground etc.",
                "parameters":{
                    "type":"object",
                    "properties":{
                        "question":{
                            "type":"string",
                            "description":"A rules , regulations or concept related question about cricket."
                        },
                    },
                    "required":["question"]
                }
            }
        }
    ]

client = OpenAI()
def run_agent(question):
    """
    Raw function-calling loop: send the conversation to the model, and as
    long as it keeps requesting tool calls, execute them and append their
    results as "tool" messages, then ask again — until it returns a
    message with no tool calls, which is the final answer.
    """
    messages=[
        {"role":"system",
         "content":"You are a cricket assistant who is capable of answering rules, regulations, decorum and concepts of cricket game, also you can provide assistance for the T20 WC 2024. Answer ONLY using the provided tools. If a question cannot be answered by your tools (e.g., it's about cricket administration, other tournaments, or general knowledge), politely say you can only help with T20 WC 2024 stats and rules — do NOT answer from your own knowledge about anything.If a question (or part of one) isn't answerable by the tools, say so — don't fill the gap from memory.For a multi-part question where one part is answerable and one isn't (like 'who had the lowest economy [answerable] and is it good [not]'), answer the answerable part from the tool and decline the rest — rather than answering the good part from memory."},
        {"role":"user","content":question}
    ]

    while True:
        response=client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )
        msg=response.choices[0].message
        # right after getting msg:
        print(f"--- LLM response: tool_calls={bool(msg.tool_calls)} ---")
        if not msg.tool_calls:
            #we got final answer

            print("\n=====================================================================\n")
            print(messages)
            print("\n=====================================================================\n")
            return msg.content
        messages.append(msg)

        # The model can request multiple tool calls in one turn (e.g. a
        # question needing both stats and rules) — execute each and add
        # its own "tool" role message before looping back to the model.
        for tool_call in msg.tool_calls:
            fn_name=tool_call.function.name
            args=json.loads(tool_call.function.arguments)

            if fn_name=="get_t20_worldcup_2024_cricket_stats":
                print(f">>> text2sql receiving: {args['question']!r}")
                sql,result=text2sql(args["question"])
                tool_output=str(result)
            elif fn_name=="get_cricket_rules_regulations":
                print(f">>> get_hybrid_retrieval_results receiving: {args['question']!r}")
                result=get_hybrid_retrieval_results(args["question"])
                tool_output=str(result)

            else:
                tool_output=f"Unknown tool {fn_name}"

            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":tool_output
                }
            )


if __name__=="__main__":
    query_true="how many sixes did Kohli hit in the tournament?"
    query_true2="what is the height of the stumps in cricket?"
    query_true3="who had the lowest economy rate in tournament? is it considered good?"
    query_false="who is the chariman of icc?"
    query_true4="How many wickets did Arshdeep Singh take in the tournament, and what does the LBW law say about when a batter is out?"
    result=run_agent(query_true4)
    print(f"--------------------\n{result}\n--------------------------")