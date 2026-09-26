"""
Golden-set evaluation harness for the text2sql pipeline.

Runs every question in a "golden set" JSON file through text2sql(), then
uses an LLM-as-judge (see JUDGE_SYSTEM below) to grade whether the actual
query result matches the expected ground-truth answer, since exact-string
comparison would be too brittle for raw SQL row output.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage
from pydantic import BaseModel,Field
from dotenv import load_dotenv
import json
from services.text2sql import text2sql
load_dotenv()

# Instructions for the grading LLM: judge strictly against the provided
# expected answer only (never real-world cricket knowledge), tolerate
# formatting/shape differences, and treat raw row data as sufficient
# evidence even without narrative phrasing.
JUDGE_SYSTEM="""You are a strict evaluator for a cricket statistics question-answering system.
You will be given: a QUESTION, an EXPECTED ANSWER (the ground truth), and the ACTUAL RESULT the system produced.
Your ONLY job is to decide whether the ACTUAL RESULT matches the EXPECTED ANSWER.
Critial rules:
- Judge ONLY by comparing the actual result to the expected answer provided.
- DO NOT use any outside cricket knowledge. DO NOT consider whether the expected answer matches real world cricket facts. Treat the
  EXPECTED ANSWER as the sole , absolute ground truth.
- Ignore differences in formatting, shape or wrapping. For example [(6,)] and 6 are the same value. A result with extra columns is fine if the core expected value is present.
- Accept semantically equivalent names (e.g. 'India' and 'India National Team').
- For answers that are lists (e.g. multiple players or matches), the result is correct only if ALL expected items are present; order does not matter. If some are missing, it is incorrect (or partially correct — mark incorrect and say which are missing in the reasoning).
- Be strict on the actual value: a wrong number means incorrect.
- If the actual result is empty or an error, and the expected answer is non-empty, that is incorrect.
- If actual contains more data , but all the expected facts are present in actual, that should be marked as correct. 
- The ACTUAL RESULT is raw query output (rows of values). It will NOT contain narration like "yes, a hat-trick" or "Kohli scored a fifty" — it only contains the underlying data values. Judge it correct if the DATA VALUES needed to answer the question are present in the result, even if the result does not phrase or label them. For example, if the question asks "did X take a hat-trick, who?" and the result contains X's name, that IS a correct "yes, X" answer — do not require the word "hat-trick" to appear. If the question asks for a score and the score value is present among the columns, that is sufficient.
"""

def load_golden_json(path):
    # Simple/low-complexity: loads the golden-set file and returns just its
    # "cases" list (each case = one question + expected answer).
    with open(path) as f:
        data=json.load(f)
    return data["cases"]

class Verdict(BaseModel):
    # Structured-output schema the judge LLM must return.
    is_correct: bool = Field(
        description="True only if the actual result matches the expected answer. False otherwise."
    )
    reasoning: str = Field(
        description="One or two sentences explaining the judgment, referencing the expected vs actual values."
    )

def judge(question,expected_answer,actual_result):
    """
    Ask an LLM to grade one eval case: does the actual SQL result contain
    the data needed to answer the question, given the expected answer as
    ground truth.
    """
    llm=ChatOpenAI(model="gpt-4o").with_structured_output(Verdict)
    messages=[
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(content=(
            f"QUESTION:{question}\n"
            f"EXPECTED ANSWER: {expected_answer}\n"
            f"ACTUAL ANSWER : {actual_result}\n\n"
            f"Is the actual result correct?"
        )),
    ]
    return llm.invoke(messages)


def run_eval(golden_path):
    """
    Run every case in the golden set through text2sql() and judge().

    A case that raises (bad/unsafe SQL, DB error, etc.) is automatically
    marked incorrect without invoking the judge; otherwise the judge LLM
    decides correctness against the expected answer.
    """
    cases=load_golden_json(golden_path)
    results=[]

    for case in cases:
        question=case["question"]
        expected=case["expected_answer"]

        try:
            sql,actual=text2sql(question)
            error=None
        except Exception as e:
            sql,actual,error=None,None,str(e)
        if error is not None:
            verdict_correct=False
            verdict_reason=f"system raised error : {error}"
        else:
            v=judge(question,expected,actual)
            verdict_correct=v.is_correct
            verdict_reason=v.reasoning
        results.append({
            "id":case["id"],
            "question":question,
            "expected":expected,
            "actual":actual,
            "sql":sql,
            "error":error,
            "correct":verdict_correct,
            "reasoning":verdict_reason,
        })
    return results

def report(results):
    # Simple/low-complexity: prints an overall correct/total score, then
    # details for every failing case only (passes aren't printed).
    total=len(results)
    correct=sum(1 for r in results if r["correct"])
    print(f"\n=== EVAL: {correct}/{total} correct ===\n")

    print("--- FAILURES ---")
    for r in results:
        if not r["correct"]:
            print(f"\n[{r['id']}] {r['question']}")
            print(f"  expected: {r['expected']}")
            print(f"  actual:   {r['actual']}")
            print(f"  reason:   {r['reasoning']}")
            if r["error"]:
                print(f"  error:    {r['error']}")
            print(f"  sql:      {r['sql']}")
if __name__=="__main__":
   golden_path='./documents/golden_set/golden_set.json'

   results=run_eval(golden_path)
   report(results)

