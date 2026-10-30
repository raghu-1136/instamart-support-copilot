import json
import shutil
import os
import pandas as pd
from src.part_d.agent import run_agent
from src.part_d.agent import SimulatedCrash, resume_agent

shutil.rmtree("state", ignore_errors=True)
RUNS_PATH = "outputs/agent_runs.jsonl"
LEDGER_PATH = "outputs/refund_ledger.jsonl"

os.makedirs("outputs", exist_ok=True)
for p in (RUNS_PATH, LEDGER_PATH):
    if os.path.exists(p):
        os.remove(p)

sc = pd.read_csv("data/raw/agent_scenarios.csv")
s1 = sc.loc[0, "message"]

print("| scenario | status | refund_inr | model calls | tokens |")
for _, row in sc.iterrows():
    r = run_agent(row["message"], row["scenario_id"])
    refund = next((s["observation"]["refund_inr"] for s in r["steps"]
                   if s["tool"] == "calculate_refund"), 0)

    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "scenario_id": row["scenario_id"],
            "status": r["status"],
            "final_answer": r["final_answer"],
            "llm_calls": r["llm_calls"],
            "tokens_used": r["tokens_used"],
            "steps": r["steps"],
        }) + "\n")

    print(f"| {row['scenario_id']} | {r['status']} | {refund} | {r['llm_calls']} | {r['tokens_used']} |")

    assert r["status"].startswith(row["expected_outcome"]), row["scenario_id"]
    assert abs(refund - row["expected_refund_inr"]) < 0.01, row["scenario_id"]
print("S1 token_limit=200:", run_agent(s1, "S1", token_limit=200)["status"])
print("S1 time_limit_seconds=0:", run_agent(s1, "S1", time_limit_seconds=0)["status"])

try:
    run_agent(s1, "S1", run_id="S1_demo", crash_after_tools=2)
except SimulatedCrash as e:
    print("crash:", e)

r = resume_agent("S1_demo")
print("resume:", r["status"], r["executed_tools"])
refund = next(s["observation"]["refund_inr"] for s in r["steps"]
              if s["tool"] == "calculate_refund")
print("refund:", refund)

assert r["executed_tools"] == ["get_order", "search_policy",
                               "calculate_refund", "issue_refund"]
assert r["status"] == "REFUND" and abs(refund - 128.25) < 0.01
