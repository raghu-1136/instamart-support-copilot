import json
import os
import time
from mock_llm import chat
from src.part_d.tools import TOOLS_SPEC, TOOL_MAP

SYSTEM = "You are a refund AGENT. Use the tools to decide the refund request."
STATE_DIR = "state"


class SimulatedCrash(Exception):
    """Raised on purpose to test resume."""


def _fallback(status, scenario_id, steps, llm_calls, tokens_used, reason):
    done = ", ".join(s["tool"] for s in steps) or "no tools"
    summary = (f"{status}: handing scenario {scenario_id} to a human agent. "
               f"{reason}. Steps completed: {done}.")
    return {"status": status, "final_answer": summary, "steps": steps,
            "llm_calls": llm_calls, "tokens_used": tokens_used,
            "executed_tools": [s["tool"] for s in steps]}


def _save_checkpoint(run_id, scenario_id, messages, tokens_used,
                     llm_calls, steps, executed):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(f"{STATE_DIR}/{run_id}.json", "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "scenario_id": scenario_id,
            "messages": messages,
            "tokens_used": tokens_used,
            "llm_calls": llm_calls,
            "step": len(steps),
            "executed_tools": executed,
            "steps": steps,
        }, f)


def _loop(run_id, scenario_id, messages, steps, tokens_used, llm_calls,
          executed, max_iterations, token_limit, time_limit_seconds,
          crash_after_tools):
    start = time.monotonic()

    while llm_calls < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return _fallback("FALLBACK_TIMEOUT", scenario_id, steps, llm_calls,
                             tokens_used, f"Time limit of {time_limit_seconds}s reached")

        response = chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        if tokens_used > token_limit:
            return _fallback("FALLBACK_TOKEN_LIMIT", scenario_id, steps, llm_calls,
                             tokens_used, f"Token limit of {token_limit} exceeded")

        msg = response.choices[0].message
        if not msg.tool_calls:
            final = msg.content
            return {"status": final.split(":")[0], "final_answer": final,
                    "steps": steps, "llm_calls": llm_calls,
                    "tokens_used": tokens_used, "executed_tools": executed}

        tc = msg.tool_calls[0]
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name,
                                         "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "name": tc.name, "content": json.dumps(result)})
        steps.append({"tool": tc.name, "arguments": tc.arguments,
                      "observation": result})
        executed.append(tc.name)

        _save_checkpoint(run_id, scenario_id, messages, tokens_used,
                         llm_calls, steps, executed)
        if crash_after_tools is not None and len(executed) >= crash_after_tools:
            raise SimulatedCrash(f"simulated crash after {len(executed)} tools")

    return _fallback("FALLBACK_MAX_ITERATIONS", scenario_id, steps, llm_calls,
                     tokens_used, f"No final answer after {max_iterations} model calls")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tools=None):
    run_id = run_id or f"{scenario_id}_{int(time.time() * 1000)}"
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message}]
    return _loop(run_id, scenario_id, messages, [], 0, 0, [],
                 max_iterations, token_limit, time_limit_seconds,
                 crash_after_tools)


def resume_agent(run_id, max_iterations=6, token_limit=6000,
                 time_limit_seconds=10):
    with open(f"{STATE_DIR}/{run_id}.json", encoding="utf-8") as f:
        cp = json.load(f)
    return _loop(run_id, cp["scenario_id"], cp["messages"], cp["steps"],
                 cp["tokens_used"], cp["llm_calls"], cp["executed_tools"],
                 max_iterations, token_limit, time_limit_seconds, None)
