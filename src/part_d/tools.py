import os
import json
import sqlite3 as sql
from datetime import datetime
from .rag import retrieve


DB_PATH = "data/instamart.db"


def get_order(order_id):
    conn = sql.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT order_id, status, order_amount, delivery_minutes "
            "FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": None if row[3] is None else float(row[3]),
    }

def search_policy(query):
    return {"results": retrieve(query)}

# Policy rule: Only completed orders delivered in >45 minutes qualify for late-delivery refund (100% amount)
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or delivery_minutes <= 45:
        return {"refund_inr": 0}
    return {"refund_inr": round(float(order_amount), 2)}

LEDGER_PATH = "outputs/refund_ledger.jsonl"

def _log(tool, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def issue_refund(order_id, refund_inr):
    _log("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}
# Automatic refund issuance threshold is INR 1000; amounts strictly above INR 1000 must escalate to human review
def escalate_to_human(order_id, refund_inr):
    _log("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}

TOOLS_SPEC = [
        {"name": "get_order",
         "description": "Look up an order by id and return its status, amount and delivery time.",
         "parameters": {"type": "object",
                        "properties": {"order_id": {"type": "string"}},
                        "required": ["order_id"]}},
         {"name": "search_policy",
          "description": "Search the policy documents and return the most relevant passages.",
          "parameters": {"type": "object",
                         "properties": {"query": {"type": "string"}},
                         "required": ["query"]}},
          {"name": "calculate_refund",
           "description": "Compute the refund in INR from the order amount and delivery minutes.",
           "parameters": {"type": "object",
                          "properties": {"order_amount": {"type": "number"},
                                         "delivery_minutes": {"type": "number"}},
                          "required": ["order_amount", "delivery_minutes"]}},
           {"name": "issue_refund",
            "description": "Issue an automatic refund of up to INR 1000 and record it in the ledger.",
            "parameters": {"type": "object",
                           "properties": {"order_id": {"type": "string"},
                                          "refund_inr": {"type": "number"}},
                           "required": ["order_id", "refund_inr"]}},
            {"name": "escalate_to_human",
             "description": "Send a refund above INR 1000 to a human support lead and record it.",
             "parameters": {"type": "object",
                            "properties": {"order_id": {"type": "string"},
                                           "refund_inr": {"type": "number"}},
                            "required": ["order_id", "refund_inr"]}},
             ]

TOOL_MAP = {
        "get_order": get_order,
        "search_policy": search_policy,
        "calculate_refund": calculate_refund,
        "issue_refund": issue_refund,
        "escalate_to_human": escalate_to_human,
        }
assert set(TOOL_MAP) == {t["name"] for t in TOOLS_SPEC}

if __name__ == "__main__":
    print(get_order("ORD-999999"))
    print(get_order("ORD-100004"))
    print(calculate_refund(285.0, 50))
    print(calculate_refund(285.0, 45))
    print(calculate_refund(285.0, None))
