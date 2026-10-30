import re
import os
import json
import random, time
import mock_llm
import pandas as pd
from mock_llm import chat, reset, TransientLLMError,est_tokens

tickets = pd.read_csv("data/raw/tickets.csv",keep_default_na = False)

MAX_INPUT_TOKENS = 400

def check_input(ticket):
    """Return a status string if the ticket must not reach the model, else None."""
    if not ticket.strip():
        return "skipped_empty"
    if est_tokens(ticket) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None

tickets["guard_status"] = tickets["text"].apply(check_input)

print("Guardrail status counts: ",tickets["guard_status"].value_counts(dropna=False))

print(tickets.loc[tickets["guard_status"] == "flagged_long", "text"].apply(est_tokens).unique())


EMAIL_RE = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_RE = r"\b[6-9]\d{9}\b"
ORDER_RE = r"ORD-\d{6}"

# Redact PII in strict sequence (Email -> Phone -> Order ID) before prompt construction to guarantee data privacy
def redact(text):
    text, n_email = re.subn(EMAIL_RE, "[EMAIL]", text)
    text, n_phone = re.subn(PHONE_RE, "[PHONE]", text)
    text, n_order = re.subn(ORDER_RE, "[ORDER_ID]", text)
    return text, n_email, n_phone, n_order

results = tickets["text"].apply(redact)
tickets["redacted"] = results.apply(lambda r: r[0])

n_email = results.apply(lambda r: r[1]).sum()
n_phone = results.apply(lambda r: r[2]).sum()
n_order = results.apply(lambda r: r[3]).sum()

print("emails:", n_email, "phones:", n_phone, "order ids:", n_order,
      "total:", n_email + n_phone + n_order)

left = sum(len(re.findall(p, t))
           for t in tickets["redacted"]
           for p in (EMAIL_RE, PHONE_RE, ORDER_RE))
print("left after redaction:", left)
assert left == 0

ZERO_SHOT = (
    "You are a customer support assistant. "
    "CLASSIFY this customer-support ticket. "

    "Choose one: "
    "billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.'
)

FEW_SHOT = (
    "You are a customer support assistant. "
    "CLASSIFY this customer-support ticket. "

    "Choose one: "
    "billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    'Ticket: There is an unauthorized transaction on my card for [ORDER_ID]. I did not place it. -> {"category": "fraud", "reason": "unauthorized transaction"}\n'
    'Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}\n'
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
    'Ticket: My order has not arrived and the rider is not picking up. -> {"category": "delivery", "reason": "order not arrived"}'
)

for p in (ZERO_SHOT, FEW_SHOT):
    assert "CLASSIFY" in p
    assert all(c in p for c in ["billing", "delivery", "technical", "fraud"])
assert "EXAMPLES:" not in ZERO_SHOT
assert "EXAMPLES:" in FEW_SHOT
print(est_tokens(ZERO_SHOT), est_tokens(FEW_SHOT))


VALID_CATEGORIES = {"billing", "fraud", "technical", "delivery"}

def validate_output(text):
    """Return (status, category). category is None when flagged."""
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("category") in VALID_CATEGORIES:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", text.lower()))
        found = words & VALID_CATEGORIES
        if len(found) == 1:
            return "classified_keyword", next(iter(found))
        return "flagged_invalid_output", None

assert validate_output('{"category": "billing", "reason": "x"}') == ("classified_json", "billing")
assert validate_output("The customer appears to have a fraud-related problem.") == ("classified_keyword", "fraud")
assert validate_output('{"category": "shipping", "reason": "x"}') == ("flagged_invalid_output", None)

MAX_RETRIES = 3
BASE_DELAY = 0.05
transient_errors = 0

def call_with_retry(messages, ticket_id,verbose = True):
    global transient_errors
    n = 0
    while True:
        try:
            return chat(messages)
        except TransientLLMError:
            transient_errors += 1
            if n >= MAX_RETRIES:
                raise                       # Exponential backoff with random uniform jitter (0.9 to 1.15) to prevent thundering herd during transient LLM failures
            wait = BASE_DELAY * 2**n * random.uniform(0.9, 1.15)
            if verbose:
                    print(f"retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)
            n += 1

def run_pipeline(prompt, token_budget = None,verbose = True):
    global transient_errors
    reset()
    transient_errors = 0

    rows = []
    total_tokens = 0

    for _, t in tickets.iterrows():
        row = {"ticket_id": t["ticket_id"], "customer_id": t["customer_id"],
               "status": None, "category": None, "tokens_used": 0}

        # 1. input guardrails: no model call
        guard = check_input(t["text"])
        if guard is not None:
            row["status"] = guard
            rows.append(row)
            continue

        # 2. budget check: no model call
        if token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
            rows.append(row)
            continue

        # 3. redact, build messages, call
        redacted_text = redact(t["text"])[0]
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": redacted_text}]

        for p in (EMAIL_RE, PHONE_RE, ORDER_RE):
            assert not re.search(p, messages[1]["content"]), "PII reached the prompt"

        response = call_with_retry(messages, t["ticket_id"],verbose = verbose)
        used = response.usage.total_tokens
        total_tokens += used

        # 4. validate the reply
        status, category = validate_output(response.choices[0].message.content)
        row.update(status=status, category=category, tokens_used=used)
        rows.append(row)

    return pd.DataFrame(rows)


zs = run_pipeline(ZERO_SHOT)

counts = zs["status"].value_counts()
assert counts["classified_json"] == 135 and counts["classified_keyword"] == 12
assert counts["flagged_invalid_output"] == 9
assert transient_errors == 19

fs = run_pipeline(FEW_SHOT, verbose = False)

truth = tickets[["ticket_id", "true_category"]]

def summarise(res):
    ok = res[res["status"].isin(["classified_json", "classified_keyword"])].merge(truth, on="ticket_id")
    correct = (ok["category"] == ok["true_category"]).sum()
    total = res["tokens_used"].sum()
    return len(ok), correct, round(correct / len(ok) * 100, 1), total, round(total / len(ok), 1)

print("zero-shot:", summarise(zs))
print("few-shot: ", summarise(fs))

acc_zs, acc_fs = summarise(zs)[2], summarise(fs)[2]
assert abs(acc_zs - 85.7) <= 0.5 and abs(acc_fs - 100) <= 0.5

ok_zs = zs[zs["status"].isin(["classified_json", "classified_keyword"])].merge(truth, on="ticket_id")
print(pd.crosstab(ok_zs["true_category"], ok_zs["category"]))

os.makedirs("outputs", exist_ok=True)
demo = run_pipeline(ZERO_SHOT, token_budget=6000,verbose = False)

n_exhausted = (demo["status"] == "budget_exhausted").sum()
print("Budget demo (budget 6000): exhausted =", n_exhausted)
assert n_exhausted >= 1
demo.to_csv("outputs/triage_budget_demo.csv", index=False)

print(demo["status"].value_counts())

final = run_pipeline(FEW_SHOT, token_budget=100000, verbose = False)

seg = pd.read_csv("outputs/customer_segments.csv")[["customer_id", "segment_name", "is_premium"]]
final = final.merge(seg, on="customer_id", how="left")
final["is_premium"] = final["is_premium"].fillna(False).astype(bool)

HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}

# Route tickets by safety priority: Edge/Guardrail cases -> Human Review; Fraud -> Security; Premium -> VIP Desk; Others -> Standard Queue
def route(row):
    if row["status"] in HUMAN:
        return "human_review", "NORMAL"
    if row["category"] == "fraud":
        return "security_team", "HIGH"
    if row["is_premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

final[["route", "priority"]] = final.apply(lambda r: pd.Series(route(r)), axis=1)

cols = ["ticket_id", "customer_id", "status", "category", "segment_name",
        "priority", "route", "tokens_used"]
assert len(final) == 164
final[cols].to_csv("outputs/triage_results.csv", index=False)
print(len(final), final["route"].value_counts())
