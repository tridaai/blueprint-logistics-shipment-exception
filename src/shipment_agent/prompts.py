"""Prompt assets.

The mock backend renders drafts from deterministic templates (see
``model_backends.MockModelBackend``). These prompts are what the optional
LLM backends receive, kept in one place so they can be reviewed, versioned,
and evaluated like any other code.
"""

DRAFT_SYSTEM_PROMPT = """\
You are a logistics operations assistant drafting a customer update about a
shipment exception. Rules:
- State what happened, what is being done, and when the next update arrives.
- Cite the policy IDs provided in brackets, e.g. [POL-DELAY-01].
- Never promise compensation, a refund, or a guaranteed delivery time.
- Keep it under 180 words. Plain, factual, professional tone.
- The draft is for human review. Do not claim anything was already sent.
"""

DRAFT_USER_TEMPLATE = """\
Shipment: {shipment_id}
Route: {origin} -> {destination} (carrier: {carrier})
Exception: {exception_type} (severity: {severity})
What happened: {rationale}
Signals: {signals}
Delay hours: {delay_hours}
Document mismatches: {mismatches}
Relevant policies:
{policies}

Write the customer update now, with the subject line on the first line
prefixed by "Subject: ".
"""
