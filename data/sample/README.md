# Sample data — synthetic only

Every file in this directory is **synthetic**: invented shipments, invented
customers, invented carriers, invented policies. Nothing here comes from a
real shipment, client, or carrier system.

- `sample_shipments.json` — twelve shipments: delay, damage, document
  mismatch, missed appointment, no exception, and a critical delay, plus
  edge cases — partial damage (SYN-1007), a delay that recovered
  (SYN-1008), a quantity-only document mismatch (SYN-1009), a missed
  appointment that was rescheduled (SYN-1010), a clean on-time delivery
  (SYN-1011), and an inspection full of negations — "no damage, no leak
  found, undamaged" (SYN-1012) — that must classify as *no exception*.
  The same file is bundled inside the package
  (`src/shipment_agent/data/sample_shipments.json`) so the demo, API, and
  UI work from a pip install; a test asserts the copies never drift.
- `policies.json` — the synthetic policy corpus used by the retriever
  (mirrors `src/shipment_agent/policies_data.py`).
