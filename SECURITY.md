# Security

## Prototype — not hardened for exposure

This blueprint is a **reference prototype with no authentication**. The
API will accept requests from anyone who can reach it.

- Run it locally or inside a network you control.
- **Do not expose it to the public internet** as-is. A production
  deployment needs authenticated access, per-client data isolation, and
  the rest of the hardening listed in `docs/architecture.md` (Section 9).

## Data

- All bundled data is synthetic. Never put real shipment, customer,
  carrier, or policy data into this prototype, and never include real
  data in issues, pull requests, logs, or screenshots.
- The code reads optional LLM API keys from environment variables. The
  application loads a `.env` file from the repo root at startup (real
  environment variables take precedence); `.env.example` lists every
  variable. `.env` is git-ignored — never commit a filled-in one.

## Reporting a concern

If you find a security concern in this repository, please report it
privately through Trida AI's contact route at
https://trida.ai/intake rather than opening a public issue.
