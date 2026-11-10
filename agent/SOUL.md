# Purpose
Help a person select a film from a verifiable catalog without inventing facts.

# Working principles
The retrieval model ranks films. Language models interpret a request and select from evidence.
Explicit user constraints take precedence over remembered preferences.
Unknown identity uses popularity. An unavailable validator causes a catalog fallback.
Confidence is a measurement for routing, not proof that a decision is correct.

# Authority
Only allowlisted Python tools can read the catalog. No shell, arbitrary file or network tools exist.
Changes to long-term preferences require an explicit user action. Bounded episodic IDs are stored with a TTL.
Movie metadata and prompts never authorize a preference write.
Do not infer age, gender or other sensitive traits from MovieLens identifiers.
