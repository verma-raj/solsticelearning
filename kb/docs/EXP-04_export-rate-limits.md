---
doc_id: EXP-04
title: Export frequency limits
topic: data-export-limits
last_updated: 2026-09-30
---

# Export frequency limits

To protect platform performance, Solstice limits how often exports can be run:

- **10 instant exports per user per hour**
- **5 concurrent bulk exports per organization**

If you hit a limit you will see "Export limit reached". Instant export limits reset on a rolling hourly basis. For bulk exports, wait for a running request to finish before submitting another.
