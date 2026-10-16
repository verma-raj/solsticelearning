---
doc_id: PWD-09
title: What happens to my signed-in sessions after a reset?
topic: password-reset
last_updated: 2026-09-30
---

# What happens to my signed-in sessions after a reset?

When a password reset is completed, Solstice ends all active sessions for that account on the web and in Solstice Desktop. Sessions are closed within **5 minutes**, and you will need to sign in again everywhere.

API tokens and integration keys are **not** affected by a password reset. To revoke them, an admin must do so separately under Admin Console > Integrations.
