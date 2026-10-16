---
doc_id: EXP-07
title: Exporting data through the API
topic: data-export-limits
last_updated: 2026-09-30
---

# Exporting data through the API

The Solstice Reporting API lets you pull the same data programmatically.

- Rate limit: **1,000 requests per hour** per organization API token
- Page size: up to **500 records** per request
- Use the cursor value returned with each page to fetch the next page. There is no cap on the total number of records you can page through.

Requests over the rate limit return HTTP 429. Wait for the hourly window to roll over before retrying.
