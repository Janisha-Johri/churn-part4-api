# Monitoring Plan: D2C Churn Scoring API

This plan covers what should be tracked after this model/API is deployed for use by the CRM tool.

---

## 1. Data Drift

**What to monitor:** The distribution of incoming feature values (especially the top model
drivers identified in Part 3: `recency_days`, `negative_ticket_rate_90d`, `frequency_180d`,
`monetary_180d`) compared to the training data distribution.

**How:**
- Log summary statistics (mean, median, percentiles) of each numeric feature for every batch of
  `/predict` and `/batch_predict` calls, on a rolling weekly basis.
- Compare against the training-set baseline distribution (available from `rfm_modeling_snapshot.csv`
  in Part 3) using a simple statistical test (e.g., Population Stability Index or
  Kolmogorov-Smirnov test) per feature.
- **Alert threshold:** Flag any feature where the weekly distribution differs from the training
  baseline by more than a moderate PSI threshold (commonly PSI > 0.2 indicates significant drift)
  for manual review.

**Why it matters:** If, for example, the brand's customer base shifts (new acquisition channel
mix, seasonal buying pattern), the model's learned relationships may no longer hold, even if its
code runs without errors.

---

## 2. Prediction Distribution

**What to monitor:** The distribution of `churn_probability` scores returned by the API over time,
and the proportion of customers classified as `predicted_class = 1` at the current threshold (0.40).

**How:**
- Log every prediction's probability score and predicted class.
- Track the weekly/monthly average probability and the % of customers flagged as at-risk.
- **Alert threshold:** A sudden jump (e.g., the at-risk percentage doubling week-over-week) likely
  indicates either a real business event (e.g., a price increase) or a data pipeline issue feeding
  bad inputs to the API -- both warrant investigation before campaigns act on the scores.

---

## 3. Business Outcomes

**What to monitor:** Actual churn outcomes for customers who were scored, once enough time has
passed to observe their real 60-day behavior (the same target window used in training).

**How:**
- Periodically (e.g., monthly) join scored customers against actual subsequent purchase behavior.
- Recompute live ROC-AUC, precision, and recall on this real-world outcome data, and compare
  against the Part 3 test-set baseline (ROC-AUC 0.868, Precision 0.767, Recall 0.863).
- Track retention-campaign ROI by segment (tying back to Part 2's segment-budget logic) -- i.e.,
  did customers who received a model-triggered intervention actually retain at a higher rate than
  a comparable untreated group?

**Alert threshold:** If live ROC-AUC drops more than ~0.05 below the Part 3 baseline (i.e., below
~0.82), this signals the model's real-world performance has meaningfully degraded.

---

## 4. API Errors

**What to monitor:** HTTP error rates and types returned by the API.

**How:**
- Log all non-200 responses with their status code and the `detail` message (validation errors
  return 422 from Pydantic; model-not-loaded or prediction failures return 503/500 per
  `app/main.py`).
- Track the rate of 422 validation errors specifically -- a sudden spike may indicate the CRM tool
  upstream changed its data format or is sending malformed/incomplete customer records.
- Track latency (p50/p95 response time) for `/predict` and `/batch_predict`, especially as batch
  sizes grow (current limit: 500 customers per batch request).

**Alert threshold:** Error rate > 1% of total requests over a rolling 1-hour window, or p95
latency exceeding an agreed SLA (e.g., 2 seconds for `/predict`).

---

## 5. Retraining Triggers

The model should be retrained when any of the following occur:

1. **Performance degradation:** Live ROC-AUC on actual outcomes drops more than 0.05 below the
   Part 3 baseline.
2. **Significant data drift:** Multiple key features (Section 1) show PSI > 0.2 sustained over
   several weeks, not just a single noisy week.
3. **Business changes:** A new product category, pricing structure, acquisition channel, or
   market (e.g., new city tier) is introduced that wasn't represented in the original training
   data.
4. **Scheduled cadence:** Even absent triggers, retrain on a fixed cadence (e.g., quarterly) using
   a fresh snapshot date, since customer behavior naturally evolves and the original snapshot
   (2025-09-30) will become increasingly stale.
5. **Manual override patterns:** If the CRM/retention team frequently overrides or ignores the
   model's `risk_level` for a particular customer segment (logged via their workflow tool), this
   is a signal the model may be systematically wrong for that segment and needs investigation.

---

## 6. Practical Implementation Notes

- Logging can start simply (structured JSON logs of each request/response written to a file or
  basic logging service) and doesn't require a dedicated MLOps platform on day one.
- Drift and outcome-performance checks can run as a scheduled batch job (e.g., weekly cron) rather
  than real-time, since churn is a 60-day-window phenomenon, not a real-time event.
- Dashboards summarizing Sections 1-4 should be reviewed by the data team on a regular cadence
  (e.g., monthly) alongside the retention team's campaign performance reviews.
