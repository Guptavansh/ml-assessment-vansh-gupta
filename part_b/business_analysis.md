# Part B — Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

This is a **multi-class classification problem**.

**Target variable:** `promotion_type` — one of five classes:
Flat Discount, BOGO, Free Gift with Purchase, Category-Specific
Offer, or Loyalty Points Bonus.

**Input features:**
- Store attributes: store_id, store_size, location_type,
  competition_density, monthly_footfall
- Temporal features: month, is_weekend, is_festival,
  day_of_week, is_month_end
- Historical performance: past_avg_items_sold_by_promotion,
  past_avg_items_sold_by_store
- Customer demographics: age_group_distribution,
  income_segment (if available)

**Justification for classification:** The output is a discrete
recommendation — the model must pick exactly one promotion from
a fixed set of five options for each store-month combination.
This is not regression (we are not predicting a continuous
quantity) and not binary classification (there are five
possible outputs, not two).

An alternative framing would be to train a regression model
that predicts items_sold for each promotion separately, then
recommend the promotion with the highest predicted volume.
This indirect approach offers more interpretability and avoids
treating five qualitatively different promotions as arbitrary
class labels.

---

### B1(b) — Why Items Sold is a Better Target Than Revenue

Revenue is the product of price and volume. When a Flat Discount
promotion runs, unit prices are deliberately reduced — so revenue
can appear lower even when the promotion is highly effective at
driving footfall and basket size. A model trained on revenue would
systematically underrate discount-heavy promotions and overrate
premium promotions that sell fewer units at higher margins.

Items sold (sales volume) is promotion-neutral — it measures
customer response to the promotion independently of the price
effect. If BOGO drives 40% more units than Loyalty Points in
the same store, that signal is visible in volume but may be
obscured in revenue depending on the discount depth.

The broader principle this illustrates is **target variable
alignment**: the target should directly measure what the
business intervention is trying to influence. Since the
marketing team is choosing which promotion to run — not
which price to set — items sold is the variable they can
most directly move with their decision. Revenue is a
downstream outcome affected by too many other factors
(margin strategy, price elasticity, seasonal demand) to
serve as a clean feedback signal for promotion selection.

---

### B1(c) — Against a Single Global Model

A single global model trained across all 50 stores would learn
average behaviour across all location types, store sizes, and
customer demographics simultaneously. The problem is that
"average" behaviour may not exist in any single store — a
rural store's customers respond to Free Gift promotions very
differently from an urban flagship store's customers who may
respond better to Category-Specific Offers tied to new arrivals.

A more appropriate strategy is a **stratified ensemble approach**:

1. Segment stores into groups based on location_type and
   store_size (e.g., urban-large, urban-small, semi-urban,
   rural). These segments reflect genuinely different customer
   profiles and competitive environments.

2. Train one model per segment. Each model learns the
   promotion-response patterns specific to that segment
   without being diluted by data from structurally different
   stores.

3. For stores with sufficient history, additionally train
   **store-level fine-tuned models** using transfer learning
   — start with the segment model's weights and fine-tune
   on that store's own data.

This approach balances the data volume problem (individual
stores may not have enough history to train reliably alone)
with the heterogeneity problem (pooling all stores destroys
meaningful local signals).

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

The four tables — transactions, store attributes, promotion
details, and calendar — should be joined at the grain of
**one row per store per transaction date**.

**Join sequence:**
1. Start with the transactions table as the base
   (store_id, transaction_date, items_sold)
2. Left join store attributes on store_id to attach
   store_size, location_type, competition_density,
   and monthly_footfall
3. Left join promotion details on (store_id, month) to
   attach which promotion was active for that store
   in that month
4. Left join the calendar table on transaction_date to
   attach is_weekend and is_festival flags

**Final grain:** One row = one store on one transaction date
with that day's items_sold, the active promotion for that
month, and all store and calendar attributes attached.

**Aggregations before modelling:**
- If modelling at monthly grain rather than daily, aggregate
  items_sold to monthly totals per store
- Calculate historical averages: avg_items_sold per promotion
  per store over the past 3 months (rolling features)
- Calculate competition_response_ratio: how much a store's
  sales change relative to its competition density — this
  captures local competitive pressure effects

---

### B2(b) — EDA Strategy

**Analysis 1 — Promotion distribution by location type**
Bar chart showing how frequently each promotion was deployed
across urban, semi-urban, and rural stores. Looking for: whether
certain promotions were deployed more in specific locations,
which would create confounding — the model may learn location
effects rather than promotion effects if this is not controlled.

**Analysis 2 — Items sold by promotion type (box plot)**
One box per promotion showing the distribution of items_sold.
Looking for: whether some promotions have consistently higher
medians, or whether the variance is so large that no promotion
dominates. High variance within a promotion class suggests the
promotion's effectiveness is highly context-dependent — which
supports the stratified modelling approach.

**Analysis 3 — Monthly seasonality heatmap**
Heatmap with store on one axis and month on the other,
with items_sold as the colour. Looking for: whether certain
months are universally strong (e.g., festive season) or whether
seasonal patterns differ by location type. This directly
influences whether month should be a simple numeric feature
or whether festive months need special encoding.

**Analysis 4 — Promotion × is_festival interaction plot**
Line plot showing average items_sold for each promotion type
on festival days vs non-festival days. Looking for: whether
some promotions dramatically outperform others specifically
on festival days. If Free Gift + festival produces a spike
that no other combination matches, we need an interaction
feature promotion_x_festival in the model.

---

### B2(c) — Handling the 80% No-Promotion Imbalance

If 80% of transactions occurred without any promotion, a naive
model will learn that "no promotion" is the baseline and
will underestimate the impact of the five promotion types
because they appear in only 20% of training data.

**Impact on modelling:**
- The model may predict items_sold based primarily on
  store and calendar features, with promotion type having
  very low feature importance — not because promotions
  don't matter, but because the model has seen too few
  examples to learn their effects reliably.

**Steps to address this:**
1. **Filter the training data** to include only promoted
   transactions when training the promotion recommendation
   model. The baseline (no-promotion) periods are useful
   for establishing a counterfactual benchmark but should
   not dominate the training signal.

2. **Create a lift feature** — for each store-promotion
   combination, compute the lift ratio:
   items_sold_with_promotion / avg_items_sold_without_promotion.
   Train the model to predict lift rather than raw volume.
   This makes the target promotion-relative and removes
   the baseline noise entirely.

3. **Use stratified sampling** during train-test split to
   ensure all five promotion types are represented equally
   in both splits — preventing any one promotion from being
   underrepresented in the test set purely by chance.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split Strategy and Metrics

**Why random split is inappropriate:**
With three years of monthly data across 50 stores, a random
split would mix observations from early 2022 into the test
set alongside late 2024 records. The model would train on
the future and be tested on the past — a complete inversion
of the real prediction task. Promotions also have temporal
dependencies: December 2023 performance is correlated with
November 2023 performance in ways a random split would sever.

**Correct split strategy:**
Use a **rolling time-based split**. Train on the first 24
months (months 1-24), test on months 25-36. This mirrors
deployment — the model is always predicting the next period
using only historical data available at that point. For
robust evaluation, use **time-series cross-validation**
with multiple folds (e.g., train on months 1-12, test on
13-15; train on 1-15, test on 16-18; etc.) to assess
stability across different time windows.

**Evaluation metrics:**
- **RMSE** — measures average prediction error in units of
  items_sold. In this context, an RMSE of 30 means the
  model's recommendations are off by 30 items on average —
  the business team can decide if that margin is acceptable
  given typical store volumes.
- **MAE** — more interpretable than RMSE for communicating
  to non-technical stakeholders. "Our predictions are wrong
  by 25 items on average" is immediately meaningful to a
  store manager.
- **Promotion recommendation accuracy** — treated as a
  classification metric: of all the times the model
  recommended Promotion X, what percentage of the time
  was Promotion X genuinely the best option (highest
  actual items_sold)? This directly measures the quality
  of recommendations, not just prediction error.

---

### B3(b) — Investigating Different Recommendations for Store 12

The model recommended Loyalty Points Bonus in December and
Flat Discount in March for the same store. This seems
contradictory but reflects exactly what feature importance
analysis is designed to explain.

**Investigation approach:**

1. Extract the feature values for Store 12 in December
   and March side by side — specifically: is_festival,
   month, competition_density, and historical_avg_items_sold.
   December likely has high is_festival = 1 and high footfall
   from the festive season. March is post-festival with
   lower footfall and potentially higher competition activity.

2. Use **SHAP (SHapley Additive exPlanations)** values to
   decompose each recommendation into individual feature
   contributions. For December, the SHAP output might show
   that is_festival and month=12 together pushed the model
   strongly toward Loyalty Points — customers in the festive
   season respond to reward accumulation because they are
   already buying in higher volumes. For March, low footfall
   and high competition_density pushed the model toward
   Flat Discount — a more aggressive price signal needed
   to drive traffic against competition.

**Communication to marketing team:**
Present a simple two-column comparison table showing
Store 12's key feature values in December vs March alongside
the SHAP contribution of each feature to the final
recommendation. Frame it as: "In December, festive demand
means customers are already motivated to buy — loyalty
rewards compound that motivation. In March, we need to
attract footfall first, and a visible price discount is
the most effective trigger in a competitive environment."
This translates model logic into a narrative the marketing
team can validate from their own experience.

---

### B3(c) — End-to-End Deployment and Monitoring

**Saving the model:**
Serialise the trained pipeline (including the preprocessor
and model) using `joblib.dump(pipeline, 'promo_model.pkl')`.
Store this file in a version-controlled model registry
(e.g., MLflow Model Registry or a versioned S3 bucket)
so that every deployed model can be traced back to the
exact training data and hyperparameters used to produce it.

**Monthly inference process:**
At the start of each month, a scheduled job (Airflow DAG
or cron job) runs the following steps automatically:
1. Pull the latest store attributes, calendar flags, and
   rolling historical averages for all 50 stores
2. Assemble these into the same feature schema the model
   was trained on — same column names, same encoding
3. Load the saved model with `joblib.load('promo_model.pkl')`
4. Run `model.predict(X_new)` to generate one recommendation
   per store
5. Write recommendations to a database table that the
   marketing dashboard reads from

No retraining occurs in this step — the same saved model
generates all 50 recommendations in seconds.

**Monitoring for model degradation:**
Deploy three monitoring checks that run automatically
after each month's actual results come in:

1. **Prediction error drift** — compute RMSE and MAE on
   the most recent month's actuals vs predictions. If
   RMSE increases by more than 15% compared to the
   rolling 3-month average, trigger a retraining alert.

2. **Feature distribution shift** — use statistical tests
   (Kolmogorov-Smirnov test) to compare the distribution
   of key input features (competition_density, footfall)
   in the current month against the training data
   distribution. Significant shifts indicate the world has
   changed in ways the model has not seen — a leading
   indicator of degradation before prediction errors appear.

3. **Recommendation diversity check** — if the model
   starts recommending the same promotion for 80%+ of
   stores in a given month (collapsed recommendations),
   that is a sign the model has overfit to a dominant
   pattern and is no longer capturing store-level
   heterogeneity. Flag this for human review before
   the recommendations are acted on.

Retraining is triggered when any two of these three
checks fire in the same month — using a sliding window
of the most recent 18 months of data to keep the model
current without discarding long-term seasonal patterns.
