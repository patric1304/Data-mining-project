# 🏠 Data Mining Project — Airbnb Listings: Pricing & Guest Experience Analysis

## 📌 Project Overview

**Theme:** Analyze Airbnb listings by combining structured listing features (price, location, amenities, host info) with guest review text to understand what drives pricing and guest satisfaction.

**Data Story:** *"What makes an Airbnb listing successful — and can we predict its price and segment the market using data mining?"*

**Team size:** 2
**Deliverables:** Python notebook (commented) + Presentation (max 7 minutes)
**Grading:** ⅓ Presentation clarity · ⅓ Technical rigor & creativity · ⅓ Insightfulness

---

## 📂 Dataset

**Source:** [Inside Airbnb — Get the Data](http://insideairbnb.com/get-the-data/)

Pick **one city** with enough data (recommended: Amsterdam, Paris, Barcelona, London, or New York).

Download these files for the chosen city:
- `listings.csv.gz` — full listing details (price, amenities, host info, location, ratings)
- `reviews.csv.gz` — guest review text with dates

> **IMPORTANT:** Download the **detailed** listings file (not the summary), it contains 70+ columns.

**Expected size:** ~20k–80k listings depending on city, 200k–1M+ reviews.

---

## 🔧 6 Technical Topics (Locked In)

| # | Topic | Description |
|---|---|---|
| 1 | **Data Cleaning & Preprocessing** *(mandatory)* | Handle missing values, parse amenities JSON/lists, clean price strings (`$1,200` → `1200`), normalize numerical features, handle outliers, merge listings + reviews |
| 2 | **EDA / Data Visualization** | Distribution of prices, room types, map visualizations (price by neighborhood), correlation heatmaps, review count vs. rating, host response stats |
| 3 | **Sentiment Analysis** | Analyze review text to compute a sentiment score per listing. Compare sentiment vs. numerical rating. Find listings where sentiment disagrees with star rating |
| 4 | **Feature Importance / PCA** | Identify which features matter most for price (amenities, location, room type, host attributes). Use PCA to reduce dimensionality for visualization |
| 5 | **Supervised Model 1** — Price Prediction | Predict listing price from features. Try multiple models (e.g. Random Forest, Gradient Boosting / XGBoost). Include hyperparameter tuning, cross-validation, evaluation metrics (RMSE, R², MAE) |
| 6 | **Unsupervised Model 1** — Market Segmentation | Cluster listings into segments (e.g. budget, mid-range, luxury, unique stays). Use K-Means and/or DBSCAN. Evaluate with silhouette score. Interpret cluster characteristics |

---

## 👥 Work Split

### 🅰️ Person A — Data Foundation & Tabular Analysis

**Responsible for:**

#### 1. Data Cleaning & Preprocessing (tabular part)
- Download and load the dataset
- Clean price, parse amenities, handle missing values
- Merge listings + reviews (aggregate review sentiment per listing)
- Encode categorical variables, normalize numerics
- Handle outliers (e.g. price > $10,000/night)
- Produce a clean, merged DataFrame ready for analysis

#### 2. EDA / Data Visualization
- Univariate distributions (price, ratings, room type, neighborhood)
- Bivariate analysis (price vs. room type, price vs. location, superhost vs. not)
- Map visualization (folium or plotly) showing price heatmap by neighborhood
- Correlation heatmap of numerical features
- Summary statistics and key observations

#### 3. Supervised Model 1 — Price Prediction
- Feature engineering for the model
- Train/test split
- Train at least 2 models (e.g. Random Forest + XGBoost)
- Hyperparameter tuning (GridSearch or RandomizedSearch)
- Evaluation: RMSE, MAE, R²
- Feature importance plot from best model
- Interpretation: which features drive price?

---

### 🅱️ Person B — Text Analysis & Unsupervised Learning

**Responsible for:**

#### 4. Data Cleaning & Preprocessing (text/review part)
- Clean review text (lowercase, remove punctuation, stopwords, tokenize)
- Handle non-English reviews (detect and filter or translate)
- Prepare text data for sentiment analysis

#### 5. Sentiment Analysis
- Apply sentiment analysis (VADER or TextBlob) to reviews
- Aggregate sentiment scores per listing
- Compare sentiment vs. numeric review scores
- Identify "hidden gem" listings (high sentiment, low visibility) and "disappointment" listings (low sentiment, high rating)
- Visualize sentiment distribution, sentiment by neighborhood

#### 6. Feature Importance / PCA
- Run PCA on numerical features — how many components explain 90%+ variance?
- 2D/3D PCA scatter plot colored by price range or room type
- Feature importance analysis (correlation-based + model-based)
- Interpret: are there redundant features?

#### 7. Unsupervised Model 1 — Market Segmentation
- Select features for clustering (price, location scores, amenity count, ratings, sentiment)
- Scale features
- Run K-Means (elbow method + silhouette score for optimal k)
- Try DBSCAN as alternative
- Profile each cluster (what defines "budget" vs. "luxury" vs. "unique"?)
- Visualize clusters in PCA space

---

### 🤝 Shared Responsibilities

| Task | Details |
|---|---|
| **Choose the city** | Agree on which city dataset to use |
| **Notebook integration** | Merge individual sections into one coherent notebook |
| **Presentation** | Co-create slides, each presents their own sections (~3.5 min each) |
| **Interpretation & insights** | Both contribute to the "so what?" for every analysis |
| **Code review** | Review each other's code before final merge |

---

## 📅 Suggested Timeline

| Phase | Tasks | Deadline (adjust as needed) |
|---|---|---|
| **Week 1** | Download data, data cleaning, initial EDA | Day 4 |
| **Week 2** | Sentiment analysis, PCA, supervised model, unsupervised model | Day 10 |
| **Week 3** | Polish notebook, create presentation, rehearse | Day 14 |
| **Final** | Merge everything, final review, practice presentation timing | Submission day |

---

## 🛠️ Coding Conventions

- **Language:** Python 3.x
- **Notebook:** Jupyter Notebook (`.ipynb`)
- **Libraries:** pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, nltk/spacy, textblob/vaderSentiment, folium (for maps)
- **Comments:** Every code cell should have a markdown cell above it explaining what it does and why
- **Naming:** Use `snake_case` for variables and functions
- **Sections:** Use clear markdown headers in the notebook matching the 6 topics

---

## 📊 Key Insights to Pursue

These are the "non-obvious" findings the jury will reward:

1. **The superhost premium** — How much more do superhosts charge, and is it justified by reviews?
2. **Amenity ROI** — Which amenities (pool, wifi, AC) have the biggest impact on price vs. satisfaction?
3. **Sentiment–rating disconnect** — Listings where text sentiment doesn't match numeric rating (hidden problems or hidden gems)
4. **Neighborhood arbitrage** — Are there underpriced neighborhoods with high satisfaction?
5. **Market segments** — Do data-driven clusters match intuitive categories (budget/mid/luxury)?
6. **Prediction accuracy** — Can we predict price well? What are the hardest listings to predict and why?

---

## 🔀 Git Workflow

We work on **one shared notebook** (`airbnb_analysis.ipynb`), alternating commits via git.

### Rules
1. **Pull before you start.** Always `git pull` before opening the notebook.
2. **Only edit your sections.** Each section is marked with `Person A` or `Person B` — stay in your lane.
3. **Commit when you're done for the session.** Don't leave uncommitted changes overnight.
4. **Clear outputs before committing.** In Jupyter: *Kernel → Restart & Clear Output*, then save. This avoids bloated diffs.
5. **Never work at the same time.** Coordinate so only one person has the notebook open. Notebook JSON doesn't merge well.

### Typical session
```bash
git pull
# open notebook, work on your section
# Kernel → Restart & Clear Output → Save
git add airbnb_analysis.ipynb
git commit -m "Person A: completed EDA section"
git push
# message your teammate that you pushed
```

### Boundary markers
The notebook has markers like this between sections:
```
═══ END Person A: Data Cleaning (Tabular) ═══
▶ START Person B: Data Cleaning (Text/Reviews)
```
Person B inserts their cells **between** the START and END markers for their section. Delete all markers before final submission.

### If something goes wrong
```bash
# Discard your local changes and take the remote version
git checkout -- airbnb_analysis.ipynb

# Or stash your changes temporarily
git stash
git pull
git stash pop
```

---

## ✅ Deliverables Checklist

- [ ] Clean, commented Jupyter notebook with all 6 technical topics
- [ ] Each topic has: code + visualization + written interpretation
- [ ] Presentation slides (max 7 minutes, no code on slides)
- [ ] Both team members can explain any part of the project
- [ ] Archive file (.zip) containing notebook + presentation
