# Person B — TODO

**Dataset:** Amsterdam — Inside Airbnb
**Notebook:** `airbnb_analysis.ipynb`
**Your sections are marked with** `Person B — place your ... cells here.`

> **Git rule:** `git pull` before you start. Only edit between your START/END markers. Clear outputs before committing.

---

## Your 4 sections

### 1. Data Cleaning — Text/Reviews
**Location:** Between `START Person B: Data Cleaning` and `END Person B: Data Cleaning`

**What to do:**
- Load `reviews_raw` (already loaded in cell 0)
- Clean review text: lowercase, remove punctuation, strip stopwords, tokenize
- Detect and filter non-English reviews (use `langdetect` library)
- Remove very short reviews (< 10 characters) — likely "Great!" type reviews with no substance
- Save cleaned reviews for sentiment analysis

**Output:** A cleaned DataFrame with columns `[listing_id, comments_clean]`

**Libraries:** `nltk`, `re`, `langdetect` (install with `pip install langdetect`)

```python
# Suggested approach
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

reviews = reviews_raw.copy()
reviews = reviews.dropna(subset=['comments'])
reviews['comments_clean'] = reviews['comments'].apply(clean_text)
reviews = reviews[reviews['comments_clean'].str.len() > 10]
```

---

### 2. Sentiment Analysis
**Location:** Between `START Person B: Sentiment Analysis` and `END Person B: Sentiment Analysis`

**What to do:**
- Apply VADER sentiment to each cleaned review
- Aggregate sentiment per listing (mean compound score)
- Save as `data/review_sentiments.csv` with columns `[listing_id, avg_sentiment]`
- Compare sentiment vs. numeric `review_scores_rating` — scatter plot + correlation
- Identify "hidden gems" (high sentiment, low visibility) and "disappointments" (low sentiment, high rating)
- Visualize: sentiment distribution, sentiment by neighbourhood
- **Add interpretation** after each visualization (2-3 sentences)

**Libraries:** `vaderSentiment` (install with `pip install vaderSentiment`)

```python
# Suggested approach
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
reviews['compound'] = reviews['comments_clean'].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

# Aggregate per listing
sentiment_per_listing = (
    reviews.groupby('listing_id')['compound']
    .agg(['mean', 'count'])
    .rename(columns={'mean': 'avg_sentiment', 'count': 'review_count'})
    .reset_index()
)

# Save for merge with Person A's data
sentiment_per_listing.to_csv(f'{DATA_DIR}/review_sentiments.csv', index=False)
```

**After saving:** Tell Person A to uncomment the merge cell and re-run the supervised model. The `avg_sentiment` feature will be automatically picked up.

---

### 3. Feature Importance / PCA
**Location:** Between `START Person B: Feature Importance / PCA` and `END Person B: Feature Importance / PCA`

**What to do:**
- Select numerical features from the clean listings DataFrame
- Standardize features with `StandardScaler`
- Run PCA — plot explained variance ratio (cumulative)
- Report: how many components explain 90% of variance?
- 2D PCA scatter plot, colored by `room_type` or price range
- Feature importance: correlation-based ranking with price
- **Add interpretation** (which features are redundant? what does PC1 vs PC2 represent?)

```python
# Suggested approach
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

num_cols = listings.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'price' and c != 'id']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(listings[num_cols].fillna(0))

pca = PCA()
pca.fit(X_scaled)

# Cumulative explained variance plot
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.argmax(cumvar >= 0.90) + 1
print(f'{n_90} components explain 90% of variance')

plt.plot(cumvar, marker='o')
plt.axhline(0.9, color='r', linestyle='--')
plt.xlabel('Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA — Explained Variance')
plt.show()

# 2D scatter
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=listings['price'], cmap='YlOrRd', alpha=0.4, s=5)
plt.colorbar(label='Price ($)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA — 2D Projection')
plt.show()
```

---

### 4. Unsupervised Model — Market Segmentation
**Location:** Between `START Person B: Unsupervised Model` and `END Person B: Unsupervised Model`

**What to do:**
- Select features for clustering: `price, accommodates, bedrooms, beds, bathrooms, amenities_count, review_scores_rating, dist_to_center, avg_sentiment` (if available)
- Scale with `StandardScaler`
- **K-Means:** elbow method (k=2..10) + silhouette score → pick optimal k
- **DBSCAN:** try as alternative, tune `eps` and `min_samples`
- Profile each cluster: print mean values per cluster, give each a label (e.g. "Budget", "Mid-range", "Luxury")
- Visualize clusters in PCA 2D space
- **Add interpretation** (do clusters match market intuition? what defines each segment?)

```python
# Suggested approach
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

cluster_features = ['price', 'accommodates', 'bedrooms', 'bathrooms',
                    'amenities_count', 'review_scores_rating']
cluster_features = [c for c in cluster_features if c in listings.columns]

X_clust = listings[cluster_features].fillna(0)
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

# Elbow + silhouette
inertias, sil_scores = [], []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_clust_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(K_range, inertias, marker='o')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Inertia')

axes[1].plot(K_range, sil_scores, marker='o', color='green')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Score')
plt.tight_layout()
plt.show()

# Final clustering with best k
best_k = K_range[np.argmax(sil_scores)]
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
listings['cluster'] = km_final.fit_predict(X_clust_scaled)

# Profile clusters
print(listings.groupby('cluster')[cluster_features].mean().round(2))
```

---

## Checklist

- [ ] Section 1 — Data Cleaning (Text/Reviews)
- [ ] Section 2 — Sentiment Analysis + `review_sentiments.csv` saved
- [ ] Section 3 — Feature Importance / PCA
- [ ] Section 4 — Unsupervised Model (K-Means + DBSCAN)
- [ ] Interpretation markdown cells after every visualization
- [ ] Notify Person A when sentiment CSV is ready (to re-run supervised model)
- [ ] Help write Conclusions section
- [ ] Co-create presentation (~3.5 min your sections)
- [ ] Final: clear outputs, delete markers, test full run

---

## Important Notes

- The clean listings DataFrame is saved at `data/listings_clean.csv` — load it if you need a fresh start
- Price was log-transformed for modelling. The actual prices are in the DataFrame's `price` column
- `dist_to_center` feature was already added by Person A (distance from Dam Square)
- Refer to `context.md` for the full project overview and grading criteria
