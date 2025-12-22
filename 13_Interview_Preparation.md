# AI Engineer Interview Preparation Guide

## 📖 Table of Contents
1. [Technical Questions](#technical-questions)
2. [Coding Challenges](#coding-challenges)
3. [System Design](#system-design)
4. [ML Concepts](#ml-concepts)
5. [Portfolio Presentation](#portfolio-presentation)

---

## Technical Questions

### Machine Learning Fundamentals

**Q: What is the difference between supervised and unsupervised learning?**

**A:**
- **Supervised Learning**: Uses labeled data to learn mapping from inputs to outputs
  - Examples: Classification, Regression
  - Algorithms: Linear Regression, Random Forest, Neural Networks
  
- **Unsupervised Learning**: Finds patterns in unlabeled data
  - Examples: Clustering, Dimensionality Reduction
  - Algorithms: K-Means, PCA, Autoencoders

**Q: Explain bias-variance tradeoff.**

**A:**
- **Bias**: Error from oversimplifying assumptions
  - High bias = Underfitting
  - Low bias = Model can capture complex patterns
  
- **Variance**: Error from sensitivity to small fluctuations
  - High variance = Overfitting
  - Low variance = Model is stable

- **Tradeoff**: Reducing bias increases variance and vice versa
- **Solution**: Find optimal balance, use regularization, ensemble methods

**Q: What is cross-validation and why use it?**

**A:**
- **Cross-validation**: Split data into k folds, train on k-1, test on 1
- **Benefits**:
  - Better estimate of model performance
  - Reduces overfitting risk
  - Uses all data for training and testing
- **Types**: K-fold, Stratified K-fold, Leave-one-out

**Q: Explain overfitting and how to prevent it.**

**A:**
- **Overfitting**: Model learns training data too well, fails on new data
- **Prevention**:
  - More training data
  - Regularization (L1/L2)
  - Dropout (neural networks)
  - Early stopping
  - Reduce model complexity
  - Cross-validation
  - Ensemble methods

---

## Coding Challenges

### Challenge 1: Implement K-Means

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
    
    def fit(self, X):
        # Initialize centroids randomly
        n_samples, n_features = X.shape
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = np.array([X[clusters == k].mean(axis=0) 
                                     for k in range(self.n_clusters)])
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def predict(self, X):
        return self._assign_clusters(X)
```

### Challenge 2: Implement Logistic Regression

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_pred]
```

### Challenge 3: Feature Engineering

```python
def create_features(df):
    # Time features
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Interaction features
    df['feature1_x_feature2'] = df['feature1'] * df['feature2']
    df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-6)
    
    # Aggregation features
    df['mean_by_category'] = df.groupby('category')['value'].transform('mean')
    df['std_by_category'] = df.groupby('category')['value'].transform('std')
    
    # Rolling features
    df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
    
    return df
```

---

## System Design

### Design a Recommendation System

**Components:**
1. **Data Collection**: User interactions, item features
2. **Feature Engineering**: User embeddings, item embeddings
3. **Model**: Collaborative filtering, content-based, hybrid
4. **Serving**: Real-time API, batch recommendations
5. **Evaluation**: A/B testing, offline metrics

**Approach:**
- **Collaborative Filtering**: User-item matrix, matrix factorization
- **Content-Based**: Item features, user preferences
- **Hybrid**: Combine both approaches

### Design a ML Pipeline

**Components:**
1. **Data Ingestion**: Batch/streaming
2. **Data Processing**: Cleaning, feature engineering
3. **Model Training**: Training pipeline, hyperparameter tuning
4. **Model Serving**: API, batch predictions
5. **Monitoring**: Performance, data drift

**Tools:**
- **Data**: Apache Spark, Pandas
- **Training**: MLflow, Kubeflow
- **Serving**: TensorFlow Serving, FastAPI
- **Monitoring**: Prometheus, Grafana

---

## ML Concepts

### Deep Learning

**Q: Explain backpropagation.**

**A:**
- Algorithm to compute gradients in neural networks
- Forward pass: Compute predictions
- Backward pass: Compute gradients using chain rule
- Update weights using gradients and learning rate

**Q: What are activation functions and why needed?**

**A:**
- Introduce non-linearity to neural networks
- **ReLU**: Most common, solves vanishing gradient
- **Sigmoid**: For binary classification output
- **Softmax**: For multi-class classification
- **Tanh**: Alternative to sigmoid

**Q: Explain dropout.**

**A:**
- Regularization technique
- Randomly set neurons to zero during training
- Prevents overfitting
- Forces network to learn redundant representations

### NLP

**Q: Explain attention mechanism.**

**A:**
- Allows model to focus on relevant parts of input
- Computes attention weights for each position
- Used in transformers, improves performance
- Self-attention: Attention within same sequence

**Q: What is BERT?**

**A:**
- Bidirectional Encoder Representations from Transformers
- Pre-trained on large text corpus
- Can be fine-tuned for specific tasks
- State-of-the-art for many NLP tasks

---

## Portfolio Presentation

### Structure Your Presentation

1. **Introduction** (1 min)
   - Your background
   - What you'll present

2. **Problem Statement** (2 min)
   - What problem you solved
   - Why it's important

3. **Approach** (3 min)
   - Data used
   - Methods/techniques
   - Architecture

4. **Results** (2 min)
   - Metrics
   - Visualizations
   - Key insights

5. **Challenges & Solutions** (1 min)
   - Problems faced
   - How you solved them

6. **Future Work** (1 min)
   - Improvements
   - Extensions

### Key Points to Highlight

- **Technical depth**: Show understanding of algorithms
- **Problem-solving**: How you approached challenges
- **Results**: Quantifiable improvements
- **Code quality**: Clean, documented code
- **Deployment**: Working application

---

## Common Interview Questions

### Behavioral Questions

- "Tell me about a challenging ML project."
- "How do you handle imbalanced datasets?"
- "Describe your approach to feature engineering."
- "How do you evaluate model performance?"
- "What's your experience with deployment?"

### Technical Deep Dives

- "Explain how Random Forest works."
- "What's the difference between bagging and boosting?"
- "How would you handle a dataset with 1 million features?"
- "Explain gradient descent variants."
- "How do you prevent overfitting in neural networks?"

---

## Practice Resources

### Coding Practice
- **LeetCode**: Algorithm problems
- **Kaggle**: Competitions and datasets
- **HackerRank**: ML challenges

### Mock Interviews
- **Pramp**: Practice interviews
- **InterviewBit**: Technical prep
- **Exponent**: System design prep

### Study Materials
- Review your projects
- Read ML papers
- Practice explaining concepts
- Code implementations from scratch

---

## Key Takeaways

1. **Understand fundamentals** - Don't just memorize
2. **Practice coding** - Implement algorithms
3. **Know your projects** - Be ready to explain
4. **Think out loud** - Show your thought process
5. **Ask questions** - Show interest and curiosity

---

**Prepare thoroughly and practice explaining concepts clearly!**

