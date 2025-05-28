import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Assume df is your cleaned, ready-to-train DataFrame
X = df_scaled.values  # Extract numpy array

# Standard scaling (if not already done, but safe to apply again)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define ensemble model
class AnomalyEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.iso = IsolationForest(contamination=0.01, random_state=42)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=True)
        self.env = EllipticEnvelope(contamination=0.01)
        
    def fit(self, X, y=None):
        self.iso.fit(X)
        self.lof.fit(X)
        self.env.fit(X)
        return self

    def predict(self, X):
        preds = np.vstack([
            self.iso.predict(X),
            self.lof.predict(X),
            self.env.predict(X)
        ])
        # Convert predictions: -1 → 1 (abnormal), 1 → 0 (normal)
        preds = (preds == -1).astype(int)
        majority_vote = np.round(preds.mean(axis=0))  # 0 (normal) or 1 (abnormal)
        return majority_vote

# Train the ensemble
ensemble_model = AnomalyEnsemble()
ensemble_model.fit(X_scaled)

# Save the model and scaler
dump(ensemble_model, "anomaly_ensemble_model.joblib")
dump(scaler, "scaler.joblib")

print("✅ Model training complete and saved.")
