"""
=============================================================================
SENTINEL-FRAUD: Training Pipeline - THE BRAIN
=============================================================================
ML training pipeline with SMOTE for handling class imbalance.

Features:
    - Automated feature engineering
    - SMOTE for oversampling minority class (fraud)
    - XGBoost model training with hyperparameter tuning
    - Model evaluation and serialization
    - Cross-validation with stratification
=============================================================================
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
logger = logging.getLogger("training")


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    # Data paths
    data_path: str = "data/transactions.csv"
    model_output_path: str = "ml/artifacts/xgboost_fraud_model.json"
    preprocessor_output_path: str = "ml/artifacts/preprocessor.pkl"
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    n_folds: int = 5
    
    # SMOTE parameters
    use_smote: bool = True
    smote_strategy: str = "auto"  # 'auto', 0.1, 0.5, etc.
    smote_k_neighbors: int = 5
    
    # XGBoost parameters
    xgboost_params: dict = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "gamma": 0.1,
        "scale_pos_weight": 1,  # Will be adjusted based on class imbalance
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": ["auc", "aucpr"],
    })
    
    # Hyperparameter tuning
    enable_tuning: bool = True
    tuning_param_grid: dict = field(default_factory=lambda: {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [4, 6, 8],
        "classifier__learning_rate": [0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9],
    })


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """
    Automated feature engineering for fraud detection.
    
    Creates the following feature types:
    - Transaction features (amount, currency, merchant)
    - Temporal features (hour, day, weekend, holidays)
    - User behavior features (average, std, velocity)
    - Geographic features (distance, travel speed)
    - Risk scores (merchant risk, channel risk)
    """

    def __init__(self):
        self.numeric_features = [
            "amount",
            "amount_log",
            "amount_zscore",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "time_since_last_txn",
            "txn_count_1h",
            "txn_count_24h",
            "txn_count_7d",
            "total_amount_24h",
            "distance_from_last",
            "travel_speed",
            "is_impossible_travel",
            "user_avg_amount",
            "user_std_amount",
            "user_txn_count",
            "user_fraud_history",
            "merchant_risk_score",
            "velocity_score",
        ]
        
        self.categorical_features = [
            "currency",
            "merchant_category",
            "channel",
            "country",
        ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw transaction data into features."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Basic amount features
        df["amount_log"] = np.log1p(df["amount"])
        
        # Temporal features
        if "timestamp" in df.columns:
            df["hour_of_day"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # User-level features (need to be pre-computed in production)
        user_stats = df.groupby("user_id").agg({
            "amount": ["mean", "std", "count"]
        }).reset_index()
        user_stats.columns = ["user_id", "user_avg_amount", "user_std_amount", "user_txn_count"]
        user_stats["user_std_amount"] = user_stats["user_std_amount"].fillna(0)
        
        df = df.merge(user_stats, on="user_id", how="left")
        df["amount_zscore"] = np.where(
            df["user_std_amount"] > 0,
            (df["amount"] - df["user_avg_amount"]) / df["user_std_amount"],
            0
        )
        
        # Velocity features (simplified for training)
        df["txn_count_1h"] = df.groupby("user_id")["transaction_id"].transform(
            lambda x: x.rolling(10, min_periods=1).count()
        ).fillna(1).astype(int)
        
        df["txn_count_24h"] = df.groupby("user_id")["transaction_id"].transform(
            lambda x: x.rolling(50, min_periods=1).count()
        ).fillna(1).astype(int)
        
        df["txn_count_7d"] = df.groupby("user_id")["transaction_id"].transform(
            lambda x: x.rolling(100, min_periods=1).count()
        ).fillna(1).astype(int)
        
        # Velocity score
        df["velocity_score"] = np.clip(
            (df["txn_count_1h"] / 10) * 0.3 + (df["txn_count_24h"] / 50) * 0.7,
            0, 1
        )
        
        # Distance features (simplified for training)
        if "latitude" in df.columns and "longitude" in df.columns:
            df["distance_from_last"] = df.groupby("user_id").apply(
                self._calculate_distance_rolling
            ).reset_index(level=0, drop=True).fillna(0)
            
            df["travel_speed"] = df.groupby("user_id").apply(
                self._calculate_speed_rolling
            ).reset_index(level=0, drop=True).fillna(0)
            
            df["is_impossible_travel"] = (df["travel_speed"] > 1200).astype(int)
        else:
            df["distance_from_last"] = 0
            df["travel_speed"] = 0
            df["is_impossible_travel"] = 0
        
        # Time since last transaction
        if "timestamp" in df.columns:
            df["time_since_last_txn"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds().fillna(0)
        
        # Merchant risk score (simplified)
        merchant_risk = df.groupby("merchant_id")["is_fraud"].mean().reset_index()
        merchant_risk.columns = ["merchant_id", "merchant_risk_score"]
        df = df.merge(merchant_risk, on="merchant_id", how="left")
        df["merchant_risk_score"] = df["merchant_risk_score"].fillna(0.5)
        
        # User fraud history
        user_fraud_history = df.groupby("user_id")["is_fraud"].cumsum() - df["is_fraud"]
        df["user_fraud_history"] = user_fraud_history
        
        # Total amount 24h (simplified)
        df["total_amount_24h"] = df.groupby("user_id")["amount"].transform(
            lambda x: x.rolling(50, min_periods=1).sum()
        ).fillna(df["amount"])
        
        return df

    @staticmethod
    def _calculate_distance_rolling(group):
        """Calculate rolling distance between consecutive transactions."""
        if len(group) < 2:
            return pd.Series([0] * len(group))
        
        from math import radians, sin, cos, sqrt, asin
        
        lat1 = radians(group["latitude"].iloc[0])
        lon1 = radians(group["longitude"].iloc[0])
        
        distances = [0]
        for i in range(1, len(group)):
            lat2 = radians(group["latitude"].iloc[i])
            lon2 = radians(group["longitude"].iloc[i])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            
            distances.append(6371 * c)  # Earth radius in km
            
            lat1, lon1 = lat2, lon2
        
        return pd.Series(distances, index=group.index)

    @staticmethod
    def _calculate_speed_rolling(group):
        """Calculate rolling speed between transactions."""
        distances = FeatureEngineer._calculate_distance_rolling(group)
        
        if "time_since_last_txn" in group.columns:
            time_diff = group["time_since_last_txn"].values
        else:
            time_diff = pd.Series([0] + [3600] * (len(group) - 1)).values
        
        speed = np.where(
            time_diff > 0,
            distances / (time_diff / 3600),
            0
        )
        
        return pd.Series(speed, index=group.index)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class FraudDetectionTrainer:
    """
    Complete training pipeline for fraud detection model.
    
    Steps:
    1. Load and preprocess data
    2. Engineer features
    3. Apply SMOTE for class imbalance
    4. Train XGBoost classifier
    5. Evaluate model performance
    6. Save model artifacts
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.model: Optional[xgb.XGBClassifier] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.metrics: dict = {}

    def load_data(self, path: str) -> pd.DataFrame:
        """Load transaction data from CSV."""
        logger.info(f"Loading data from {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} transactions")
        
        # Check class distribution
        if "is_fraud" in df.columns:
            fraud_rate = df["is_fraud"].mean()
            logger.info(f"Fraud rate: {fraud_rate:.4%}")
            logger.info(f"Class distribution:\n{df['is_fraud'].value_counts()}")
        
        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Preprocess data and create features."""
        logger.info("Engineering features...")
        
        # Engineer features
        df_features = self.feature_engineer.transform(df)
        
        # Define feature columns
        feature_cols = (
            self.feature_engineer.numeric_features +
            self.feature_engineer.categorical_features
        )
        
        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df_features.columns]
        
        X = df_features[available_cols]
        y = df_features["is_fraud"] if "is_fraud" in df_features.columns else None
        
        logger.info(f"Feature matrix shape: {X.shape}")
        
        if fit:
            # Create preprocessor
            numeric_transformer = ImbPipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            
            categorical_transformer = ImbPipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            
            # Get actual available columns
            available_numeric = [c for c in self.feature_engineer.numeric_features if c in X.columns]
            available_categorical = [c for c in self.feature_engineer.categorical_features if c in X.columns]
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, available_numeric),
                    ("cat", categorical_transformer, available_categorical),
                ],
                remainder="drop"
            )
            
            X_processed = self.preprocessor.fit_transform(X)
        else:
            X_processed = self.preprocessor.transform(X)
        
        logger.info(f"Processed feature matrix shape: {X_processed.shape}")
        
        return X_processed, y.values if y is not None else None

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE for handling class imbalance."""
        if not self.config.use_smote:
            return X, y
        
        logger.info("Applying SMOTE for class balancing...")
        
        smote = SMOTE(
            sampling_strategy=self.config.smote_strategy,
            k_neighbors=self.config.smote_k_neighbors,
            random_state=self.config.random_state
        )
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        logger.info(f"Original class distribution: {np.bincount(y)}")
        logger.info(f"Resampled class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> xgb.XGBClassifier:
        """Train XGBoost classifier."""
        logger.info("Training XGBoost classifier...")
        
        # Adjust scale_pos_weight for original imbalance (before SMOTE)
        # This helps model learn the true class distribution
        scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))
        
        params = self.config.xgboost_params.copy()
        params["scale_pos_weight"] = scale_pos_weight
        
        self.model = xgb.XGBClassifier(**params)
        
        # Prepare eval set if validation data provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        logger.info("Training complete")
        return self.model

    def hyperparameter_tuning(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict:
        """Perform hyperparameter tuning with cross-validation."""
        logger.info("Performing hyperparameter tuning...")
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=self.config.random_state)),
            ("classifier", xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=self.config.random_state,
                n_jobs=-1,
            ))
        ])
        
        # Stratified K-Fold
        cv = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        # Grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid=self.config.tuning_param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best AUC score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """Evaluate model performance."""
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        
        # Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        metrics["pr_auc"] = auc(recall, precision)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["classification_report"] = report
        
        # ROC curve points
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
        
        # Log results
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"PR AUC: {metrics['pr_auc']:.4f}")
        
        logger.info(f"\nConfusion Matrix:\n{cm}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        
        self.metrics = metrics
        return metrics

    def save_model(self, model_path: str, preprocessor_path: str) -> None:
        """Save model and preprocessor to disk."""
        logger.info(f"Saving model to {model_path}")
        
        # Create directory if needed
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model in JSON format (more portable)
        self.model.save_model(model_path)
        
        # Save preprocessor
        logger.info(f"Saving preprocessor to {preprocessor_path}")
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        
        # Save metrics
        metrics_path = Path(model_path).parent / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        logger.info("Model artifacts saved successfully")

    def run_full_pipeline(self, data_path: str) -> None:
        """Run the complete training pipeline."""
        start_time = time.time()
        
        # Load data
        df = self.load_data(data_path)
        
        # Preprocess
        X, y = self.preprocess(df, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Further split training for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_train
        )
        
        # Apply SMOTE only to training data
        X_train_resampled, y_train_resampled = self.apply_smote(X_train, y_train)
        
        # Hyperparameter tuning (optional)
        if self.config.enable_tuning:
            best_params = self.hyperparameter_tuning(X_train_resampled, y_train_resampled)
            # Update config with best params
            for key, value in best_params.items():
                param_name = key.replace("classifier__", "")
                if param_name in self.config.xgboost_params:
                    self.config.xgboost_params[param_name] = value
        
        # Train model
        self.train(X_train_resampled, y_train_resampled, X_val, y_val)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test)
        
        # Save model
        self.save_model(
            self.config.model_output_path,
            self.config.preprocessor_output_path
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Training pipeline completed in {elapsed:.2f} seconds")


# =============================================================================
# SYNTHETIC DATA GENERATOR (for testing)
# =============================================================================

def generate_synthetic_data(
    n_samples: int = 100000,
    fraud_rate: float = 0.001,
    output_path: str = "data/transactions.csv"
) -> pd.DataFrame:
    """Generate synthetic transaction data for testing."""
    logger.info(f"Generating {n_samples} synthetic transactions...")
    
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    # Normal transactions
    normal_data = {
        "transaction_id": [f"TXN_{i:010d}" for i in range(n_normal)],
        "user_id": [f"USER_{np.random.randint(1, 10000):08d}" for _ in range(n_normal)],
        "amount": np.abs(np.random.lognormal(4, 1.5, n_normal)),
        "currency": np.random.choice(["USD", "EUR", "GBP"], n_normal),
        "merchant_id": [f"MERCH_{np.random.randint(1, 1000):06d}" for _ in range(n_normal)],
        "merchant_category": np.random.choice(
            ["retail", "food_dining", "online", "grocery", "travel"],
            n_normal
        ),
        "latitude": np.random.uniform(25, 50, n_normal),
        "longitude": np.random.uniform(-125, -70, n_normal),
        "channel": np.random.choice(["online", "offline", "mobile"], n_normal),
        "is_fraud": [0] * n_normal,
    }
    
    # Fraudulent transactions (higher amounts, unusual patterns)
    fraud_data = {
        "transaction_id": [f"TXN_{i:010d}" for i in range(n_normal, n_samples)],
        "user_id": [f"USER_{np.random.randint(1, 10000):08d}" for _ in range(n_fraud)],
        "amount": np.abs(np.random.lognormal(7, 2, n_fraud)),  # Higher amounts
        "currency": np.random.choice(["USD"], n_fraud),
        "merchant_id": [f"MERCH_{np.random.randint(1, 100):06d}" for _ in range(n_fraud)],
        "merchant_category": np.random.choice(
            ["online", "travel", "gaming"],
            n_fraud
        ),
        "latitude": np.random.uniform(-90, 90, n_fraud),  # Random locations
        "longitude": np.random.uniform(-180, 180, n_fraud),
        "channel": np.random.choice(["online", "api"], n_fraud),
        "is_fraud": [1] * n_fraud,
    }
    
    # Combine
    df = pd.concat([
        pd.DataFrame(normal_data),
        pd.DataFrame(fraud_data)
    ], ignore_index=True)
    
    # Add timestamp
    df["timestamp"] = pd.date_range(
        start="2024-01-01",
        periods=len(df),
        freq="s"
    )
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic data saved to {output_path}")
    logger.info(f"Total: {len(df)}, Normal: {n_normal}, Fraud: {n_fraud}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, default="data/transactions.csv",
                       help="Path to transaction data")
    parser.add_argument("--generate-synthetic", action="store_true",
                       help="Generate synthetic data for testing")
    parser.add_argument("--n-samples", type=int, default=100000,
                       help="Number of synthetic samples")
    parser.add_argument("--fraud-rate", type=float, default=0.001,
                       help="Fraud rate for synthetic data")
    parser.add_argument("--tune", action="store_true",
                       help="Enable hyperparameter tuning")
    parser.add_argument("--no-smote", action="store_true",
                       help="Disable SMOTE")
    args = parser.parse_args()
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        generate_synthetic_data(
            n_samples=args.n_samples,
            fraud_rate=args.fraud_rate,
            output_path=args.data
        )
    
    # Create config
    config = TrainingConfig(
        data_path=args.data,
        enable_tuning=args.tune,
        use_smote=not args.no_smote,
    )
    
    # Run training
    trainer = FraudDetectionTrainer(config)
    trainer.run_full_pipeline(args.data)


if __name__ == "__main__":
    main()
