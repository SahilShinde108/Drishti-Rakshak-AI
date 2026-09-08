import numpy as np
import logging
import yaml
import os
import pickle
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using default config.")
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class StackingEnsemble:
    def __init__(self, config: dict = None):
        if config is None:
            config = load_config().get('ensemble', {})
        self.config = config
        self.models = {}
        
        # Load models based on config
        try:
            from catboost import CatBoostClassifier
            self.models['catboost'] = CatBoostClassifier(iterations=100, verbose=0)
        except ImportError:
            logger.warning("CatBoost not installed.")
            
        try:
            from xgboost import XGBClassifier
            self.models['xgboost'] = XGBClassifier(n_estimators=100, eval_metric='mlogloss', objective='multi:softprob')
        except ImportError:
            logger.warning("XGBoost not installed.")
            
        from sklearn.ensemble import RandomForestClassifier
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X_train, y_train, X_val, y_val):
        for name, model in self.models.items():
            logger.info(f"Training base model: {name}")
            if name == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            elif name == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
                
            val_preds = model.predict(X_val)
            acc = accuracy_score(y_val, val_preds)
            logger.info(f"Model {name} Validation Accuracy: {acc:.4f}")
        return self
        
    def predict_proba(self, X):
        probs = []
        for name, model in self.models.items():
            probs.append(model.predict_proba(X))
        
        # Soft voting
        avg_probs = np.mean(probs, axis=0)
        return avg_probs
        
    def predict(self, X):
        avg_probs = self.predict_proba(X)
        ensemble_pred = np.argmax(avg_probs, axis=1)
        
        indiv_preds = {name: model.predict(X) for name, model in self.models.items()}
        
        return {
            'ensemble_pred': ensemble_pred,
            'individual_preds': indiv_preds
        }
        
    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(directory, f"{name}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(model, f)
                
    def load(self, directory: str):
        for name in self.models.keys():
            path = os.path.join(directory, f"{name}.pkl")
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                    
    def get_feature_importance(self) -> dict:
        importances = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances[name] = model.feature_importances_
        return importances

class EnsembleTrainer:
    def __init__(self, config: dict = None):
        self.ensemble = StackingEnsemble(config)
        
    def train_from_embeddings(self, train_embeddings, train_labels, val_embeddings, val_labels):
        # Implement out-of-fold strategy if needed. Currently simple train/val
        logger.info("Training ensemble from embeddings...")
        self.ensemble.train(train_embeddings, train_labels, val_embeddings, val_labels)
        return self.ensemble
