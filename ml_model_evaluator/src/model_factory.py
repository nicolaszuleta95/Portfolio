from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class ModelFactory:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def get_default_models(self):
        return {
            'Logistic Regression': LogisticRegression(
                solver='liblinear', class_weight='balanced', random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced', random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf', probability=True, class_weight='balanced',
                C=1.0, gamma='scale', random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100, use_label_encoder=False, eval_metric='logloss',
                scale_pos_weight=5, random_state=self.random_state
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100, class_weight='balanced', random_state=self.random_state
            )
        }

    def build_model(self, name, params):
        if name == 'Logistic Regression':
            return LogisticRegression(
                solver='liblinear',
                C=params.get('C', 1.0),
                class_weight='balanced',
                random_state=self.random_state
            )
        elif name == 'Random Forest':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                min_samples_split=params.get('min_samples_split', 2),
                min_samples_leaf=params.get('min_samples_leaf', 1),
                class_weight='balanced',
                random_state=self.random_state
            )
        elif name == 'SVM':
            return SVC(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                gamma=params.get('gamma', 'scale'),
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )
        elif name == 'XGBoost':
            return XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                scale_pos_weight=params.get('scale_pos_weight', 1.0),
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=self.random_state
            )
        elif name == 'LightGBM':
            return LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                learning_rate=params.get('learning_rate', 0.1),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                min_child_samples=params.get('min_child_samples', 20),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Modelo no soportado: {name}")
