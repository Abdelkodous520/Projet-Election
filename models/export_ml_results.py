"""
Exporte les résultats du Machine Learning pour le dashboard.

Reprend exactement la démarche des notebooks 05 et 06 (même variables, même découpage
75/25, random_state=42) et écrit dans  <racine>/models/ :
    metrics.csv             une ligne par modèle (MAE, RMSE, R2 sur le test, MAE en validation croisée)
    feature_importance.csv  importance par permutation du meilleur modèle
    predictions.csv         valeur réelle / prédite pour les communes du jeu de test
    best_model.joblib       meilleur modèle entraîné (pipeline complet)

Lancement (racine du projet, .venv actif) :  python Script/export_ml_results.py
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from kharita_pipeline import build_features, clean_data, load_raw  # noqa: E402

OUT = ROOT / "models"
SEED = 42


class LastElectionBaseline(BaseEstimator, RegressorMixin):
    """Référence naïve : la participation de 2021 serait celle de 2015 (colonne turnout_2015)."""

    def fit(self, X, y):
        self.fallback_ = float(np.nanmedian(y))
        return self

    def predict(self, X):
        return X["turnout_2015"].fillna(self.fallback_).to_numpy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, _, party_cols = clean_data(load_raw())
    features = build_features(df, party_cols)
    data = features.dropna(subset=["turnout_2021"])
    y = data["turnout_2021"]
    X = data.drop(columns="turnout_2021")
    # colonnes entièrement vides (partis absents en 2015) : inutiles pour l'apprentissage
    X = X.loc[:, X.notna().any()]
    cat = ["region"]
    num = [c for c in X if c not in cat]
    print(f"{len(X)} communes, {X.shape[1]} variables (cible : participation 2021)")

    def preprocess():
        return ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(handle_unknown="ignore")), cat),
        ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    models = {
        "Référence 2015": LastElectionBaseline(),
        "Moyenne": DummyRegressor(strategy="mean"),
        "Linear Regression": make_pipeline(preprocess(), LinearRegression()),
        "Random Forest": make_pipeline(preprocess(), RandomForestRegressor(
            n_estimators=250, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(preprocess(), GradientBoostingRegressor(random_state=SEED)),
        "HistGradientBoosting": make_pipeline(preprocess(), HistGradientBoostingRegressor(random_state=SEED)),
    }

    rows, fitted, preds = [], {}, {}
    for name, est in models.items():
        est.fit(Xtr, ytr)
        p = est.predict(Xte)
        fitted[name], preds[name] = est, p
        cv_mae = -cross_val_score(est, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=1).mean()
        rows.append({
            "modele": name,
            "MAE": mean_absolute_error(yte, p),
            "RMSE": mean_squared_error(yte, p) ** 0.5,
            "R2": r2_score(yte, p),
            "MAE_validation_croisee_5": cv_mae,
        })
        print(f"  {name:<22} MAE {rows[-1]['MAE']:.3f}  R2 {rows[-1]['R2']:.3f}")
    metrics = pd.DataFrame(rows).sort_values("MAE").round(4)
    metrics.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")

    # meilleur modèle appris (on exclut les deux références naïves)
    learned = metrics[~metrics["modele"].isin(["Référence 2015", "Moyenne"])]
    best = learned.iloc[0]["modele"]
    print("Meilleur modèle :", best)

    imp = permutation_importance(fitted[best], Xte, yte, n_repeats=10, random_state=SEED,
                                 scoring="neg_mean_absolute_error", n_jobs=1)
    (pd.DataFrame({"variable": X.columns, "importance": imp.importances_mean})
       .sort_values("importance", ascending=False).round(5)
       .to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig"))

    pred = pd.DataFrame({
        "commune_code": Xte.index, "region": Xte["region"].to_numpy(),
        "y_true": yte.to_numpy(), "y_pred": preds[best], "modele": best,
    })
    pred["erreur_absolue"] = (pred["y_pred"] - pred["y_true"]).abs()
    pred.round(4).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    joblib.dump(fitted[best], OUT / "best_model.joblib")
    print("Résultats écrits dans", OUT)


if __name__ == "__main__":
    main()
