import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


def load_features():
    path = "data/features_spark.csv"
    if not os.path.exists(path):
        path = "data/sample_cases.csv"
        df = pd.read_csv(path, parse_dates=["data"])
        df = df.sort_values(["municipio", "data"])
        df["casos_4s"] = (
            df.groupby("municipio")["casos"]
            .rolling(4, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )
        df["casos_4_major"] = df.groupby("municipio")["casos_4s"].shift(-4)
        thr = df["casos_4s"].quantile(0.8)
        df["surto_4s"] = (df["casos_4_major"] >= thr).astype(int)
    else:
        df = pd.read_csv(path, parse_dates=["data"])
    df = df.dropna(subset=["surto_4s"])
    return df


def treinamento(df):
    feat = ['casos_4s', 'chuva', 'temp', 'umid']
    X = df[feat].fillna(0)
    y = df['surto_4s'].astype(int)

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    print('Acurácia no conjunto de teste:')
    print('-------')
    print('ROC', roc_auc_score(yte, proba))
    print(classification_report(yte, (proba >= 0.5).astype(int)))
    print('-------')
    os.makedirs('modelo', exist_ok=True)
    joblib.dump(clf, 'modelo/surtos.pkl')
    return clf


if __name__ == '__main__':
    df = load_features()
    treinamento(df)
