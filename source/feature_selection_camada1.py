"""
Feature selection - Camada 1 (filtro estatistico, sem modelo).

Aplica tres filtros em sequencia:
  1. Quasi-constantes: remove features cujo valor mais comum domina > 95% das linhas.
  2. Correlacao par-a-par: em cada par com |rho| > 0.95, descarta uma.
  3. Univariada por label: mantem features significativas (ANOVA F-test) para
     pelo menos um dos 4 labels-alvo.

IMPORTANTE (anti-leakage): rode sobre X_train apenas. Use a lista retornada
para filtrar X_train e X_test.

Uso:
    from source.feature_selection_camada1 import aplicar_camada1
    features_mantidas, relatorio = aplicar_camada1(X_train, y_train)
    X_train_c1 = X_train[features_mantidas]
    X_test_c1  = X_test[features_mantidas]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif


def _quasi_constantes(X: pd.DataFrame, dominancia: float = 0.95) -> list[str]:
    """Features onde o valor mais frequente domina > `dominancia` das linhas."""
    n = len(X)
    descartar = []
    for col in X.columns:
        top = X[col].value_counts(dropna=False).iloc[0]
        if top / n > dominancia:
            descartar.append(col)
    return descartar


def _correlacionadas(X: pd.DataFrame, limite: float = 0.95) -> list[str]:
    """Em cada par com |rho| > limite, descarta a segunda coluna (ordem do DataFrame)."""
    corr = X.corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    return [c for c in superior.columns if (superior[c] > limite).any()]


def _univariada_por_label(
    X: pd.DataFrame,
    y: pd.DataFrame,
    metodo: str = "anova",
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, list[str]]:
    """Para cada label, retorna features significativas individualmente."""
    significativas: dict[str, list[str]] = {}
    X_np = X.to_numpy()
    for label in y.columns:
        y_bin = y[label].to_numpy()
        if metodo == "anova":
            _, p = f_classif(X_np, y_bin)
            keep = [c for c, pv in zip(X.columns, p) if pv < alpha]
        elif metodo == "mi":
            mi = mutual_info_classif(X_np, y_bin, random_state=random_state)
            keep = [c for c, m in zip(X.columns, mi) if m > 0]
        else:
            raise ValueError(f"metodo desconhecido: {metodo}")
        significativas[label] = keep
    return significativas


def aplicar_camada1(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dominancia: float = 0.95,
    corr_limite: float = 0.95,
    alpha_univariada: float = 0.05,
    metodo_univariada: str = "anova",
    verbose: bool = True,
) -> tuple[list[str], dict]:
    """
    Aplica os tres filtros em sequencia e retorna (features_finais, relatorio).

    Parametros
    ----------
    X : DataFrame de features (ja sem patient_id, arquivo_wav e labels).
    y : DataFrame com colunas AS, AR, MR, MS (binarias).
    dominancia : fracao maxima do valor mais frequente antes de considerar a
        feature quasi-constante.
    corr_limite : limite absoluto de correlacao de Pearson para remover.
    alpha_univariada : nivel de significancia do F-test por label.
    metodo_univariada : 'anova' (F-test) ou 'mi' (mutual information).
    """
    relatorio: dict = {
        "entrada": {"n_features": X.shape[1], "n_amostras": X.shape[0]},
        "colunas_entrada": list(X.columns),
    }

    drop_qc = _quasi_constantes(X, dominancia=dominancia)
    X1 = X.drop(columns=drop_qc)
    relatorio["passo1_quasi_constantes"] = drop_qc
    if verbose:
        print(f"[1/3] Quasi-constantes (>{int(dominancia*100)}% valor dominante): "
              f"{len(drop_qc)} removidas")
        if drop_qc:
            print(f"      {drop_qc}")
        print(f"      Sobrou: {X1.shape[1]} features")

    drop_corr = _correlacionadas(X1, limite=corr_limite)
    X2 = X1.drop(columns=drop_corr)
    relatorio["passo2_correlacionadas"] = drop_corr
    if verbose:
        print(f"[2/3] Correlacionadas (|rho|>{corr_limite}): "
              f"{len(drop_corr)} removidas")
        if drop_corr:
            print(f"      {drop_corr}")
        print(f"      Sobrou: {X2.shape[1]} features")

    sig = _univariada_por_label(
        X2, y, metodo=metodo_univariada, alpha=alpha_univariada
    )
    uniao = sorted(set().union(*sig.values()))
    drop_univ = [c for c in X2.columns if c not in uniao]
    relatorio["passo3_significativas_por_label"] = sig
    relatorio["passo3_nao_significativas"] = drop_univ
    if verbose:
        print(f"[3/3] Univariada {metodo_univariada} (alpha={alpha_univariada}) "
              f"por label:")
        for lab, feats in sig.items():
            print(f"      {lab}: {len(feats)} features significativas")
        print(f"      Nao significativas em nenhum label: {len(drop_univ)} removidas")
        if drop_univ:
            print(f"      {drop_univ}")
        print(f"      Sobrou: {len(uniao)} features")

    relatorio["features_finais"] = uniao
    relatorio["resumo"] = {
        "entrada": X.shape[1],
        "apos_quasi_constantes": X1.shape[1],
        "apos_correlacao": X2.shape[1],
        "apos_univariada": len(uniao),
    }
    if verbose:
        print(f"\nResumo: {X.shape[1]} -> {X1.shape[1]} -> {X2.shape[1]} "
              f"-> {len(uniao)}")

    return uniao, relatorio


if __name__ == "__main__":
    # Demo: roda sobre dataset_final.csv usando split ingenuo (so para inspecao).
    # Em producao, chame a funcao dentro do notebook APOS o split por paciente.
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    df = pd.read_csv(os.path.join(root, "dataset_final.csv"))
    alvos = ["AS", "AR", "MR", "MS"]
    meta = ["patient_id", "arquivo_wav"]
    X_all = df.drop(columns=meta + alvos)
    y_all = df[alvos]
    print(f"Demo sobre dataset_final.csv: X={X_all.shape}, y={y_all.shape}\n")
    feats, rel = aplicar_camada1(X_all, y_all)
    print(f"\nFeatures finais ({len(feats)}):")
    for f in feats:
        print(f"  - {f}")
