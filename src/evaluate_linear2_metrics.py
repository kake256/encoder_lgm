# ファイル名: evaluate_linear2_metrics.py
# 内容: 自己嗜好性(Self-Preference), 相関, 混同行列の類似度計算

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from evaluate_linear2_config import GENERATOR_SOURCE_MAP, STANDARD_EVALUATOR

def calculate_self_preference(final_results, output_dir):
    """
    自己選好性(Self-Preference)は、全評価モデルの結果を横断して計算するため、
    特定の target_evaluator に依存しません。
    """
    df = pd.DataFrame(final_results)
    if "Accuracy" not in df.columns:
        return
    try:
        piv = df.pivot(index="Generator", columns="Evaluator", values="Accuracy")
        preference_results = []
        for gen_name in piv.index:
            source_models = GENERATOR_SOURCE_MAP.get(gen_name, [])
            for source_model in source_models:
                if source_model not in piv.columns:
                    continue
                self_score = piv.loc[gen_name, source_model]
                other_cols = [c for c in piv.columns if c != source_model]
                if not other_cols:
                    continue
                avg_other_score = piv.loc[gen_name, other_cols].mean()
                gap = self_score - avg_other_score
                preference_results.append({
                    "Generator": gen_name,
                    "Source_Model": source_model,
                    "Self_Score": float(self_score),
                    "Avg_Other_Score": float(avg_other_score),
                    "Self_Preference_Gap": float(gap),
                })
        if preference_results:
            pd.DataFrame(preference_results).to_csv(
                os.path.join(output_dir, "self_preference_policy.csv"),
                index=False
            )
    except Exception as e:
        print(f"[Warning] Failed to calc self preference: {e}")

def calculate_complex_correlations(acc_store, experiments, output_dir, target_evaluator=STANDARD_EVALUATOR):
    """
    target_evaluator を基準にしてスピアマン相関を計算します。
    """
    results = []
    # 指定された target_evaluator のデータセット内に Real_Baseline があるか確認
    if target_evaluator not in acc_store or "Real_Baseline" not in acc_store[target_evaluator]:
        return

    for gen_name in experiments.keys():
        source_models = GENERATOR_SOURCE_MAP.get(gen_name, [])
        for source_model in source_models:
            if source_model not in acc_store:
                continue

            # Policy A: 指定されたターゲットモデルにおける (合成データ vs 実画像) の相関
            vec_target_syn = acc_store[target_evaluator].get(gen_name)
            vec_target_real = acc_store[target_evaluator].get("Real_Baseline")
            
            # Policy D: 生成元モデル(Source)における (合成データ vs 実画像) の相関
            vec_source_syn = acc_store[source_model].get(gen_name)
            vec_source_real = acc_store[source_model].get("Real_Baseline")

            row = {"Generator": gen_name, "Target_Source_Model": source_model}

            # --- Policy A 計算 ---
            if vec_target_syn is not None and vec_target_real is not None:
                a = list(vec_target_syn)
                b = list(vec_target_real)
                len_diff = len(b) - len(a)
                if len_diff > 0:
                    a = a + [0] * len_diff
                corr, _ = spearmanr(a, b)
                # 列名に評価モデル名を含める
                row[f"Policy_A ({target_evaluator}_Syn vs Real)"] = float(corr)

            # --- Policy D 計算 ---
            if vec_source_syn is not None and vec_source_real is not None:
                a = list(vec_source_syn)
                b = list(vec_source_real)
                len_diff = len(b) - len(a)
                if len_diff > 0:
                    a = a + [0] * len_diff
                corr, _ = spearmanr(a, b)
                row["Policy_D (Source_Syn vs Source_Real)"] = float(corr)

            results.append(row)

    if results:
        # ファイル名に target_evaluator を含めて保存 (上書き防止)
        filename = f"complex_correlation_policies_{target_evaluator}.csv"
        pd.DataFrame(results).to_csv(
            os.path.join(output_dir, filename),
            index=False
        )

def calculate_confusion_matrix_similarity(cm_store, experiments, output_dir, target_evaluator=STANDARD_EVALUATOR):
    """
    target_evaluator を基準にして混同行列の類似度を計算します。
    """
    if target_evaluator not in cm_store:
        return
    
    generators = list(cm_store[target_evaluator].keys())

    sim_matrix_data = []
    for gen1 in generators:
        row = {"Generator": gen1}
        cm1 = cm_store[target_evaluator][gen1].flatten()
        norm1 = np.linalg.norm(cm1)
        
        for gen2 in generators:
            cm2 = cm_store[target_evaluator][gen2].flatten()
            norm2 = np.linalg.norm(cm2)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(cm1, cm2) / (norm1 * norm2))
            else:
                sim = 0.0
            row[gen2] = sim
        sim_matrix_data.append(row)

    # ファイル名に target_evaluator を含めて保存
    filename = f"confusion_matrix_similarity_{target_evaluator}.csv"
    pd.DataFrame(sim_matrix_data).to_csv(
        os.path.join(output_dir, filename),
        index=False
    )