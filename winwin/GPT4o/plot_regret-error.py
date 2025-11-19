import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast


def add_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regret bound を構成する3成分を DataFrame に追加する。
    comp_pol  : L * E_pol
    comp_pred : L * C * E_pred
    comp_mis  : L * E_mis
    """
    df = df.copy()
    df['comp_pol']  = df['L_eff'] * df['policy_error']
    df['comp_pred'] = df['L_eff'] * df['C'] * df['ece']      # ece = E_pred
    df['comp_mis']  = df['L_eff'] * df['mismatch']           # mismatch = E_mis
    return df


def summarize_condition(df: pd.DataFrame, condition_name: str) -> dict:
    df = add_components(df)
    model = df['model'].iloc[0]

    # ★ trial ごとに平均 regret を取る
    if 'trial' in df.columns:
        per_ep_regret = df.groupby('trial')['regret_t'].mean()
    else:
        per_ep_regret = pd.Series([df['regret_t'].mean()])

    mean_regret = per_ep_regret.mean()      # エピソード平均の平均
    std_regret  = per_ep_regret.std()       # エピソード間のばらつき

    mean_slack  = df['slack'].mean() if 'slack' in df.columns else np.nan

    mean_comp_pol  = df['comp_pol'].mean()
    mean_comp_pred = df['comp_pred'].mean()
    mean_comp_mis  = df['comp_mis'].mean()

    total_bound = mean_comp_pol + mean_comp_pred + mean_comp_mis
    if total_bound > 0:
        ratio_pol  = mean_comp_pol  / total_bound
        ratio_pred = mean_comp_pred / total_bound
        ratio_mis  = mean_comp_mis  / total_bound
    else:
        ratio_pol = ratio_pred = ratio_mis = np.nan

    if 'rho' in df.columns:
        meaningful = df[df['regret_t'] > 0.01]
        mean_rho = meaningful['rho'].mean() if not meaningful.empty else np.nan
    else:
        mean_rho = np.nan

    return {
        "model": model,
        "condition": condition_name,
        "mean_regret": mean_regret,
        "std_regret": std_regret,          # ★ 追加
        "mean_slack": mean_slack,
        "mean_comp_pol": mean_comp_pol,
        "mean_comp_pred": mean_comp_pred,
        "mean_comp_mis": mean_comp_mis,
        "ratio_pol": ratio_pol,
        "ratio_pred": ratio_pred,
        "ratio_mis": ratio_mis,
        "mean_rho": mean_rho,
    }



def plot_results_from_df(df, p_y1_mean=0.5, save_prefix=""):
    """
    1 条件分について:
      - 上: regret vs bound (mean ± std)
      - 下: bound 分解 (comp_pred / comp_mis / comp_pol の時系列 stackplot)
    を保存する。
    """
    df = add_components(df)
    model = df['model'].iloc[0]

    # ラウンドごとに mean / std
    df_mean = df.groupby('t').mean(numeric_only=True)
    df_std  = df.groupby('t').std(numeric_only=True)

    rounds = df_mean.index
    regret_mean = df_mean['regret_t']
    bound_mean  = df_mean['regret_bound_t']
    regret_std  = df_std['regret_t']
    bound_std   = df_std['regret_bound_t']

    # --- 図1: regret vs bound ---
    plt.figure(figsize=(12, 5))

    # regret
    plt.plot(rounds, regret_mean, label="Actual Regret (mean)", color="blue")
    plt.fill_between(rounds,
                     regret_mean - regret_std,
                     regret_mean + regret_std,
                     color="blue", alpha=0.2, label="Regret (±1 std)")

    # bound
    plt.plot(rounds, bound_mean, label="Regret Bound (mean)", color="orange")
    plt.fill_between(rounds,
                     bound_mean - bound_std,
                     bound_mean + bound_std,
                     color="orange", alpha=0.2, label="Bound (±1 std)")

    plt.title(f"Per-Round Regret vs. Bound ({model}, Opponent Mean={p_y1_mean})")
    plt.xlabel("Round (t)")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"regret_{save_prefix}_plot.png")
    plt.close()

    # --- 図2: bound 分解 (時系列 stackplot) ---
    comp_pol_mean  = df_mean['comp_pol']
    comp_pred_mean = df_mean['comp_pred']
    comp_mis_mean  = df_mean['comp_mis']

    plt.figure(figsize=(12, 5))
    plt.stackplot(
        rounds,
        comp_pred_mean,   # 下: Prediction Error
        comp_mis_mean,    # 中: Policy Mismatch
        comp_pol_mean,    # 上: Policy Error
        labels=[
            r'Prediction Error ($L \cdot C \cdot E_{\mathrm{pred}}$)',
            r'Policy Mismatch ($L \cdot E_{\mathrm{mis}}$)',
            r'Policy Error ($L \cdot E_{\mathrm{pol}}$)'
        ],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    plt.title(f"Decomposition of Regret Bound ({model}, Opponent Mean={p_y1_mean})")
    plt.xlabel("Round (t)")
    plt.ylabel("Bound Component Value")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"error_components_{save_prefix}_plot.png")
    plt.close()


def summarize_condition(df: pd.DataFrame, condition_name: str) -> dict:
    df = add_components(df)
    model = df['model'].iloc[0]

    # ★ trial ごとに平均 regret を取る
    if 'trial' in df.columns:
        per_ep_regret = df.groupby('trial')['regret_t'].mean()
    else:
        per_ep_regret = pd.Series([df['regret_t'].mean()])

    mean_regret = per_ep_regret.mean()      # エピソード平均の平均
    std_regret  = per_ep_regret.std()       # エピソード間のばらつき

    mean_slack  = df['slack'].mean() if 'slack' in df.columns else np.nan

    mean_comp_pol  = df['comp_pol'].mean()
    mean_comp_pred = df['comp_pred'].mean()
    mean_comp_mis  = df['comp_mis'].mean()

    total_bound = mean_comp_pol + mean_comp_pred + mean_comp_mis
    if total_bound > 0:
        ratio_pol  = mean_comp_pol  / total_bound
        ratio_pred = mean_comp_pred / total_bound
        ratio_mis  = mean_comp_mis  / total_bound
    else:
        ratio_pol = ratio_pred = ratio_mis = np.nan

    if 'rho' in df.columns:
        meaningful = df[df['regret_t'] > 0.01]
        mean_rho = meaningful['rho'].mean() if not meaningful.empty else np.nan
    else:
        mean_rho = np.nan

    return {
        "model": model,
        "condition": condition_name,
        "mean_regret": mean_regret,
        "std_regret": std_regret,          # ★ 追加
        "mean_slack": mean_slack,
        "mean_comp_pol": mean_comp_pol,
        "mean_comp_pred": mean_comp_pred,
        "mean_comp_mis": mean_comp_mis,
        "ratio_pol": ratio_pol,
        "ratio_pred": ratio_pred,
        "ratio_mis": ratio_mis,
        "mean_rho": mean_rho,
    }


def plot_summary_for_model(summary_df: pd.DataFrame, save_prefix: str = ""):
    model = summary_df['model'].iloc[0]
    conditions = summary_df['condition'].tolist()
    x = np.arange(len(conditions))

    # --- 図A: mean regret のバー + エラーバー ---
    plt.figure(figsize=(6, 4))
    y    = summary_df['mean_regret'].values
    yerr = summary_df.get('std_regret', pd.Series([0]*len(summary_df))).values

    plt.bar(x, y, width=0.6, yerr=yerr, capsize=5)
    plt.xticks(x, conditions)
    plt.ylabel("Mean Regret")
    plt.title(f"Mean Regret by Condition ({model})")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}mean_regret_bar.png")
    plt.close()

    # --- 図B: ratio の stacked bar + % ラベル ---
    width = 0.6
    pred = summary_df['ratio_pred'].values
    mis  = summary_df['ratio_mis'].values
    pol  = summary_df['ratio_pol'].values

    plt.figure(figsize=(6, 4))
    p1 = plt.bar(x, pred, width=width, label="Pred", color="#1f77b4")
    p2 = plt.bar(x, mis,  width=width, bottom=pred,        label="Mis",  color="#ff7f0e")
    p3 = plt.bar(x, pol,  width=width, bottom=pred+mis,    label="Pol",  color="#2ca02c")

    plt.xticks(x, conditions)
    plt.ylabel("Fraction of Bound")
    plt.ylim(0, 1.05)
    plt.title(f"Decomposition Ratios by Condition ({model})")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ★ ここで各セグメントに % を書く
    for i in range(len(x)):
        # Pred
        if pred[i] > 0.03:  # 小さすぎるときは文字を省略（調整可）
            plt.text(
                x[i],
                pred[i] / 2,
                f"{pred[i]*100:.1f}%",
                ha="center", va="center", fontsize=6, color="black"
            )
        # Mis
        if mis[i] > 0.03:
            plt.text(
                x[i],
                pred[i] + mis[i] / 2,
                f"{mis[i]*100:.1f}%",
                ha="center", va="center", fontsize=9, color="black"
            )
        # Pol
        if pol[i] > 0.03:
            plt.text(
                x[i],
                pred[i] + mis[i] + pol[i] / 2,
                f"{pol[i]*100:.1f}%",
                ha="center", va="center", fontsize=9, color="black"
            )
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}bound_ratio_bar.png")
    plt.close()



def main_plotter():
    """
    各 CSV (Baseline / Targeted / NonTargeted) を読み込み、
      - coverage・平均値などを print
      - per-round 図を保存
      - summary_df を作って保存
      - summary からバー図を描画
    """
    csv_files_to_plot = {
        "Baseline":    "results_none.csv",
        "Targeted":    "results_target.csv",
        "NonTargeted": "results_non_target.csv"
    }

    print("--- グラフ生成スクリプト (from CSV) を開始 ---")

    summaries = []

    for cond_name, filename in csv_files_to_plot.items():
        print(f"\n--- {cond_name} ({filename}) の処理を開始 ---")

        try:
            df = pd.read_csv(filename)

            # coverage
            if 'covered' in df.columns:
                total_rounds = len(df)
                covered_rounds = df['covered'].sum()
                coverage_rate = (covered_rounds / total_rounds) * 100
                print(f"  カバレッジ率: {coverage_rate:.2f}%  ({covered_rounds} / {total_rounds} ラウンド)")
                if coverage_rate < 100.0:
                    print("  ★警告: カバレッジ率が100%未満です。L, C の設定を再確認すべきかもしれません。")
            else:
                print("  カバレッジ率: 'covered' カラムが見つかりません。")

            # 平均 regret
            if 'regret_t' in df.columns:
                mean_regret = df['regret_t'].mean()
                print(f"  平均 Regret: {mean_regret:.4f}")
            else:
                print("  平均 Regret: 'regret_t' カラムが見つかりません。")

            # 平均 slack
            if 'slack' in df.columns:
                mean_slack = df['slack'].mean()
                print(f"  平均 Slack (bound - regret): {mean_slack:.4f}")
            else:
                print("  平均 Slack: 'slack' カラムが見つかりません。")

            # 平均 rho
            if 'rho' in df.columns and 'regret_t' in df.columns:
                meaningful = df[df['regret_t'] > 0.01]
                if not meaningful.empty:
                    mean_rho = meaningful['rho'].mean()
                    print(f"  平均 Rho (bound / regret): {mean_rho:.4f}  (regret > 0.01 のラウンドのみ)")
                else:
                    print("  平均 Rho: regret > 0.01 のラウンドがありませんでした。")
            else:
                print("  平均 Rho: 'rho' または 'regret_t' カラムが見つかりません。")

            # opponent の平均 P(Y=1)
            if 'mu_true_mean' in df.columns:
                try:
                    mu = ast.literal_eval(str(df['mu_true_mean'].iloc[0]))
                    p_y1_mean = float(mu[1])
                    p_y1_info = f"{p_y1_mean:.2f}"
                except Exception:
                    p_y1_info = "Error"
            else:
                p_y1_info = "N/A"

            # per-round 図
            file_prefix = f"{cond_name.lower()}_"
            plot_results_from_df(df, p_y1_mean=p_y1_info, save_prefix=file_prefix)

            # summary 行を作成
            summary_row = summarize_condition(df, cond_name)
            summaries.append(summary_row)

        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません {filename}。スキップします。")
        except Exception as e:
            print(f"エラー: {filename} の処理中に問題が発生しました: {e}")

    # 全条件の summary を DataFrame 化
    if summaries:
        summary_df = pd.DataFrame(summaries)
        model = summary_df['model'].iloc[0]
        print(f"\n=== Summary ({model}) ===")
        print(summary_df)
        summary_df.to_csv("summary_regret_components.csv", index=False)

        # ★ 論文用に使うテーブル（絶対値と割合）だけ抜き出す
        table_cols = [
            "condition",
            "mean_regret",
            "mean_comp_pred",
            "mean_comp_mis",
            "mean_comp_pol",
            "ratio_pred",
            "ratio_mis",
            "ratio_pol",
        ]
        table_df = summary_df[table_cols].copy()

        print("\n=== Table for paper (regret & components) ===")
        print(table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        table_df.to_csv("table_regret_components_for_paper.csv", index=False)

        # 条件ごとのバー図
        plot_summary_for_model(summary_df, save_prefix="summary_")

    print("\n--- 全てのグラフ生成が完了しました ---")


if __name__ == "__main__":
    main_plotter()