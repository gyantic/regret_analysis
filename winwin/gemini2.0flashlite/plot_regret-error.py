import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_results_from_df(df, model_name_tag="model", model="gpt-4o", p_y1_mean=0.5, save_prefix=""):
    """
    DataFrameからリグレットとエラー成分のグラフを描画する
    """
    
    # --- 1. データの前処理 ---
    # バウンドを構成する3つの成分を計算
    # (play_round_llm の計算ロジック L*(Epol + C*Epred + Emis) に基づく)
    df['comp_pol']  = df['L_eff'] * df['policy_error']
    df['comp_pred'] = df['L_eff'] * df['C'] * df['ece'] # ece = E_pred
    df['comp_mis']  = df['L_eff'] * df['mismatch']     # mismatch = E_mis
    
    # ラウンド(t) ごとに平均と標準偏差を計算
    # (trials をまたいですべてのラウンドtの統計を取る)
    df_mean = df.groupby('t').mean(numeric_only=True)
    df_std = df.groupby('t').std(numeric_only=True)

    # グラフ用のx軸 (ラウンド 1 から T)
    rounds = df_mean.index
    
    # 平均値
    regret_mean = df_mean['regret_t']
    bound_mean  = df_mean['regret_bound_t']
    
    # 標準偏差
    regret_std = df_std['regret_t']
    bound_std  = df_std['regret_bound_t']

    # --- 2. グラフ1: リグレット vs バウンド (論文の上の図) ---
    plt.figure(figsize=(12, 5))
    
    # リグレット (平均)
    plt.plot(rounds, regret_mean, label="Actual Regret (mean)", color="blue")
    # リグレット (±1 std dev)
    plt.fill_between(rounds, 
                     regret_mean - regret_std, 
                     regret_mean + regret_std, 
                     color="blue", alpha=0.2, label="Regret (±1 std)")
                     
    # バウンド (平均)
    plt.plot(rounds, bound_mean, label="Regret Bound (mean)", color="orange")
    # バウンド (±1 std dev)
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

    # --- 3. グラフ2: エラー成分の積み上げ (論文の下の図) ---
    
    # 積み上げグラフ用のデータ (3成分の平均値)
    comp_pol_mean  = df_mean['comp_pol']
    comp_pred_mean = df_mean['comp_pred']
    comp_mis_mean  = df_mean['comp_mis']

    plt.figure(figsize=(12, 5))
    
    # 積み上げグラフ
    plt.stackplot(rounds, 
                  comp_pred_mean,  # 下: E_pred (Prediction Error)
                  comp_mis_mean,   # 中: E_mis (Policy Mismatch)
                  comp_pol_mean,   # 上: E_pol (Policy Error)
                  labels=[
                      f'Prediction Error ($L \cdot C \cdot E_{{pred}}$)', 
                      f'Policy Mismatch ($L \cdot E_{{mis}}$)',
                      f'Policy Error ($L \cdot E_{{pol}}$)'
                  ],
                  colors=["#1f77b4", "#ff7f0e", "#2ca02c"]) # 青、オレンジ、緑
    
    plt.title(f"Decomposition of Regret Bound ({model}, Opponent Mean={p_y1_mean})")
    plt.xlabel("Round (t)")
    plt.ylabel("Bound Component Value")
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"error_components_{save_prefix}_plot.png")
    plt.close()

def main_plotter():
    """
    指定されたCSVファイルを読み込み、それぞれ
    「リグレット」と「エラー成分」のグラフを描画する。
    """
    
    # 前のステップで保存したCSVファイルと、グラフのタイトル/プレフィックスを定義
    csv_files_to_plot = {
        "Baseline": "results_none.csv",
        "Targeted": "results_target.csv",
        "NonTargeted": "results_non_target.csv",
        "BothTargeted": "results_both_target.csv"
    }

    print("--- グラフ生成スクリプト (from CSV) を開始 ---")

    for key, filename in csv_files_to_plot.items():
        print(f"\n--- {key} ({filename}) の処理を開始 ---")
        
        try:
            # 1. CSVファイルを読み込む
            df = pd.read_csv(filename)

            if 'covered' in df.columns:
                total_rounds = len(df)
                covered_rounds = df['covered'].sum() # True の数を合計
                coverage_rate = (covered_rounds / total_rounds) * 100 # パーセンテージに
                
                print(f"  カバレッジ率: {coverage_rate:.2f}%  ({covered_rounds} / {total_rounds} ラウンド)")
                
                # レビューアへの重要な反論材料：もしカバレッジが100%未満なら
                if coverage_rate < 100.0:
                    print(f"  ★警告: カバレッジ率が100%未満です。バウンドの計算(L, C)に誤りがある可能性があります。")
            else:
                print("  カバレッジ率: 'covered' カラムが見つかりません。")
            
            
            # ★★★ 2. 平均リグレット (ここから) ★★★
            if 'regret_t' in df.columns:
                mean_regret = df['regret_t'].mean()
                print(f"  平均 Regret (実際のリグレット): {mean_regret:.4f}")
            else:
                print("  平均 Regret: 'regret_t' カラムが見つかりません。")
            # ★★★ (ここまで) ★★★
            
            # slack の計算
            if 'slack' in df.columns:
                mean_slack = df['slack'].mean()
                print(f"  平均 Slack (バウンド - リグLET): {mean_slack:.4f}")
            else:
                print("  平均 Slack: 'slack' カラムが見つかりません。")

            # rho の計算 (リグレットが0.01より大きいラウンドのみ対象)
            if 'rho' in df.columns and 'regret_t' in df.columns:
                # リグレットが小さすぎるラウンドを除外 (rhoが発散するため)
                meaningful_rounds_df = df[df['regret_t'] > 0.01]
                
                if not meaningful_rounds_df.empty:
                    mean_rho = meaningful_rounds_df['rho'].mean()
                    print(f"  平均 Rho (バウンド / リグレット): {mean_rho:.4f}  (regret > 0.01 のラウンドのみ)")
                else:
                    print("  平均 Rho: リグレットが0.01を超えるラウンドがありませんでした。")
            else:
                print("  平均 Rho: 'rho' または 'regret_t' カラムが見つかりません。")
            
            # 2. グラフタイトル用の情報をCSVから取得
            # (ログには全行同じ情報が入っていると仮定し、最初の行から取得)
            model_name = df['model'].iloc[0]
            
            # mu_true_mean は [0.5, 0.5] のようなリスト文字列として保存されているため
            # eval でリストに変換し、[1]番目 (P(Defect)) を取得
            try:
                # mu_true_mean がログに含まれていることを確認
                if 'mu_true_mean' in df.columns:
                    p_y1_mean = pd.eval(df['mu_true_mean'].iloc[0])[1] # P(Y=1) の平均
                    p_y1_info = f"{p_y1_mean:.2f}"
                else:
                    p_y1_info = "N/A" # 古いCSV用
            except:
                p_y1_info = "Error"
            
            # グラフのタイトルとファイル名を生成
            title_tag = f"{model_name} ({key})" # 例: "gpt-4o (Baseline)"
            file_prefix = f"{key.lower()}_"      # 例: "baseline_"
            
            # 3. グラフ描画関数を呼び出す
            plot_results_from_df(
                df,
                model_name_tag=title_tag,
                model="GPT-4o",
                p_y1_mean=p_y1_info,
                save_prefix=file_prefix
            )

        except FileNotFoundError:
            print(f"エラー: ファイルが見つかりません {filename}。スキップします。")
        except Exception as e:
            print(f"エラー: {filename} の処理中に問題が発生しました: {e}")

    print("\n--- 全てのグラフ生成が完了しました ---")

if __name__ == "__main__":
    # このスクリプトを直接実行した場合、プロット処理を開始する
    main_plotter()