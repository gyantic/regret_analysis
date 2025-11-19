from openai import OpenAI
from dilemma import *
import numpy as np
import pandas as pd, os, json
import matplotlib.pyplot as plt



#LLM用のhelper関数
def call_llm_and_parse(model: str, temperature: float):
    """
    LLM から以下の JSON を受け取る想定：
    {
      "phi_hat": [11 floats in [0,1]],
      "pi":      [11 floats >=0, sum=1.0],
      "reasoning": "..."  # 任意
    }
    """
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[],
    )
    txt = resp.choices[0].message.content
    return txt

def sample_offer(pi: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    j = int(rng.choice(len(pi), p=np.asarray(pi, float)))
    return j, j

def _soft_ref_from_Qhat(Q_hat, beta):
    z = np.logaddexp.reduce(beta * Q_hat)
    return np.exp(beta * Q_hat - z)

def softmax_beta(x):
    x = np.asarray(x, float)
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()

def fit_beta_f(Q_hat, pi):
    betas = np.logspace(-3, 3, 400)
    pi = np.clip(pi, 1e-12, 1); pi = pi / pi.sum()
    best = betas[0]; best_kl = 1e9
    for b in betas:
        s = softmax_beta(b * Q_hat)
        kl = float(np.sum(pi * (np.log(pi) - np.log(np.clip(s,1e-12,1.0)))))
        if kl < best_kl: best_kl, best = kl, b
    return float(best)

def make_s_hat(mu_hat, offers, T_ref=0.5, score_type="mu"):
    R = 100.0 - np.asarray(offers, float)
    if score_type == "mu":
        S = np.asarray(mu_hat, float)
    elif score_type == "muR":
        S = np.asarray(mu_hat, float) * R
    else:
        raise ValueError("score_type must be 'mu' or 'muR'")
    beta = 1.0 / max(1e-8, T_ref)
    return softmax_beta(beta * S), beta, R

def lipschitz_C(beta, score_type, R):
    if score_type == "mu":      # s = softmax(beta * μ̂)
        return float(beta / 4.0)
    elif score_type == "muR":   # s = softmax(beta * (μ̂⊙R))
        return float((beta / 4.0) * np.max(R))
    else:
        raise ValueError


def bound_soft_observable(pi, phi_hat, offers, ece_t,
                          L=100.0, C=1.0,
                          beta_mode="from_temp", temperature=None):
    """観測のみ: pi, phi_hat, offers, ece_t → soft上界"""
    offers  = np.asarray(offers, float)
    pi      = np.clip(np.asarray(pi, float), 1e-12, 1.0)
    pi     /= pi.sum()
    Q_hat   = np.asarray(phi_hat, float) * (100.0 - offers)

    # β の決め方
    if beta_mode == "from_temp" and (temperature is not None):
        beta = 1.0 / max(1e-6, float(temperature))
    else:
        betas = np.logspace(-3, 3, 400)
        def kl(b):
            s = _soft_ref_from_Qhat(Q_hat, b)
            q = np.clip(s, 1e-12, 1.0)
            return float(np.sum(pi * (np.log(pi) - np.log(q))))
        beta = float(betas[int(np.argmin([kl(b) for b in betas]))])

    s = _soft_ref_from_Qhat(Q_hat, beta)                    # s_beta
    kl_soft = float(np.sum(pi * (np.log(pi) - np.log(np.clip(s,1e-12,1.0)))))
    soft_gap = float(Q_hat.max() - float(np.sum(s * Q_hat)))
    return kl_soft, soft_gap, beta


#一ラウンド実行
def play_round_llm(
    env,
    call_llm_fn,
    model: str,
    temperature: float,
    rng: np.random.Generator,
    *,
    L: float = 3.0,        # 固定 L に使う既定値
    C: float = 1.0,          # f のリプシッツ既定値
    use_range_L: bool = False,  # True なら L := range(Q_true) を使用
    extra_ctx=None,
    T_ref: float = 0.7,
    score_type: str = "muR",
    eps_ratio: float = 1e-9,
    payoff_x,
):
    ctx = {
        "actions": [0,1],
        "model": model,
        "temperature": temperature,
    }
    ctx["payoff_x"] = payoff_x
    # extra_ctx に含まれる履歴情報（今回は相手の行動履歴）を ctx に追加
    if extra_ctx:
        ctx.update(extra_ctx)


    out = call_llm_fn(ctx)
    phi_hat = np.asarray(out["phi_hat"], float)
    phi_hat = np.clip(phi_hat, 1e-12, 1.0); phi_hat /= phi_hat.sum()
    pi      = np.asarray(out["pi"], float)
    pi = np.clip(pi, 1e-12, 1.0); pi /= pi.sum()
    assert phi_hat.shape == (2,), f"phi_hat len={len(phi_hat)}"
    assert pi.shape == (2,), f"pi len={len(pi)}"
    assert np.isclose(pi.sum(), 1.0), "pi must sum to 1"
    #mu_true = np.asarray(getattr(env, "mu", [0.5, 0.5]), float)
    #mu_true = np.clip(mu_true, 0.0, 1.0); mu_true /= max(1e-12, mu_true.sum())


    # 自分の行動 a_t と相手の実現行動 b_t
    a_t = int(rng.choice(2, p=pi))
    out_env = env.step(a_t)          # 環境が相手行動 b_t を返す想定
    b_t = int(out_env["b"])          # {0,1}
    mu_true_noise = np.asarray(out_env["mu_true"], float)
    mu_true_mean = np.array([1.0 - env.p_y1_mean, env.p_y1_mean])

    R, S, T, P = env.payoff_x['R'], env.payoff_x['S'], env.payoff_x['T'], env.payoff_x['P']
    def u(a, b):
        if a == 0 and b == 0: return R
        if a == 0 and b == 1: return S
        if a == 1 and b == 0: return T
        return P

    #実際のregretの計算
    """
    u_t = u(a_t, b_t)
    u_best = max(u(0,b_t), u(1,b_t))
    regret = float(u_best-u_t)
    """

    # 真値と各種量
    Q_true = np.array([mu_true_mean[0]*R + mu_true_mean[1]*S,
                       mu_true_mean[0]*T + mu_true_mean[1]*P], float)
    EV     = float(np.dot(pi, Q_true))
    Q_star = float(Q_true.max())
    a_star_i = int(np.argmax(Q_true))
    pi_star = [1, 0] if a_star_i == 0 else [0, 1]
    #----------ここまでオラクル計算(上界には使用しない)
    regret = float(Q_star - EV)

    #fをsoftmaxとみなす
    Q_hat = np.array([phi_hat[0]*R + phi_hat[1]*S,
                      phi_hat[0]*T + phi_hat[1]*P], float)
    beta_f = (1.0 / max(1e-12,T_ref))      #0除算の回避
    s_hat = softmax_beta(beta_f * Q_hat) #\s_{\hat{\mu_t}} のこと

    f_mu  = softmax_beta(beta_f * Q_true)
    f_hat = softmax_beta(beta_f * Q_hat) #(f(\hat{μ}))：相手の予測行動分布からの方策
    s_mu = f_mu
    p_star = float(np.clip(f_mu[a_star_i], 1e-12, 1.0))
    tv_star = 1.0 - p_star

    """
    tv_obs = 1.0 - float(s_hat.min())
    tv_obs_tight = 1.0 - float(s_hat[acc])   #これら：policy Errorの項
    """
    TV = 0.5 * float(np.abs(mu_true_mean - phi_hat).sum())
    ece_t = TV


    #fをsoftmaxにしている:そのミスマッチ項
    M_t  = 0.5 * float(np.abs(pi - s_hat).sum())


    # L (Lipschitz定数) の計算
    L_safe = max(abs(R - S), abs(T - P)) # Q値のグローバルなLipschitz定数 (安全側)
    # L_tight = abs(Q_true[1] - Q_true[0]) # ローカルQ値の差 (タイトだが、安全性が保証されない可能性がある)

    # ★ 修正: L_eff の計算は、常に最も安全なグローバル定数L_safeを使用する
    # use_range_L が True/False にかかわらず、カバレッジを保証するためL_safeを使う
    L_eff = L_safe 
    # L_eff = L_tight if use_range_L else L_safe # (この行を削除またはコメントアウト)
    C  = 0.5 * beta_f

    # LDO_t
    policy_error =  0.5 * float(np.abs(pi_star - f_mu).sum())
    """
    PE = g_ref + (2+C)*ece_t #policyerrorとしてはこっちの方がタイトなはず
    PE_oracle = 1 - float(np.clip(s_mu[a_star_i], 1e-12, 1.0))
    print(f"{policy_error}, {PE}, {PE_oracle}")
    """
    
    #regret_bound_t = L_eff * C * ece_t + L_eff * math.sqrt(ldo_t / 2.0) + L_eff * M_t
    regret_bound_t = L_eff * (policy_error+ C * ece_t  +  M_t)
    rho   = float(regret_bound_t / max(regret, eps_ratio))
    slack = float(regret_bound_t - regret)
    # ★ 修正: 浮動小数点演算の丸め誤差を許容する
    # 非常に小さな負の値（例: -1e-9）までは「カバーされている」とみなす
    tolerance = 1e-9
    covered = bool(regret_bound_t >= regret + tolerance)

    return {
        # LLM 出力
        "offer_idx": a_t, # offer_idx としても a_t を保存
        "b": b_t,         # 相手の実現行動 {0, 1}
        "r_x": out_env["r_x"], # プレイヤーXの実現利得
        "r_y": out_env["r_y"], # プレイヤーYの実現利得
        
        # 真値関連 (診断と評価用)
        "env_id": env.env_id,
        "mu_true_mean": mu_true_mean.tolist(), # LLMが予測すべき「真の平均戦略」
        "mu_true_noise": mu_true_noise.tolist(),# そのラウンドでサンプリングされた「ノイズ」
        "regret_t": regret,        # 真の平均戦略に対する期待リグレット (Q* - EV)
        
        # 指標 (オラクル情報)
        "EV": EV,                  # LLMポリシーの期待値
        "Q_star": Q_star,          # 最適Q値
        "o_star": a_star_i,        # 最適行動 {0, 1}
        "regret_exp": regret,      # regret_t と同じ (互換性のため)
        
        # 3つのエラー項 (診断の核)
        "ece": float(ece_t),             # E_pred: 予測エラー (vs 真の平均)
        "mismatch": float(M_t),          # E_mis: ポリシーミスマッチ (pi vs f(μ_hat))
        "policy_error": policy_error,    # E_pol: ポリシーエラー (pi* vs f(μ_true))
        
        # バウンド関連
        "regret_bound_t": float(regret_bound_t), # 最終的なリグレット上界 L*(Epol + C*Epred + Emis)
        
        # 設定値 (ログ用)
        "temperature": temperature,
        "model": model,
        "L_eff": L_eff,
        "C": float(C),
        
        # 実験用の評価指標
        "rho": rho,                # バウンド / リグレット
        "slack": slack,            # バウンド - リグレット
        "covered": covered,        # (バウンド >= リグレット)
    }
    

def create_intervention_prompts():
    """完全に独立した介入プロンプトを生成"""
    
    # 予測修正プロンプト（方策への影響を最小化）
    prompt_pred_fix = """[PREDICTION MODULE ALERT]
ERROR DETECTED: Opponent model (phi_hat) calibration issue.

TASK: Update ONLY your opponent prediction phi_hat.
- Analyze the opponent_history carefully
- Calculate empirical frequencies: count(action=0) and count(action=1)
- Update phi_hat to match observed frequencies
- Output format: phi_hat = [prob_Cooperate, prob_Defect]

CRITICAL CONSTRAINTS:
- Do NOT change your decision rule for choosing actions.
- Do NOT introduce any new criteria for action selection.
- Do NOT think about payoff maximization in this step.
- Your action probabilities pi may change ONLY as a consequence
  of plugging the updated phi_hat into the SAME decision rule as before.
- ONLY focus on accurately predicting opponent's next action.

After updating phi_hat, apply your existing decision rule to determine pi."""

    # 方策修正プロンプト（予測への影響を最小化）
    prompt_policy_fix = """[POLICY MODULE ALERT]
ERROR DETECTED: Action selection (pi) optimization issue.

TASK: Optimize ONLY your action policy pi.
- Your opponent prediction phi_hat is FIXED and assumed CORRECT
- Review YOUR payoff matrix carefully
- For each of your actions (0:Cooperate, 1:Defect), calculate:
  Expected_Payoff(action) = phi_hat[0] * Payoff[action, 0] + phi_hat[1] * Payoff[action, 1]
- Choose action that MAXIMIZES your expected payoff
- Output format: pi = [prob_Cooperate, prob_Defect]

CRITICAL CONSTRAINTS:
- Do NOT modify phi_hat
- Do NOT reconsider opponent's strategy
- Do NOT update your belief about opponent
- ONLY focus on selecting optimal action given current phi_hat

Treat phi_hat as ground truth for this decision."""

    return prompt_pred_fix, prompt_policy_fix

#ctxが環境の情報で、その情報からLLMが意思決定する
def call_llm_fn(ctx):
    """
    ctx:
      - "actions": [0,1]
      - "model": str
      - "temperature": float
      - "opponent_history": list of int (opponent actions {0,1}) or None
      - "payoff_x": dict (payoff matrix for player X)
      - "intervention_prompt": str or None (介入プロンプト) # ★ 追加
    return: {"phi_hat":[2], "pi":[2], "reasoning": str}
    """
    actions = ctx["actions"]
    opponent_history = ctx.get("opponent_history", [])
    payoff_x = ctx.get("payoff_x")
    
    # ★ 介入プロンプトを取得
    intervention_prompt = ctx.get("intervention_prompt") 

    system_msg = (
        "You are an assistant that returns ONLY valid JSON. "
        "No prose, no markdown. Output keys: phi_hat, pi, reasoning."
    )

    user_msg = {
        "Your_Role": "You are Player X in a 2x2 general-sum game.",
        "Your_Available_Actions": {"0": "Cooperate", "1": "Defect"},
        
        "Player_X_Payoff_Matrix_Description": "YOUR PAYOFF MATRIX (Player X). Rows=Your Actions (0:Cooperate, 1:Defect), Columns=Opponent Actions (0:Cooperate, 1:Defect). Values are your payoffs. [R, S; T, P]", # (内容は変更なし)
        "Payoff_Matrix_Notation_Key": {"R": "Mutual Cooperation", "S": "Your Cooperation, Opponent Defection", "T": "Your Defection, Opponent Cooperation", "P": "Mutual Defection"},      # (内容は変更なし)
        "Player_X_Payoff_Matrix_Values": [
            [payoff_x['R'], payoff_x['S']], 
            [payoff_x['T'], payoff_x['P']]
        ] if payoff_x is not None else "NOT PROVIDED.",

        "Opponent_Action_History_Description": f"OPPONENT ACTION HISTORY (0:Cooperate, 1:Defect). Order: oldest to newest. This history is CRUCIAL for predicting the opponent's next move and determining your best response.", # (内容は変更なし)
        "Opponent_Action_History_Values": opponent_history if opponent_history else "NOT PROVIDED.",

        "Your_Task": "Based on the provided YOUR PAYOFF MATRIX and the OPPONENT ACTION HISTORY, predict the opponent's NEXT ACTION PROBABILITY (phi_hat) AND determine YOUR BEST POLICY (pi). You MUST analyze and use BOTH the payoff matrix and the opponent's history to inform your prediction (phi_hat) and policy (pi). Failure to use BOTH the payoff matrix and the history will result in a poor outcome and an incorrect response.",

        "Output_Format_Requirements": {
            "phi_hat": "List of 2 floats [prob_Cooperate, prob_Defect] for opponent's next move. Each in [0,1], Sum = 1.0.",
            "pi":      "List of 2 floats [prob_Cooperate, prob_Defect] for your action policy. Each >=0, Sum = 1.0.",
            "reasoning": "Single concise sentence explaining rationale using the provided payoff matrix and history."
        },
        "Output_Example": {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "..."},
    }

    # ★ 介入ロジックの追加 ★
    if intervention_prompt:
        # 介入プロンプトを user_msg の最上部に追加
        user_msg["Intervention_Alert"] = intervention_prompt
        # タスク指示を変更し、アラートを最優先で読むよう指示
        user_msg["Your_Task"] = (
            f"IMPORTANT: First, read the 'Intervention_Alert' below. Then, execute your main task. "
            f"{user_msg['Your_Task']}"
        )
    # ★ 介入ロジックここまで ★
    def check_for_sets(obj, path="root"):
        """再帰的に set 型をチェックし、パスを出力する"""
        if isinstance(obj, set):
            raise TypeError(f"!!! CRITICAL DEBUG ERROR: Found set at path: {path}")
        if isinstance(obj, dict):
            for k, v in obj.items():
                check_for_sets(v, path=f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                check_for_sets(v, path=f"{path}[{i}]")

    try:
        check_for_sets(user_msg) # ★ これを挿入して実行
        user_message_content = json.dumps(user_msg, indent=2)
    except TypeError as e:
        # set が見つかった場合、どのキーが問題か特定できます。
        print("FATAL ERROR: JSON Pre-check Failed.")
        raise

    # --- Debug print of the payload being sent ---
    user_message_content = json.dumps(user_msg, indent=2)
    # print(f"--- User message sent to LLM (from call_llm_fn): ---")
    # print(user_message_content)
    # print(f"---------------------------------------------------")


    resp = client.chat.completions.create(
        model=ctx["model"],
        temperature=ctx["temperature"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_message_content}
        ],
        response_format={ "type": "json_object" }
    )
    text = resp.choices[0].message.content.strip()

    # ( ... JSONのパースと検証ロジック ... )
    # ( ... 変更なし ... )
    
    # (JSONパースと検証のコードは省略)
    # (前のコードブロックと同じものをここに追加してください)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            payload = json.loads(m.group(0))
        else:
            raise ValueError(f"LLM response does not contain valid JSON: {text}")
    phi_hat_raw = payload.get("phi_hat")
    pi_raw = payload.get("pi")
    reasoning = payload.get("reasoning", "")
    # ( ... phi_hat, pi の検証と正規化 ... )
    if not isinstance(phi_hat_raw, list) or len(phi_hat_raw) != 2:
           phi_hat_raw = [0.5, 0.5] # フォールバック
    phi_hat = np.asarray(phi_hat_raw, dtype=float)
    phi_hat = np.clip(phi_hat, 1e-12, 1.0)
    phi_hat /= phi_hat.sum()
    if not isinstance(pi_raw, list) or len(pi_raw) != 2:
           pi_raw = [0.5, 0.5] # フォールバック
    pi = np.asarray(pi_raw, dtype=float)
    pi = np.clip(pi, 1e-12, 1.0)
    pi /= pi.sum()
    
    return {
        "phi_hat": phi_hat.tolist(),
        "pi": pi.tolist(),
        "reasoning": reasoning
    }
    
    
def run_sweep_llm(
    call_llm_fn,
    *,
    make_env,
    # ★ 介入モード引数を追加
    intervention_mode: str = "none",  # 'none', 'target', 'non_target'
    intervention_threshold: float = 0.01, # 介入を発動する最小エラー閾値
    tremble=0.0,
    T=200,
    temperatures=(0.2, 0.5, 0.7),
    trials=5,
    seed=42,
    model="gpt-4o",
    L=4.0,
    C=1.0,
    use_range_L=False,
    payoff_x,
    history_length=5
):
    """
    intervention_mode:
      - 'none': 介入なし (ベースライン)
      - 'target': 診断に基づき、支配的なエラーを修正する (グループA)
      - 'non_target': 診断に基づき、*逆*のエラーを修正する (グループB)
    """
    prompt_pred_fix, prompt_policy_fix = create_intervention_prompts()
    logs = []
    temperatures = (temperatures,)
    for ti, temp in enumerate(temperatures):
        for r in range(trials):
            env = make_env(tremble=tremble, seed=seed + 1000*ti + r)
            rng = np.random.default_rng(seed + 2000*ti + r)
            
            opponent_history = []
            
            # ★ 1試行(trial)内のログを保存するリスト
            trial_logs = [] 

            for t in range(1, T+1):
                
                # --- ★★★ 介入ロジック (ここから) ★★★ ---
                intervention_prompt = None
                
                # ラウンドt=1では介入不可 (t > 1 かつ 介入モードが 'none' でない)
                if t > 1 and intervention_mode != "none":
                    # 1. 診断: t-1 のログを取得
                    log_prev = trial_logs[-1] 
                    
                    # 2. 診断: t-1 のエラーの根本原因を特定
                    # (L_eff, C は変動するため、正規化されたエラー成分で比較)
                    pred_blame   = log_prev['C'] * log_prev['ece']
                    policy_blame = log_prev['policy_error'] + log_prev['mismatch']
                    
                    total_error_prev = pred_blame + policy_blame
                    
                    # 3. 介入決定: エラーが閾値より大きい場合のみ
                    if total_error_prev > intervention_threshold:
                        is_pred_dominant = (pred_blame > policy_blame)

                        if intervention_mode == "target":
                            # ターゲット介入 (グループA)
                            if is_pred_dominant:
                                intervention_prompt = prompt_pred_fix
                            else:
                                intervention_prompt = prompt_policy_fix
                        
                        elif intervention_mode == "non_target":
                            # 非ターゲット介入 (グループB)
                            if is_pred_dominant:
                                intervention_prompt = prompt_policy_fix # 逆！
                            else:
                                intervention_prompt = prompt_pred_fix # 逆！
                
                # --- ★★★ 介入ロジック (ここまで) ★★★ ---

                ctx_opponent_history = opponent_history[:]
                
                # 4. 実行: 介入プロンプト(Noneの場合も)を渡す
                log = play_round_llm(
                    env, call_llm_fn, model, temp, rng,
                    L=L, C=C, use_range_L=use_range_L,
                    extra_ctx={
                        "opponent_history": ctx_opponent_history,
                        "intervention_prompt": intervention_prompt # ★ 追加
                    },
                    payoff_x=payoff_x
                )
                
                # ( ... ログ集計 ... )
                cum_regret = trial_logs[-1]["cum_regret"] + log["regret_t"] if t > 1 else log["regret_t"]
                cum_bound  = trial_logs[-1]["cum_regret_bound"] + log["regret_bound_t"] if t > 1 else log["regret_bound_t"]

                log["t"] = t
                log["trial"] = r
                log["cum_regret"] = cum_regret
                log["cum_regret_bound"] = cum_bound
                log["history_length"] = history_length
                
                # ★ ログに介入情報を追加
                log["intervention_mode"] = intervention_mode
                log["intervention_applied"] = intervention_prompt is not None

                logs.append(log)
                trial_logs.append(log) # 試行内ログにも追加

                # 履歴の更新
                opponent_action_this_round = log.get("b")
                if opponent_action_this_round is not None:
                    opponent_history.append(opponent_action_this_round)

    # ( ... 集計 ... )
    summary = {}
    # (集計ロジックは変更なし)
    
    return {"summary": summary, "per_round": logs}
#このper_roundの情報をグラフにしている



def make_pd_env(seed, tremble):
    # make_pd_env ではベータ分布のパラメータと平均確率を設定
    # 相手プレイヤーYが行動1 (裏切り) を選択する確率の平均 p_y1_mean を設定
    # ここでは例として0.5 (ベータ分布の中心) のままですが、必要に応じて変更可能
    p_y1_mean_for_env = 0.5

    # ここはあなたの GeneralSum 環境に合わせて置き換えてください
    # 例: GeneralSumGameEnv(payoff_x, payoff_y, p_opponent, tremble, seed, env_id)
    return GeneralSumGameENV(
        payoff_x=payoff_dilemma_x,
        payoff_y=payoff_dilemma_y,
        alpha_y=2.0,       # ベータ分布のパラメータ
        beta_y=2.0,        # ベータ分布のパラメータ
        p_y1_mean=p_y1_mean_for_env, # ベータ分布の平均確率を設定
        tremble=tremble,
        seed=seed,
    )


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("実験 1/3 (ベースライン) を実行中...")
out_none = run_sweep_llm(
    call_llm_fn, make_env=make_pd_env, T=8, temperatures=1.0,trials=1, model="gpt-5-2025-08-07", 
    payoff_x=payoff_dilemma_x, intervention_mode="none" # ★
)

logs = out_none["per_round"]
df_results = pd.DataFrame(logs)

# 3. CSVファイルとして結果を保存
csv_filename = "results_none.csv"
df_results.to_csv(csv_filename, index=False)
print(f"結果を {csv_filename} に保存しました。")

print("実験 2/3 (ターゲット介入) を実行中...")
out_target = run_sweep_llm(
    call_llm_fn, make_env=make_pd_env, T=8,temperatures=1.0,trials=2, model="gpt-5-2025-08-07", 
    payoff_x=payoff_dilemma_x, intervention_mode="target" # ★
)

logs = out_target["per_round"]
df_results = pd.DataFrame(logs)

# 3. CSVファイルとして結果を保存
csv_filename = "results_target.csv"
df_results.to_csv(csv_filename, index=False)
print(f"結果を {csv_filename} に保存しました。")

print("実験 3/3 (非ターゲット介入) を実行中...")
out_non_target = run_sweep_llm(
    call_llm_fn, make_env=make_pd_env, T=8,temperatures=1.0, trials=2, model="gpt-5-2025-08-07", 
    payoff_x=payoff_dilemma_x, intervention_mode="non_target" # ★
)

logs = out_non_target["per_round"]
df_results = pd.DataFrame(logs)

# 3. CSVファイルとして結果を保存
csv_filename = "results_non_target.csv"
df_results.to_csv(csv_filename, index=False)
print(f"結果を {csv_filename} に保存しました。")


def analyze_intervention_effects(df_none, df_target, df_non_target):
    """
    3つの介入モードを比較
    """
    print("=== 介入効果の比較 ===\n")
    
    # 累積リグレットの最終値
    none_final = df_none.groupby('trial')['cum_regret'].last().mean()
    target_final = df_target.groupby('trial')['cum_regret'].last().mean()
    non_target_final = df_non_target.groupby('trial')['cum_regret'].last().mean()
    
    print("累積リグレット（最終）:")
    print(f"  None (ベースライン):     {none_final:.4f}")
    print(f"  Target (正しい介入):     {target_final:.4f}")
    print(f"  Non-target (逆介入):     {non_target_final:.4f}")
    print(f"  改善率 (Target vs None): {(none_final - target_final)/none_final*100:.2f}%")
    print(f"  悪化率 (Non-target vs None): {(non_target_final - none_final)/none_final*100:.2f}%")
    
    # 平均リグレット
    print("\n平均リグレット（全ラウンド）:")
    print(f"  None:       {df_none['regret_t'].mean():.4f}")
    print(f"  Target:     {df_target['regret_t'].mean():.4f}")
    print(f"  Non-target: {df_non_target['regret_t'].mean():.4f}")
    
    # 介入回数
    print("\n介入回数:")
    print(f"  Target:     {df_target['intervention_applied'].sum()} 回")
    print(f"  Non-target: {df_non_target['intervention_applied'].sum()} 回")
    
    # エラー成分の平均
    print("\n平均エラー成分:")
    print(f"  ECE (予測誤差):")
    print(f"    None:       {df_none['ece'].mean():.4f}")
    print(f"    Target:     {df_target['ece'].mean():.4f}")
    print(f"    Non-target: {df_non_target['ece'].mean():.4f}")
    
    print(f"  Mismatch (方策ミスマッチ):")
    print(f"    None:       {df_none['mismatch'].mean():.4f}")
    print(f"    Target:     {df_target['mismatch'].mean():.4f}")
    print(f"    Non-target: {df_non_target['mismatch'].mean():.4f}")
    
    print(f"  Policy Error (方策誤差):")
    print(f"    None:       {df_none['policy_error'].mean():.4f}")
    print(f"    Target:     {df_target['policy_error'].mean():.4f}")
    print(f"    Non-target: {df_non_target['policy_error'].mean():.4f}")

# 実験後に検証
analyze_intervention_effects(out_none, out_target, out_non_target)


"""
def plot_results_from_df(df, model="gpt-4o", p_y1_mean=0.5):
    
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
    plt.savefig("regret_plot.png")
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
    plt.savefig("error_components_plot.png")
    plt.close()

# 4. グラフ描画関数の呼び出し
# (この下のセクションで定義する関数)
plot_results_from_df(df_results, model="gpt-4o", p_y1_mean=0.5) # env.p_y1_mean を渡す
print(f"グラフを 'regret_plot.png' と 'error_components_plot.png' として保存しました。")
"""

