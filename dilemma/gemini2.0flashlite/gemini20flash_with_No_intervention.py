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
    



#ctxが環境の情報で、その情報からLLMが意思決定する
def call_llm_fn(ctx):
    """
    ctx:
      - "actions": [0,1]
      - "model": str
      - "temperature": float
      - "opponent_history": list of int (opponent actions {0,1}) or None
      - "payoff_x": dict (payoff matrix for player X) # Added
    return: {"phi_hat":[2], "pi":[2], "reasoning": str}
    """
    actions = ctx["actions"]
    opponent_history = ctx.get("opponent_history", [])
    payoff_x = ctx.get("payoff_x") # Get payoff matrix for player X

    system_msg = (
        "You are an assistant that returns ONLY valid JSON. "
        "No prose, no markdown. Output keys: phi_hat, pi, reasoning."
    )

    # ユーザーメッセージを、LLMが最も確実に認識できるよう、トップレベルのフラットな構造に変更
    user_msg = {
        "Your_Role": "You are Player X in a 2x2 general-sum game.",
        "Your_Available_Actions": {"0": "Cooperate", "1": "Defect"},

        # 利得表をトップレベルに配置
        "Player_X_Payoff_Matrix_Description": "YOUR PAYOFF MATRIX (Player X). Rows=Your Actions (0:Cooperate, 1:Defect), Columns=Opponent Actions (0:Cooperate, 1:Defect). Values are your payoffs. [R, S; T, P]",
        "Payoff_Matrix_Notation_Key": {"R": "Mutual Cooperation", "S": "Your Cooperation, Opponent Defection", "T": "Your Defection, Opponent Cooperation", "P": "Mutual Defection"},
        "Player_X_Payoff_Matrix_Values": [
            [payoff_x['R'], payoff_x['S']], # Your action 0 (Cooperate) vs Opponent actions (0, 1) -> (R, S)
            [payoff_x['T'], payoff_x['P']]  # Your action 1 (Defect) vs Opponent actions (0, 1) -> (T, P)
        ] if payoff_x is not None else "NOT PROVIDED.",

        # 履歴をトップレベルに配置
        "Opponent_Action_History_Description": f"OPPONENT ACTION HISTORY (0:Cooperate, 1:Defect). Order: oldest to newest. This history is CRUCIAL for predicting the opponent's next move and determining your best response.", # 履歴の重要性をさらに強調
        "Opponent_Action_History_Values": opponent_history if opponent_history else "NOT PROVIDED.",

        # タスク指示をトップレベルに配置し、必須であることを強調
        "Your_Task": "Based on the provided YOUR PAYOFF MATRIX and the OPPONENT ACTION HISTORY, predict the opponent's NEXT ACTION PROBABILITY (phi_hat) AND determine YOUR BEST POLICY (pi). You MUST analyze and use BOTH the payoff matrix and the opponent's history to inform your prediction (phi_hat) and policy (pi). Failure to use BOTH the payoff matrix and the history will result in a poor outcome and an incorrect response.", # 履歴と利得表の使用を非常に強く、両方必須であることを指示

        # 出力要件と例もトップレベルに
        "Output_Format_Requirements": {
            "phi_hat": "List of 2 floats [prob_Cooperate, prob_Defect] for opponent's next move. Each in [0,1], Sum = 1.0.",
            "pi":      "List of 2 floats [prob_Cooperate, prob_Defect] for your action policy. Each >=0, Sum = 1.0.",
            "reasoning": "Single concise sentence explaining rationale using the provided payoff matrix and history."
        },
        "Output_Example": {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "..."},
    }


    # --- Debug print of the payload being sent ---
    # Debug prints are essential to verify the input structure and raw response
    user_message_content = json.dumps(user_msg, indent=2)
    print(f"--- User message sent to LLM (from call_llm_fn): ---")
    print(user_message_content)
    print(f"---------------------------------------------------")


    resp = client.chat.completions.create(
        model=ctx["model"],
        temperature=ctx["temperature"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_message_content} # Send the structured payload
        ],
        response_format={ "type": "json_object" } # JSON形式を強制
    )
    # Accessing message content based on typical OpenAI API structure
    text = resp.choices[0].message.content.strip()

    # --- Debug print of the raw LLM response ---
    print(f"--- LLM Raw Response (from call_llm_fn) ---")
    print(text)
    print(f"-------------------------------------------")


    # Attempt to find and parse JSON within the text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Fallback JSON parsing - keep this error handling
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            payload = json.loads(m.group(0))
        else:
            raise ValueError(f"LLM response does not contain valid JSON: {text}")


    # phi_hat と pi の取得と検証
    phi_hat_raw = payload.get("phi_hat")
    pi_raw = payload.get("pi")
    reasoning = payload.get("reasoning", "")

    # --- Debug print of parsed LLM output ---
    # Keep debug prints for visibility
    print(f"--- Parsed LLM output in call_llm_fn ---")
    print(f"  phi_hat: {phi_hat_raw}")
    print(f"  pi: {pi_raw}")
    print(f"  Reasoning: '{reasoning}'")
    print("------------------------------------------")


    # Validate and normalize phi_hat
    if not isinstance(phi_hat_raw, list) or len(phi_hat_raw) != 2:
         print(f"Validation Error: Invalid phi_hat format or length: {phi_hat_raw}")
         return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM phi_hat output"}
    try:
        phi_hat = np.asarray(phi_hat_raw, dtype=float)
    except ValueError:
         print(f"Validation Error: Invalid phi_hat values (not floats): {phi_hat_raw}")
         return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM phi_hat values"}

    phi_hat = np.clip(phi_hat, 1e-12, 1.0)
    s_phi = float(phi_hat.sum())
    if not np.isfinite(s_phi) or s_phi <= 1e-12:
        print(f"Validation Error: phi_hat sum invalid: {s_phi}")
        return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM phi_hat sum"}
    phi_hat /= s_phi


    # Validate and normalize pi
    if not isinstance(pi_raw, list) or len(pi_raw) != 2:
        print(f"Validation Error: Invalid pi format or length: {pi_raw}")
        return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM pi format"}
    try:
        pi = np.asarray(pi_raw, dtype=float)
    except ValueError:
         print(f"Validation Error: Invalid pi values (not floats): {pi_raw}")
         return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM pi values"}

    pi = np.clip(pi, 1e-12, 1.0)
    s = float(pi.sum())
    if not np.isfinite(s) or s <= 1e-12:
        print(f"Validation Error: pi sum invalid: {s}")
        return {"phi_hat": [0.5, 0.5], "pi": [0.5, 0.5], "reasoning": "Invalid LLM pi sum"}
    pi /= s

    # Return the validated and normalized output
    return {
        "phi_hat": phi_hat.tolist(),
        "pi": pi.tolist(),
        "reasoning": reasoning
    }
    
    
    
    
    
    
def run_sweep_llm(
    call_llm_fn,
    *,
    make_env,
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
    history_length=5 # 追加: 考慮する過去のラウンド数
):
    logs = []
    for ti, temp in enumerate(temperatures):
        for r in range(trials):
            env = make_env(tremble=tremble, seed=seed + 1000*ti + r)
            rng = np.random.default_rng(seed + 2000*ti + r)
            cum_regret = 0.0
            cum_bound  = 0.0
            # 過去の相手の行動履歴を保存するリスト
            opponent_history = []
            for t in range(1, T+1):
                # LLMに渡す履歴を準備 (最新の history_length ラウンドの相手行動)
                #ctx_opponent_history = opponent_history[-history_length:] if history_length > 0 else []
                ctx_opponent_history = opponent_history[:]

                log = play_round_llm(env, call_llm_fn, model, temp, rng,
                                     L=L, C=C, use_range_L=use_range_L,
                                     extra_ctx={"opponent_history": ctx_opponent_history}, # 相手の行動履歴を渡す
                                     payoff_x = payoff_x)
                cum_regret += log["regret_t"]
                cum_bound  += log["regret_bound_t"]
                log["t"] = t
                log["trial"] = r
                log["cum_regret"] = cum_regret
                log["cum_regret_bound"] = cum_bound
                log["history_length"] = history_length # ログに履歴長さを記録
                logs.append(log)

                # 現在のラウンドの相手の行動を履歴に追加
                # log.get("b", log.get("accepted")) は、環境からの戻り値で相手の行動が "b" または "accepted" キーで格納されていることを想定しています。
                # play_round_llm 関数を見ると、環境のstep関数は {"a": action_x, "b": action_y, "r_x": r_x, "r_y": r_y} を返しています。
                # したがって、相手の行動は "b" キーで取得するのが正しいです。
                opponent_action_this_round = log.get("b")
                if opponent_action_this_round is not None:
                    opponent_history.append(opponent_action_this_round)
                # else: # 環境のstep関数が "b" を返さない場合はエラーまたは警告を出すなど検討
                    # print(f"Warning: Opponent action 'b' not found in log for round {t}, trial {r}. Log: {log}")


    # 集計

    summary = {}
    for temp in temperatures:
        # Filter by history_length as well if needed, but for now assume one run_sweep per history_length
        rows_all   = [x for x in logs if x["temperature"] == temp]
        rows_final = [x for x in rows_all if x["t"] == T]
        final_cum      = np.array([x["cum_regret"] for x in rows_final], float)
        final_cum_bnd  = np.array([x["cum_regret_bound"] for x in rows_final], float)
        per_regret     = np.array([x["regret_exp"] for x in rows_all], float)
        per_bound      = np.array([x["regret_bound_t"] for x in rows_all], float)
        summary[temp] = {
            "final_cum_regret_mean": float(final_cum.mean()),
            "final_cum_regret_std":  float(final_cum.std(ddof=1)) if len(final_cum)>1 else 0.0,
            "final_cum_bound_mean":  float(final_cum_bnd.mean()),
            "final_cum_bound_std":   float(final_cum_bnd.std(ddof=1)) if len(final_cum_bnd)>1 else 0.0,
            "mean_regret_per_round": float(per_regret.mean()),
            "mean_bound_per_round":  float(per_bound.mean()),
        }
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


client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
out = run_sweep_llm(
    call_llm_fn,
    make_env = make_pd_env,
    tremble=0.02,
    T=5,
    temperatures=(0.8,), # Changed temperature to 1.0
    trials=20, #20
    seed=42,
    model="gpt-4o",  #"gpt-5-mini", #"gpt-5-2025-08-07"
    L=3.0,        # 固定 L を使うならここで指定
    C=1.0,
    use_range_L=True,  # True なら L := range(Q_true) を毎ラウンド再計算
    payoff_x=payoff_dilemma_x,
    history_length=1 # 履歴の長さを5に設定
)


logs = out["per_round"]
df_results = pd.DataFrame(logs)

# 3. CSVファイルとして結果を保存
csv_filename = "llm_game_results_dillema_GPT4o.csv"
df_results.to_csv(csv_filename, index=False)
print(f"結果を {csv_filename} に保存しました。")



def plot_results_from_df(df, model="gpt-4o", p_y1_mean=0.5):
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