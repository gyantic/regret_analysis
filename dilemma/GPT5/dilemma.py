from dataclasses import dataclass
import numpy as np

payoff_dilemma_x = {'T': 5, 'R': 3, 'P': 1, 'S': 0}
payoff_dilemma_y = {'T': 5, 'R': 3, 'P': 1, 'S': 0}



@dataclass
class GeneralSumGameENV:
    payoff_x: dict         # {'T','R','P','S'} for player X
    payoff_y: dict         # {'T','R','P','S'} for player Y
    alpha_y: float = 2.0   # Beta distribution alpha parameter for p_y1
    beta_y: float = 2.0    # Beta distribution beta parameter for p_y1
    p_y1_mean: float = 0.5 # Mean of the Beta distribution (used for true mu)
    tremble: float = 0.0
    seed: int = 0
    env_id: str = "general"

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        # p_y1_mean が指定されている場合は、それに合うように alpha, beta を調整
        # mean = alpha / (alpha + beta) なので、beta = alpha * (1 - mean) / mean
        if self.p_y1_mean != 0.5:
             # 簡単のため alpha + beta = 4 を維持しつつ平均を調整
             sum_ab = self.alpha_y + self.beta_y
             self.alpha_y = self.p_y1_mean * sum_ab
             self.beta_y = (1.0 - self.p_y1_mean) * sum_ab


    @staticmethod
    def _Rx(payoff_x, ax: int, ay: int) -> int:
        # X の利得: (0,0)->R, (0,1)->S, (1,0)->T, (1,1)->P
        if ax == 0 and ay == 0: return payoff_x['R']
        if ax == 0 and ay == 1: return payoff_x['S']
        if ax == 1 and ay == 0: return payoff_x['T']
        return payoff_x['P']  # (1,1)

    @staticmethod
    def _Ry(payoff_y, ax: int, ay: int) -> int:
        # Y の利得: (0,0)->R, (0,1)->T, (1,0)->S, (1,1)->P
        if ax == 0 and ay == 0: return payoff_y['R']
        if ax == 0 and ay == 1: return payoff_y['T']
        if ax == 1 and ay == 0: return payoff_y['S']
        return payoff_y['P']  # (1,1)

    def step(self, action_x: int):
        # 相手行動の生成（z を対称に適用）
        # 毎ラウンド、ベータ分布から新しい p_y1 をサンプリング
        p_y1 = self.rng.beta(self.alpha_y, self.beta_y)

        # tremble も考慮
        p1 = self.tremble + (1.0 - 2.0 * self.tremble) * float(p_y1)
        action_y = int(self.rng.random() < p1)

        r_x = self._Rx(self.payoff_x, action_x, action_y)
        r_y = self._Ry(self.payoff_y, action_x, action_y)

        # 相手の真の行動確率 mu_true を戻り値に含める
        mu_true = np.array([1.0 - p_y1, p_y1], dtype=float)

        return {"a": action_x, "b": action_y, "r_x": r_x, "r_y": r_y, "mu_true": mu_true.tolist()} # numpy array を list に変換してシリアライズ可能にする


def q_vector(mu, payoff_x):
  mu = np.asarray(mu)
  assert mu.shape == (2,)
  R,S,T,P = payoff_x['R'], payoff_x['S'], payoff_x['T'], payoff_x['P']
  Q0 = mu[0] * R + mu[1] * S
  Q1 = mu[0] * T + mu[1] * P
  return np.array([Q0, Q1], float)

def gto_action(mu, payoff_x):
  Q = q_vector(mu, payoff_x)
  idx = int(np.argmax(Q))
  return idx, float(Q[idx])

def expected_value(pi, mu, payoff_x):
    pi = np.asarray(pi, float); assert pi.shape == (2,) and np.isclose(pi.sum(), 1.0)
    return float(np.dot(pi, q_vector(mu, payoff_x)))


def compute_Q(mu, payoff_x, use_tremble=True, eps=0.0):
    mu = np.asarray(mu, float); assert mu.shape == (2,)
    if use_tremble and eps > 0.0:
        p1 = float(np.clip(eps + (1.0 - 2.0*eps)*mu[1], 0.0, 1.0))
        mu_eval = np.array([1.0 - p1, p1], float)
    else:
        mu_eval =r(mu_eval, payoff_x)
    a_star_i = int(np.argmax(Q))
    return Q, mu_eval, a_star_i