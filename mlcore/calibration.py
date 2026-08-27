from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any

from .utils import sigmoid, logit

@dataclass
class PlattCalibrator:
    """Platt scaling: p = sigmoid(a * x + b), where x is a score (logit of raw p)."""
    a: float = 1.0
    b: float = 0.0

    def predict_proba(self, p_raw: float) -> float:
        x = logit(p_raw)
        return sigmoid(self.a * x + self.b)

    def to_dict(self) -> Dict[str, Any]:
        return {"a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlattCalibrator":
        return cls(a=float(d.get("a", 1.0)), b=float(d.get("b", 0.0)))


def fit_platt_from_raw_probs(
    y_true: Iterable[int],
    p_raw: Iterable[float],
    *,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> PlattCalibrator:
    """Fit Platt scaling parameters a,b using Newton-Raphson on logistic loss.

    We use x = logit(p_raw) as input feature to keep it stable and meaningful.
    Adds small L2 regularization for stability.
    """
    xs = []
    ys = []
    for y, p in zip(y_true, p_raw):
        ys.append(1 if int(y) == 1 else 0)
        xs.append(logit(float(p)))

    n = len(xs)
    if n == 0:
        return PlattCalibrator()
    # If labels are all same, calibration is undefined; return identity.
    if sum(ys) == 0 or sum(ys) == n:
        return PlattCalibrator()

    a = 1.0
    b = 0.0

    for _ in range(max_iter):
        # gradients and Hessian
        ga = 0.0
        gb = 0.0
        haa = l2
        hab = 0.0
        hbb = l2
        for x, y in zip(xs, ys):
            z = a * x + b
            p = sigmoid(z)
            w = p * (1.0 - p)
            ga += (p - y) * x
            gb += (p - y)
            haa += w * x * x
            hab += w * x
            hbb += w

        # Newton step solve 2x2
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = ( hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det

        a_new = a - da
        b_new = b - db

        if abs(a_new - a) < tol and abs(b_new - b) < tol:
            a, b = a_new, b_new
            break

        a, b = a_new, b_new

    return PlattCalibrator(a=a, b=b)


# =====================================================================
# CALIBRADORES ANINHADOS (26/08/2026)
# =====================================================================
# Descoberta que reordenou o desenho: o PlattCalibrator acima e, palavra por
# palavra, a beta calibration com b = a (Kull, Silva Filho & Flach, AISTATS
# 2017, Proposicao 1). Rodavamos a versao de 2 parametros de uma familia de
# 3. Os tres candidatos sao MODELOS ANINHADOS:
#
#   temperature  p = sigmoid(a * logit(s))                 1 parametro
#   Platt        p = sigmoid(a * logit(s) + c)             2  (= beta[a=b])
#   beta         p = sigmoid(a*ln(s) - b*ln(1-s) + c)      3
#
# Por isso um unico ajustador serve aos tres: muda so o mapa de features.
#
# POR QUE A BETA E A APOSTA SEGURA: a funcao IDENTIDADE pertence a familia
# beta (a=b=1, c=0) e NAO pertence a familia logistica. Se o modelo ja sai
# calibrado, o Platt nao consegue "nao fazer nada" — ele necessariamente
# descalibra. A beta contem a resposta do Platt como caso particular.
#
# COMO DECIDIR ENTRE ELES: pela LOGLOSS no teste cego, nunca pelo ECE. Os
# tres sao monotonicos por construcao (a, b >= 0), e recalibracao monotonica
# preserva a resolucao — entao a diferenca de logloss entre eles E a
# diferenca de calibracao, medida por regra de pontuacao propria, sem binning
# e sem vies de estimacao. O ECE fica como leitura, nao como juiz.

import math  # noqa: E402

from .utils import clip  # noqa: E402


def _solve(a: list, b: list):
    """Gauss com pivoteamento parcial. None quando a matriz e singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-14:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _fit_logistic(X: list, y: list, *, l2: float = 1e-3,
                  max_iter: int = 200, tol: float = 1e-8) -> list:
    """Newton/IRLS para regressao logistica com k features (sem intercepto
    implicito: quem quiser um bota uma coluna de 1s). Mesma receita do Platt
    acima, generalizada — e o alvo y pode ser fracionario, para aceitar os
    alvos regularizados do Platt original."""
    if not X:
        return []
    k = len(X[0])
    w = [0.0] * k
    for _ in range(max_iter):
        grad = [l2 * w[j] for j in range(k)]
        hess = [[l2 if i == j else 0.0 for j in range(k)] for i in range(k)]
        for xi, yi in zip(X, y):
            z = sum(w[j] * xi[j] for j in range(k))
            p = sigmoid(z)
            r = p - yi
            wgt = p * (1.0 - p)
            for i in range(k):
                grad[i] += r * xi[i]
                for j in range(i, k):
                    hess[i][j] += wgt * xi[i] * xi[j]
        for i in range(k):
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = _solve(hess, grad)
        if step is None:
            break
        w_new = [w[i] - step[i] for i in range(k)]
        if max(abs(w_new[i] - w[i]) for i in range(k)) < tol:
            return w_new
        w = w_new
    return w


def platt_targets(y_true) -> list:
    """Alvos REGULARIZADOS do artigo original do Platt (1999):

        y+ = (N+ + 1)/(N+ + 2)      y- = 1/(N- + 2)

    em vez de 1 e 0 crus. Sai de Bayes com prior uniforme sobre os rotulos, e
    existe justamente para nao superajustar em conjunto pequeno — que e o
    nosso caso (~1.000 pontos por minuto no ao vivo do Dota). O
    fit_platt_from_raw_probs historico usa 0/1 cru; a correcao entra por
    GATE, nao caladamente, porque muda os numeros do titular."""
    ys = [1 if int(v) == 1 else 0 for v in y_true]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    return [t_pos if v == 1 else t_neg for v in ys]


@dataclass
class TemperatureCalibrator:
    """Temperature scaling (Guo et al., 2017): p = sigmoid(logit(s) / T).

    Guardamos a = 1/T para o mapa ser o do Platt sem intercepto. NAO altera a
    acuracia (e transformacao monotonica), entao acuracia nunca pode ser
    criterio de escolha entre calibradores."""
    a: float = 1.0

    @property
    def temperature(self) -> float:
        return float("inf") if abs(self.a) < 1e-12 else 1.0 / self.a

    def predict_proba(self, p_raw: float) -> float:
        return sigmoid(self.a * logit(p_raw))

    def to_dict(self) -> Dict[str, Any]:
        return {"a": float(self.a)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemperatureCalibrator":
        return cls(a=float(d.get("a", 1.0)))


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull et al., 2017), Algoritmo 2 do artigo:

        s'  = ln(s)          s'' = -ln(1 - s)
        (a, b, c) por regressao logistica BIVARIADA de y sobre s' e s''
        p   = 1 / (1 + 1/(e^c * s^a / (1-s)^b))

    Com a = b vira exatamente o Platt (Proposicao 1). Com a = b = 1 e c = 0
    vira a identidade — que a familia logistica nao consegue expressar."""
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0

    def predict_proba(self, p_raw: float) -> float:
        s = clip(float(p_raw), 1e-12, 1.0 - 1e-12)
        return sigmoid(self.a * math.log(s) - self.b * math.log(1.0 - s) + self.c)

    def to_dict(self) -> Dict[str, Any]:
        return {"a": float(self.a), "b": float(self.b), "c": float(self.c)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BetaCalibrator":
        return cls(a=float(d.get("a", 1.0)), b=float(d.get("b", 1.0)),
                   c=float(d.get("c", 0.0)))


def fit_temperature_from_raw_probs(y_true, p_raw, *, l2: float = 1e-3,
                                   alvos_platt: bool = True) -> TemperatureCalibrator:
    ys = [1 if int(v) == 1 else 0 for v in y_true]
    if not ys or sum(ys) in (0, len(ys)):
        return TemperatureCalibrator()
    alvo = platt_targets(ys) if alvos_platt else [float(v) for v in ys]
    w = _fit_logistic([[logit(float(p))] for p in p_raw], alvo, l2=l2)
    a = w[0] if w else 1.0
    # a < 0 inverteria a ordenacao: deixaria de ser recalibracao e viraria
    # modelo novo (pior). Cai na identidade, que e sempre valida.
    return TemperatureCalibrator(a=a if a > 0 else 1.0)


def fit_beta_from_raw_probs(y_true, p_raw, *, l2: float = 1e-3,
                            alvos_platt: bool = True) -> BetaCalibrator:
    ys = [1 if int(v) == 1 else 0 for v in y_true]
    if not ys or sum(ys) in (0, len(ys)):
        return BetaCalibrator()
    alvo = platt_targets(ys) if alvos_platt else [float(v) for v in ys]
    ss = [clip(float(p), 1e-12, 1.0 - 1e-12) for p in p_raw]
    w = _fit_logistic([[math.log(s), -math.log(1.0 - s), 1.0] for s in ss], alvo, l2=l2)
    if len(w) != 3:
        return BetaCalibrator()
    a, b, c = w
    # Cuidado documentado no artigo: a < 0 ou b < 0 produz mapa NAO
    # monotonico. A receita deles e zerar o negativo e reajustar com o resto.
    if a >= 0 and b >= 0:
        return BetaCalibrator(a=a, b=b, c=c)
    if a < 0 and b < 0:
        return BetaCalibrator()
    if a < 0:
        w2 = _fit_logistic([[-math.log(1.0 - s), 1.0] for s in ss], alvo, l2=l2)
        return BetaCalibrator(a=0.0, b=w2[0], c=w2[1]) if len(w2) == 2 else BetaCalibrator()
    w2 = _fit_logistic([[math.log(s), 1.0] for s in ss], alvo, l2=l2)
    return BetaCalibrator(a=w2[0], b=0.0, c=w2[1]) if len(w2) == 2 else BetaCalibrator()


def fit_platt_corrigido(y_true, p_raw, *, l2: float = 1e-3) -> PlattCalibrator:
    """Platt com os alvos regularizados do artigo original — o que o
    fit_platt_from_raw_probs deveria ter feito desde sempre."""
    ys = [1 if int(v) == 1 else 0 for v in y_true]
    if not ys or sum(ys) in (0, len(ys)):
        return PlattCalibrator()
    w = _fit_logistic([[logit(float(p)), 1.0] for p in p_raw], platt_targets(ys), l2=l2)
    return PlattCalibrator(a=w[0], b=w[1]) if len(w) == 2 else PlattCalibrator()
