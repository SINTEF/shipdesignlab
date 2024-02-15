import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import pandas as pd

pd.options.plotting.backend = "plotly"


def a_value(Lpp, B, T, Vs, Vc, Fn, CB, alpha, kyy):
    """
    calculate a_value (Liu&Papanikolaou, 2020)

    Parameters
    ----------
    Lpp: float
        length between perpendiculars in meters
    B: float
        breadth in meters
    T: float
        mean draught in meters
    Vs: float
        ship speed in m/s
    Vc: float
        phase velocity in m/s
    Fn: float
        froude number
    CB: float
        block coefficient
    alpha: float
        wave heading (Head waves: 180, Following waves:0)
    kyy: float
        radius of gyration of pitch

    Returns
    -------
    a1_value: float
    a2_value: float

    """
    if alpha >= np.pi / 2:
        a1_value = (
            60.3
            * CB**1.34
            * (4 * kyy) ** 2
            * (0.87 / CB) ** (-(1 + Fn) * np.cos(alpha))
            * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(alpha)) / 3))
        )
        if Fn < 0.12:
            a2_value = 0.0072 + 0.1676 * Fn

        else:
            a2_value = Fn**1.5 * np.exp(-3.5 * Fn)

    elif alpha == 0:
        if Vs == 0:
            a1_value = (
                60.3
                * CB**1.34
                * (4 * kyy) ** 2
                * (0.87 / CB) ** (-1 * np.cos(alpha))
                * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(alpha)) / 3))
            )
            a2_value = 0.0072

        elif Vs == (Vc / 4):
            a1_value = 0
            if Fn < 0.12:
                a2_value = 0.0072 + 0.1676 * Fn

            else:
                a2_value = Fn**1.5 * np.exp(-3.5 * Fn)

        elif Vs == (Vc / 2):
            a1_value = (
                60.3
                * CB**1.34
                * (4 * kyy) ** 2
                * (0.87 / CB) ** (-1 * np.cos(np.pi))
                * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(np.pi)) / 3))
            )
            a2_value = 0.0072

        elif (Vs > 0) and (Vs < (Vc / 4)):
            a1_1 = (
                60.3
                * CB**1.34
                * (4 * kyy) ** 2
                * (0.87 / CB) ** (-1 * np.cos(alpha))
                * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(alpha)) / 3))
            )
            a1_2 = 0
            a1_matrix = [a1_1, a1_2]

            a2_1 = 0.0072
            if Fn < 0.12:
                a2_2 = 0.0072 + 0.1676 * Fn

            else:
                a2_2 = Fn**1.5 * np.exp(-3.5 * Fn)

            a2_matrix = [a2_1, a2_2]
            Vs_matrix = [0, Vc / 4]

            a1_f = interp1d(Vs_matrix, a1_matrix)
            a1_value = a1_f(Vs)
            a2_f = interp1d(Vs_matrix, a2_matrix)
            a2_value = a2_f(Vs)

        elif (Vs > (Vc / 4)) and (Vs < (Vc / 2)):
            a1_1 = 0
            a1_2 = (
                60.3
                * CB**1.34
                * (4 * kyy) ** 2
                * (0.87 / CB) ** (-1 * np.cos(np.pi))
                * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(np.pi)) / 3))
            )
            a1_matrix = [a1_1, a1_2]

            if Fn < 0.12:
                a2_1 = 0.0072 + 0.1676 * Fn

            else:
                a2_1 = Fn**1.5 * np.exp(-3.5 * Fn)

            a2_2 = 0.0072

            a2_matrix = [a2_1, a2_2]
            Vs_matrix = [Vc / 4, Vc / 2]

            a1_f = interp1d(Vs_matrix, a1_matrix)
            a1_value = a1_f(Vs)
            a2_f = interp1d(Vs_matrix, a2_matrix)
            a2_value = a2_f(Vs)

        else:
            Fn_new = (Vs - Vc / 2) / np.sqrt(Lpp * 9.81)
            a1_value = (
                60.3
                * CB**1.34
                * (4 * kyy) ** 2
                * (0.87 / CB) ** (-(1 + Fn_new) * np.cos(np.pi))
                * (np.log(B / T) ** (-1) * ((1 - 2 * np.cos(np.pi)) / 3))
            )
            if Fn_new < 0.12:
                a2_value = 0.0072 + 0.1676 * Fn_new
            else:
                a2_value = Fn_new**1.5 * np.exp(-3.5 * Fn_new)

    return a1_value, a2_value


def Radiation_head_waves(Lpp, T, Tf, Ta, B, CB, Fn, lambda_, alpha, Vs, Vc, kyy):
    """
    calculate the component of added resistance due to motion (radiation)
    (Liu&Papanikolaou, 2020)

    Parameters
    ----------
    Lpp: float
        length between perpendiculars in meters
    T: float
        mean draught in meters
    Tf: float
        forward draught in meters
    Ta: float
        aft draught in meters
    B: float
        breadth in meters
    CB: float
        block coefficient
    Fn: float
        froude number
    lambda_: float
        wavelength (lambda)
    alpha: float
        wave heading (Head waves: 180, Following waves:0)
    Vs: float
        ship speed in m/s
    Vc: float
        phase velocity in m/s
    kyy: float
        radius of gyration of pitch

    Returns
    -------
    R_AWML: float
        added resistance due to motion (radiation), longer wave length range dominant
    """
    if alpha >= np.pi / 2:
        a1, a2 = a_value(Lpp, B, T, Vs, Vc, Fn, CB, alpha, kyy)
    elif alpha == 0:
        a1, a2 = a_value(Lpp, B, T, Vs, Vc, Fn, CB, alpha, kyy)
    else:
        a1_head, a2_head = a_value(Lpp, B, T, Vs, Vc, Fn, CB, np.pi / 2, kyy)
        a1_following, a2_following = a_value(Lpp, B, T, Vs, Vc, Fn, CB, 0, kyy)
        alpha_matrix = [np.pi / 2, 0]
        a1_matrix = [a1_head, a1_following]
        a2_matrix = [a2_head, a2_following]

        a1_f = interp1d(alpha_matrix, a1_matrix)
        a1 = a1_f(alpha)
        a2_f = interp1d(alpha_matrix, a2_matrix)
        a2 = a2_f(alpha)

    W = (
        2.142
        * (kyy ** (1 / 3))
        * np.sqrt(Lpp / lambda_)
        * (1 - (0.111 / CB) * (np.log(B / T) - np.log(2.75)))
        * ((CB / 0.65) ** 0.17)
        * (
            (-1.377 * Fn**2 + 1.157 * Fn) * np.abs(np.cos(alpha))
            + (0.618 * (13 + np.cos(2 * alpha))) / 14
        )
    )

    a3 = 1.0 + 28.7 * np.arctan(np.abs(Ta - Tf) / Lpp)

    b1 = np.zeros(np.size(W))
    b1[W < 1] = 11
    b1[W >= 1] = -8.5

    d1 = np.zeros(np.size(W))

    d1[W < 1] = 566 * (Lpp * CB / B) ** (-2.66)
    d1[W >= 1] = (
        -566 * (Lpp / B) ** (-2.66) * (4 - (125 * np.arctan(np.abs(Ta - Tf) / Lpp)))
    )

    R_AWML = 4 * a1 * a2 * a3 * (W**b1) * np.exp((b1 / d1) * (1 - W**d1))

    return R_AWML[0], a1, a2, a3, W, b1, d1


def Diffraction(Lpp, B, T, Vs, Fn, CB, E1, E2, alpha, lambda_, omega, g):
    """
    calculate the component of added resistance due to reflection (diffraction)
    (Liu&Papanikolaou, 2020)

    Parameters
    ----------
    Lpp: float
        length between perpendiculars in meters
    B: float
        breadth in meters
    T: float
        mean draught in meters
    Vs: float
        ship speed in m/s
    Fn: float
        froude number
    CB: float
        block coefficient
    E1: float
    E2: float
    alpha: float
        wave heading (Head waves: 180, Following waves:0)
    lambda_: float
        wavelength (lambda)
    omega: float
        circular frequency of regular waves
    g: float
        gravitational acceleration

    Returns
    -------
    RAWRL: float
        added resistance due to reflection (diffraction), shorter wave length range dominant
    """
    if (alpha >= (np.pi - E1)) and (alpha <= np.pi):
        f = -np.cos(alpha)
    elif alpha < (np.pi - E1):
        f = 0

    R1_AWRL = 0
    R2_AWRL = 0
    R3_AWRL = 0
    R4_AWRL = 0

    if (alpha <= np.pi) and (alpha >= E1):
        T_st = T
        if (lambda_ / Lpp) <= 2.5:
            alpha_T = 1 - np.exp(-4 * np.pi * ((T_st / lambda_) - (T_st / (2.5 * Lpp))))
        elif (lambda_ / Lpp) > 2.5:
            alpha_T = 0
        R1_AWRL = (
            (2.25 / 4)
            * B
            * alpha_T
            * (
                (np.sin(E1 - alpha)) ** 2
                + (2 * omega * Vs / g)
                * (np.cos(E1) * np.cos(E1 - alpha) - np.cos(alpha))
            )
            * (0.87 / CB) ** ((1 + 4 * np.sqrt(Fn)) * f)
        )
    if (alpha <= np.pi) and (alpha >= (np.pi - E1)):
        T_st = T
        if (lambda_ / Lpp) <= 2.5:
            alpha_T = 1 - np.exp(-4 * np.pi * ((T_st / lambda_) - (T_st / (2.5 * Lpp))))
        elif (lambda_ / Lpp) > 2.5:
            alpha_T = 0
        R2_AWRL = (
            (2.25 / 4)
            * B
            * alpha_T
            * (
                (np.sin(E1 + alpha)) ** 2
                + (2 * omega * Vs / g)
                * ((np.cos(E1) * np.cos(E1 + alpha)) - np.cos(alpha))
            )
            * (0.87 / CB) ** ((1 + 4 * np.sqrt(Fn)) * f)
        )
    if (alpha <= (np.pi - E2)) and (alpha >= 0):
        if CB <= 0.75:
            T_st = T * (4 + np.sqrt(np.abs(np.cos(alpha)))) / 5
        elif CB > 0.75:
            T_st = T * (2 + np.sqrt(np.abs(np.cos(alpha)))) / 3

        if (lambda_ / Lpp) <= 2.5:
            alpha_T = 1 - np.exp(-4 * np.pi * ((T_st / lambda_) - (T_st / (2.5 * Lpp))))
        elif (lambda_ / Lpp) > 2.5:
            alpha_T = 0
        R3_AWRL = (
            (-2.25 / 4)
            * B
            * alpha_T
            * (
                (np.sin(E2 + alpha)) ** 2
                + (2 * omega * Vs / g)
                * (np.cos(E2) * np.cos(E2 + alpha) - np.cos(alpha))
            )
        )

    if (alpha <= E2) and (alpha >= 0):
        if CB <= 0.75:
            T_st = T * (4 + np.sqrt(np.abs(np.cos(alpha)))) / 5
        elif CB > 0.75:
            T_st = T * (2 + np.sqrt(np.abs(np.cos(alpha)))) / 3

        if (lambda_ / Lpp) <= 2.5:
            alpha_T = 1 - np.exp(-4 * np.pi * ((T_st / lambda_) - (T_st / (2.5 * Lpp))))
        elif (lambda_ / Lpp) > 2.5:
            alpha_T = 0
        R4_AWRL = (
            (-2.25 / 4)
            * B
            * alpha_T
            * (
                (np.sin(E2 - alpha)) ** 2
                + (2 * omega * Vs / g)
                * (np.cos(E2) * np.cos(E2 - alpha) - np.cos(alpha))
            )
        )

    R_AWRL = (R1_AWRL + R2_AWRL + R3_AWRL + R4_AWRL) * (Lpp / B**2)

    return (
        R_AWRL,
        R1_AWRL * (Lpp / B**2),
        R2_AWRL * (Lpp / B**2),
        R3_AWRL * (Lpp / B**2),
        R4_AWRL * (Lpp / B**2),
    )


def Liu_method_alldir(
    Lpp, B, T, Tf, Ta, Le, Lr, CB, Fn, alpha, kyy=0.25, save_detail: bool = False
):
    """
    calculate dimensionless added wave resistance coefficient using L&P method
    (Liu&Papanikolaou, 2020)

    Parameters
    ----------
    Lpp: float
        length between perpendiculars in meters
    B: float
        breadth in meters
    T: float
        mean draught in meters
    Tf: float
        forward draught in meters
    Ta: float
        aft draught in meters
    Le: float
        length of entrance in meters
    Lr: float
        length of run in meters
    CB: float
        block coefficient
    Fn: float
        froude number
    alpha: float
        wave heading (Head waves: 180, Following waves:0)
    kyy: float
        radius of gyration of pitch

    Returns
    -------
    Raw: float
        dimensionless added wave resistance coefficient
    """
    E1 = np.arctan(0.495 * B / (Le))
    E2 = np.arctan(0.495 * B / (Lr))

    lambda_LBP = np.arange(0.15, 2.05, 0.01)

    g = 9.81

    alpha_ = alpha * np.pi / 180

    lambda_ = Lpp * lambda_LBP

    k = 2 * np.pi / lambda_
    omega = np.sqrt(k * g)

    Vc = np.sqrt((g * lambda_) / (2 * np.pi))
    Vs = Fn * np.sqrt(g * Lpp)
    R_AWML = []
    A1 = []
    A2 = []
    A3 = []
    WL = []
    B1 = []
    D1 = []
    R1_AWRL = []
    R2_AWRL = []
    R3_AWRL = []
    R4_AWRL = []
    R_AWRL = []
    for l, v_c, w in zip(lambda_, Vc, omega):
        r_awm, a1, a2, a3, W, b1, d1 = Radiation_head_waves(
            Lpp, T, Tf, Ta, B, CB, Fn, l, alpha_, Vs, v_c, kyy
        )
        R_AWML.append(r_awm)
        A1.append(a1)
        A2.append(a2)
        A3.append(a3)
        WL.append(W)
        B1.append(b1[0])
        D1.append(d1[0])
        r_awrl, r1_awrl, r2_awrl, r3_awrl, r4_awrl = Diffraction(
            Lpp, B, T, Vs, Fn, CB, E1, E2, alpha_, l, w, g
        )
        R_AWRL.append(r_awrl)
        R1_AWRL.append(r1_awrl)
        R2_AWRL.append(r2_awrl)
        R3_AWRL.append(r3_awrl)
        R4_AWRL.append(r4_awrl)

    # R_AWML = [Radiation_head_waves(Lpp, T, Tf, Ta, B, CB, Fn, lambda_[j], alpha_, Vs, Vc[j], kyy).tolist() for j in range(np.size(lambda_))]
    # R_AWML= [val for sublist in R_AWML for val in sublist]
    # R_AWRL = [Diffraction(Lpp, B, T, Vs, Fn, CB, E1, E2, alpha_, lambda_[j], omega[j], g) for j in range(np.size(lambda_))]
    Raw = np.array(R_AWML) + np.array(R_AWRL)
    if save_detail:
        df = pd.DataFrame()
        df["lambda_non_dim"] = lambda_LBP
        df["a1"] = A1
        df["a2"] = A2
        df["a3"] = A3
        df["W"] = WL
        df["b1"] = B1
        df["d1"] = D1
        df["R1_AWRL"] = R1_AWRL
        df["R2_AWRL"] = R2_AWRL
        df["R3_AWRL"] = R3_AWRL
        df["R4_AWRL"] = R4_AWRL
        df["R_AWRL"] = R_AWRL
        df["R_AWML"] = R_AWML
        df["Raw"] = Raw
        df.index = df["lambda_non_dim"]
        df.drop("lambda_non_dim", axis=1, inplace=True)
        df.to_csv(f"Liu_method_alldir_output_{180 - theta}.csv")

    return np.array(R_AWML), np.array(R_AWRL), Raw


## Example case: General cargo
Lpp = 194
B = 32.266
Tmax, Tf, Ta = 12.64, 12.64, 12.64
Cb = 0.79
Fn = 0.182
kyy = 0.25
Le = 38.51
Lr = 32.31

fig = make_subplots()

lambda_non_dim = np.arange(0.15, 2.05, 0.01)
df = pd.DataFrame(index=lambda_non_dim)
theta = 60
# for theta in [180, 150, 120, 90, 60, 30, 0]:
r_awm, r_awr, r_aw = Liu_method_alldir(
    Lpp, B, Tmax, Tf, Ta, Le, Lr, Cb, Fn, theta, kyy, save_detail=True
)
# df[f"r_awm-{theta}"] = r_awm
# df[f"r_awr-{theta}"] = r_awr
# df[f"r_aw-{theta}"] = r_aw
# df.to_csv(f"Liu_method_alldir_output_following_sea_{180-theta}.csv")
# fig = df.plot()
# fig.show()
