import os
import random
from typing import TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ship_model_lib.calm_water_resistance import (
    CalmWaterResistanceBySpeedPowerCurve,
    CalmWaterResistanceBySpeedResistanceCurve,
    CalmWaterResistanceHollenbachSingleScrewDesignDraft,
    CalmWaterResistanceHollenbachTwinScrewDesignDraft,
    ResistanceLevel,
    ShipDimensionsHollenbachSingleScrew,
    ShipDimensionsHollenbachTwinScrew,
)


def test_calm_water_resistance_by_speed_resistance_curve():
    speed_kn = np.array([2, 4, 6, 8, 10, 12, 14])
    power_kw = np.power(speed_kn, 3) * 5

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=speed_kn, y=power_kw, mode="markers", name="Points given"))

    resistance_model = CalmWaterResistanceBySpeedPowerCurve(
        speed_ref_kn=speed_kn, power_ref_kw=power_kw
    )
    speed_new = np.linspace(0, speed_kn.max(), 100)
    power_estimated = resistance_model.get_power_from_speed(speed_kn=speed_new)
    fig.add_trace(go.Scatter(x=speed_new, y=power_estimated.value, name="Power Estimated"))
    power_new = np.linspace(0, power_kw.max(), 100)
    speed_estimated = resistance_model.get_speed_from_power(power_kw=power_new)
    fig.add_trace(go.Scatter(x=speed_estimated.value, y=power_new, name="Speed Estimated"))
    fig.update_layout(title="Speed vs Power")
    fig.show(renderer="browser")

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "hollenbach_design_draft_resistance.csv",
    )
    file_path = os.path.abspath(file_path)
    df = pd.read_csv(file_path)
    print(df.vs)
    power_estimated = resistance_model.get_power_from_speed(speed_kn=df.vs.values).value
    print(power_estimated)
    resistance_k_n = np.power(speed_kn, 2) * 3

    fig = make_subplots()
    fig.add_trace(go.Scatter(x=speed_kn, y=resistance_k_n, mode="markers", name="Points given"))

    resistance_model = CalmWaterResistanceBySpeedResistanceCurve(
        speed_ref_kn=speed_kn, resistance_ref_k_n=resistance_k_n
    )
    speed_new = np.linspace(0, speed_kn.max(), 100)
    resistance_estimated = resistance_model.get_resistance_from_speed(velocity_kn=speed_new)
    fig.add_trace(
        go.Scatter(x=speed_new, y=resistance_estimated.value, name="Resistance Estimated")
    )
    resistance_new = np.linspace(0, resistance_k_n.max(), 100)
    speed_estimated = resistance_model.get_speed_from_resistance(resistance_k_n=resistance_new)
    fig.add_trace(go.Scatter(x=speed_estimated.value, y=resistance_new, name="Speed Estimated"))
    fig.update_layout(title="Speed vs Resistance Force")
    fig.show(renderer="browser")


def test_the_code_for_hollenbach_method():
    pd.options.plotting.backend = "plotly"

    # Main dimensions
    l_pp = 145
    l_fore = 3.3
    l_aft = 2.7
    b_beam = 24
    t_draft = 8.2
    v_displacement = 18872.0
    dp = 4.9

    l_wl = l_pp + l_aft
    l_os = l_wl + l_fore
    cb = v_displacement / (l_pp * b_beam * t_draft)

    Numeric = TypeVar("Numeric", float, np.ndarray)

    def kn_to_m_per_s(v_kn: Numeric) -> Numeric:
        return v_kn * 0.5144

    ship_dimensions_single_screw = ShipDimensionsHollenbachSingleScrew(
        b_beam_m=b_beam,
        lpp_length_between_perpendiculars_m=l_pp,
        lwl_length_water_line_m=l_wl,
        los_length_over_surface_m=l_os,
        cb_block_coefficient=cb,
        dp_diameter_propeller_m=dp,
        ta_draft_aft_m=t_draft,
        tf_draft_forward_m=t_draft,
        av_transverse_area_above_water_line_m2=383.76,
        area_bilge_keel_m2=52,
    )

    ship_model = CalmWaterResistanceHollenbachSingleScrewDesignDraft(
        ship_dimensions=ship_dimensions_single_screw
    )

    speed_kn = np.linspace(1, 19, 20)
    resistance_force_k_n_minimum = ship_model.get_resistance_from_speed(
        speed_kn, resistance_level=ResistanceLevel.MINIMUM
    )
    resistance_force_k_n_mean = ship_model.get_resistance_from_speed(
        speed_kn, resistance_level=ResistanceLevel.MEAN
    )
    resistance_force_k_n_mean_with_form_factor = ship_model.get_resistance_from_speed(
        speed_kn, resistance_level=ResistanceLevel.MEAN, form_factor=0.6
    )
    resistance_force_k_n_maximum = ship_model.get_resistance_from_speed(
        speed_kn, resistance_level=ResistanceLevel.MAXIMUM
    )
    df_to_plot = pd.DataFrame(index=speed_kn)
    df_to_plot["resistance_min"] = resistance_force_k_n_minimum
    df_to_plot["resistance_mean"] = resistance_force_k_n_mean
    df_to_plot["resistance_mean_form_factor"] = resistance_force_k_n_mean_with_form_factor
    df_to_plot["resistance_max"] = resistance_force_k_n_maximum
    fig = df_to_plot.plot()
    fig.update_xaxes(title="Speed [kn]")
    fig.update_yaxes(title="Resistance [kN]")

    speed_kn_ref = random.random() * 19
    resistance_k_n_ref = ship_model.get_resistance_from_speed(speed_kn_ref)
    speed_kn_estimated = ship_model.get_speed_from_resistance(resistance_k_n=resistance_k_n_ref)
    fig.add_trace(
        go.Scatter(x=[speed_kn_estimated], y=[resistance_k_n_ref], name="Resistance to speed")
    )
    fig.show(renderer="browser")

    assert np.isclose(speed_kn_ref, speed_kn_estimated), (
        f"The estimated speed - {speed_kn_estimated} - is not equal to the answer - {speed_kn_ref}."
    )

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "hollenbach_design_draft_coeff_ref.csv",
    )
    file_path = os.path.abspath(file_path)
    df_output_ref = pd.read_csv(file_path)

    assert np.isclose(ship_model._lfn_length_froude_number, 149.0)
    assert np.isclose(ship_model._k_l, 0.9778, rtol=1e-4)
    assert np.isclose(ship_model._k_bt, 1.4379, rtol=1e-4)
    assert np.isclose(ship_model._k_lb, 0.2335, rtol=1e-3)
    assert np.isclose(ship_model._k_ll, 0.8753, rtol=1e-4)
    assert np.isclose(ship_model._k_ao, 0.9364, rtol=1e-4)
    assert np.isclose(ship_model._k_pr, 0.9925, rtol=1e-4)
    assert np.isclose(ship_model._k_tr, 1.0000, rtol=1e-4)
    assert np.isclose(ship_model._k_shape_factor, 0.75938, rtol=1e-5)
    assert np.isclose(ship_model.wetted_surface_area, 4448.453)

    # Residual resistance coefficient estimation
    df_output = pd.DataFrame()
    df_output["vs"] = df_output_ref.vs.values
    df_output["fn"] = ship_model._get_fn(kn_to_m_per_s(df_output_ref.vs.values))
    fn_vec = df_output["fn"].values
    df_output["c_r_min_std"] = ship_model._get_c_r_standard_min(fn_vec)
    df_output["k_fr"] = ship_model._get_k_fr_high_froude_number_factor(fn_vec)
    df_output["c_r_min_bt"] = ship_model._get_c_r_bt_min(fn_vec)
    df_output["c_r_min"] = ship_model._get_c_r_min(fn_vec)
    df_output["c_r_std"] = ship_model._get_c_r_standard(fn_vec)
    df_output["c_r_bt"] = ship_model._get_c_r_bt(fn_vec)
    df_output["c_r"] = ship_model._get_c_r(fn_vec)

    for col_name in df_output:
        assert np.all(np.isclose(df_output_ref[col_name].values, df_output[col_name].values))

    # Resistance estimation
    assert np.isclose(ship_model._c_a, 0.06e-3)
    assert np.isclose(ship_model._c_aas, 0.08248e-3, rtol=1e-4)

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "hollenbach_design_draft_coeff_all.csv",
    )
    file_path = os.path.abspath(file_path)
    df_output_coeff_ref = pd.read_csv(file_path)

    df_output_coeff = pd.DataFrame()
    df_output_coeff["vs"] = df_output_ref.vs.values
    df_output_coeff["fn"] = fn_vec
    df_output_coeff["re"] = ship_model.get_reynolds_number(kn_to_m_per_s(df_output.vs.values))
    df_output_coeff["c_r_min"] = ship_model._get_c_r_min(fn_vec)
    df_output_coeff["c_r"] = ship_model._get_c_r(fn_vec)
    df_output_coeff["c_f"] = ship_model._get_c_f(kn_to_m_per_s(df_output.vs.values))
    df_output_coeff["c_app"] = ship_model._get_c_app(kn_to_m_per_s(df_output.vs.values))
    df_output_coeff["c_t_min"] = (
        df_output_coeff.c_r_min
        + df_output_coeff.c_f
        + df_output_coeff.c_app
        + ship_model._c_aas
        + ship_model._c_a
    )
    df_output_coeff["c_t"] = (
        df_output_coeff.c_r
        + df_output_coeff.c_f
        + df_output_coeff.c_app
        + ship_model._c_aas
        + ship_model._c_a
    )
    df_output_coeff["c_t_max"] = df_output_coeff.c_t * ship_model._factor_for_max_resistance
    for col_name in df_output_coeff:
        assert np.all(
            np.isclose(df_output_coeff_ref[col_name].values, df_output_coeff[col_name].values)
        )

    file_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "test_data",
        "hollenbach_design_draft_resistance.csv",
    )
    file_path = os.path.abspath(file_path)
    df_output_resistance_ref = pd.read_csv(file_path)

    df_output_resistance = pd.DataFrame()
    df_output_resistance["vs"] = df_output_ref.vs.values
    df_output_resistance["fn"] = fn_vec
    df_output_resistance["re"] = ship_model.get_reynolds_number(kn_to_m_per_s(df_output.vs.values))
    df_output_resistance["r_t_min"] = ship_model._get_total_resistance_min(df_output.vs.values)
    df_output_resistance["r_t"] = ship_model._get_total_resistance_mean(df_output.vs.values)
    df_output_resistance["r_t_max"] = ship_model._get_total_resistance_max(df_output.vs.values)
    for col_name in df_output_resistance:
        assert np.all(
            np.isclose(
                df_output_resistance_ref[col_name].values,
                df_output_resistance[col_name].values,
            )
        )
    df_output_resistance.plot(x="vs", y=["r_t_min", "r_t", "r_t_max"]).show(renderer="browser")

    ship_dimensions_twin_screw = ShipDimensionsHollenbachTwinScrew(
        b_beam_m=b_beam,
        lpp_length_between_perpendiculars_m=l_pp,
        lwl_length_water_line_m=l_wl,
        los_length_over_surface_m=l_os,
        cb_block_coefficient=cb,
        dp_diameter_propeller_m=dp,
        ta_draft_aft_m=t_draft,
        tf_draft_forward_m=t_draft,
        av_transverse_area_above_water_line_m2=383.76,
        area_bilge_keel_m2=52,
        number_rudders=2,
        number_shaft_bossings=2,
        number_shaft_brackets=2,
        number_thrusters=2,
    )
    ship_model = CalmWaterResistanceHollenbachTwinScrewDesignDraft(
        ship_dimensions=ship_dimensions_twin_screw
    )

    print(ship_model.wetted_surface_area)

    df_output = pd.DataFrame()
    df_output.index = df_output_ref.vs.values
    df_output["r_t_min"] = ship_model.get_resistance_from_speed(
        df_output.index.values, resistance_level=ResistanceLevel.MINIMUM
    )
    df_output["r_t"] = ship_model.get_resistance_from_speed(df_output.index.values)
    df_output["r_t_max"] = ship_model.get_resistance_from_speed(
        df_output.index.values, resistance_level=ResistanceLevel.MAXIMUM
    )

    fig = df_output.plot()
    fig.update_xaxes(title="Speed [kn]")
    fig.update_yaxes(title="Resistance [kN]")
    fig.show(renderer="browser")
