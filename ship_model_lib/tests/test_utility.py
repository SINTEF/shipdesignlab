
from ship_model_lib.utility import get_interpolation_1d_function
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

def test_interpolation_1d_function():

    interp_x_data = np.array([1, 2, 3, 4])
    interp_y_data = np.array([0.5, 4, 3.5, 2.4])

    interp_function_with_out_origo = get_interpolation_1d_function(
        x=interp_x_data, y=interp_y_data, add_origo=False
    )

    assert (
        interp_function_with_out_origo(0).value != 0
    ), "The function should not be zero when origin was not added."
    assert interp_function_with_out_origo(
        0
    ).is_extrapolated, "The value is extrapolated, while the object says that it is not."

    x_new = np.linspace(0, 5, 51)
    y_new = interp_function_with_out_origo(x_new).value
    fig = make_subplots()
    fig.add_trace(
        go.Scatter(x=interp_x_data, y=interp_y_data, name="Data given", mode="markers")
    )
    fig.add_trace(
        go.Scatter(x=x_new, y=y_new, name="Interpolation without adding the origin")
    )

    interp_function_with_origo = get_interpolation_1d_function(
        x=interp_x_data, y=interp_y_data, add_origo=True
    )

    assert (
        interp_function_with_origo(0).value == 0
    ), "The function should be zero when origin was added."
    assert interp_function_with_origo(
        0
    ).is_extrapolated, "The value is extrapolated, while the object says that it is not."

    x_new = np.linspace(0, 5, 51)
    y_new = interp_function_with_origo(x_new).value
    fig.add_trace(go.Scatter(x=x_new, y=y_new, name="Interpolation with adding the origin"))
    fig.show(renderer="png")