from ship_model_lib.calm_water_resistance import (
    CalmWaterResistanceHollenbachSingleScrewDesignDraft,
)

from ship_model_lib.ship_dimensions import (
ShipDimensionsHollenbachSingleScrew,
ShipDimensionsAddedResistance,
)

from dataclasses import dataclass

def test_ship_dimension_test():
    # | hide

    b_beam_m = 24
    lpp_length_between_perpendiculars_m = 145
    los_length_over_surface_m = 150
    lwl_length_water_line_m = 146.7
    cb_block_coefficient = 0.75
    ta_draft_aft_m = 8.2
    tf_draft_forward_m = 8.2
    wetted_surface_m2 = 4400
    av_transverse_area_above_water_line_m2 = 2
    area_bilge_keel_m2 = 1
    kyy_radius_gyration_in_lateral_direction_non_dim = 0.26
    propeller_diameter = 9.81

    @dataclass
    class ShipDimensionTest(
        ShipDimensionsHollenbachSingleScrew, ShipDimensionsAddedResistance
    ):
        pass

    test_ship_dimensions = ShipDimensionTest(
        b_beam_m=b_beam_m,
        lpp_length_between_perpendiculars_m=lpp_length_between_perpendiculars_m,
        los_length_over_surface_m=los_length_over_surface_m,
        lwl_length_water_line_m=lwl_length_water_line_m,
        cb_block_coefficient=cb_block_coefficient,
        ta_draft_aft_m=ta_draft_aft_m,
        tf_draft_forward_m=tf_draft_forward_m,
        wetted_surface_m2=wetted_surface_m2,
        area_bilge_keel_m2=area_bilge_keel_m2,
        av_transverse_area_above_water_line_m2=av_transverse_area_above_water_line_m2,
        kyy_radius_gyration_in_lateral_direction_non_dim=kyy_radius_gyration_in_lateral_direction_non_dim,
        dp_diameter_propeller_m=propeller_diameter,
    )

    calm_water_hollenbach_single_screw = (
        CalmWaterResistanceHollenbachSingleScrewDesignDraft(
            ship_dimensions=test_ship_dimensions
        )
    )
    calm_water_hollenbach_single_screw.get_resistance_from_speed(velocity_kn=10)



