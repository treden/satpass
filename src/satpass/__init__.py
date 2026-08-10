"""Tools for computing satellite passes from nominal altimeter tracks."""

from .colocation import (
    colocate_swot_passages,
    colocate_xy,
    colocate_xyt,
    distance_to_track,
)

from .satpass import (
    NominalTrack,
    assign_subcycles_to_passes,
    assign_subcycles_to_tracks,
    circular_mean_lon_360,
    disentangle_two_subcycles,
    ephemeris,
    remove_incomplete_subcycles,
    sat_pass,
)

__all__ = [
    "NominalTrack",
    "assign_subcycles_to_passes",
    "assign_subcycles_to_tracks",
    "circular_mean_lon_360",
    "colocate_swot_passages",
    "colocate_xy",
    "colocate_xyt",
    "disentangle_two_subcycles",
    "distance_to_track",
    "ephemeris",
    "remove_incomplete_subcycles",
    "sat_pass",
]

__version__ = "0.1.0"
