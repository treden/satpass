"""Associate fixed or moving in-situ observations with SWOT passages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .satpass import NominalTrack, sat_pass


_EARTH_RADIUS_M = 6_371_008.8


def _as_1d(values):
    """Return scalars and array-like inputs as a one-dimensional array."""
    return np.atleast_1d(values).reshape(-1)


def _broadcast_observations(longitude, latitude, time):
    """Broadcast a fixed mooring position or validate a moving trajectory."""
    longitude = _as_1d(longitude).astype(float)
    latitude = _as_1d(latitude).astype(float)
    time = pd.DatetimeIndex(_as_1d(time))

    size = max(len(longitude), len(latitude), len(time))
    arrays = []
    for name, values in (
        ("longitude", longitude),
        ("latitude", latitude),
        ("time", time),
    ):
        if len(values) == 1:
            values = np.repeat(values, size)
        elif len(values) != size:
            raise ValueError(
                f"{name} has {len(values)} values; expected either 1 or {size}"
            )
        arrays.append(values)

    return arrays[0], arrays[1], pd.DatetimeIndex(arrays[2])


def distance_to_track(longitude, latitude, track, *, max_segment_km=100.0):
    """Approximate the shortest geodesic distance to a sampled ground track.

    The query is projected onto every line segment, so the result is not
    limited by the spacing of track vertices. Track segments are projected
    onto a local tangent plane centred on the query point. Missing points and
    jumps longer than ``max_segment_km`` break the polyline instead of being
    bridged by an artificial segment.
    """
    coordinates = track[["lon", "lat"]].to_numpy(dtype=float)
    valid_points = np.isfinite(coordinates).all(axis=1)
    if not valid_points.any():
        return np.nan

    longitude_delta = (coordinates[:, 0] - longitude + 180.0) % 360.0 - 180.0
    x = np.deg2rad(longitude_delta) * _EARTH_RADIUS_M * np.cos(
        np.deg2rad(latitude)
    )
    y = np.deg2rad(coordinates[:, 1] - latitude) * _EARTH_RADIUS_M
    points = np.column_stack((x, y))

    point_distances = np.hypot(points[valid_points, 0], points[valid_points, 1])
    if len(points) == 1:
        return float(point_distances[0])

    starts = points[:-1]
    segments = points[1:] - starts
    consecutive = valid_points[:-1] & valid_points[1:]

    lon1 = np.deg2rad(coordinates[:-1, 0])
    lon2 = np.deg2rad(coordinates[1:, 0])
    lat1 = np.deg2rad(coordinates[:-1, 1])
    lat2 = np.deg2rad(coordinates[1:, 1])
    delta_lon = (lon2 - lon1 + np.pi) % (2 * np.pi) - np.pi
    delta_lat = lat2 - lat1
    haversine = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    segment_length_km = (
        2 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))) / 1000
    )
    usable = consecutive & (segment_length_km <= max_segment_km)
    if not usable.any():
        return float(np.min(point_distances))

    starts = starts[usable]
    segments = segments[usable]
    length_squared = np.einsum("ij,ij->i", segments, segments)
    fraction = np.divide(
        -np.einsum("ij,ij->i", starts, segments),
        length_squared,
        out=np.zeros_like(length_squared),
        where=length_squared > 0,
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    nearest = starts + fraction[:, None] * segments
    return float(np.min(np.hypot(nearest[:, 0], nearest[:, 1])))


def colocate_xy(
    x,
    y,
    date_range,
    *,
    max_distance_km=65.0,
    mission="sw",
):
    """Return every nominal SWOT passage over one position in an interval.

    This is the fixed-mooring operation: ``x`` and ``y`` are scalar longitude
    and latitude, and ``date_range`` contains the inclusive start and end UTC
    times. The returned ``time`` column is the predicted SWOT passage time.
    """
    if mission not in {"sw", "swi"}:
        raise ValueError("mission must be 'sw' or 'swi'")
    if not np.isfinite(x) or not -180 <= x <= 180:
        raise ValueError("x must be a finite longitude between -180 and 180")
    if not np.isfinite(y) or not -90 <= y <= 90:
        raise ValueError("y must be a finite latitude between -90 and 90")
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")

    dates = pd.DatetimeIndex(date_range)
    if len(dates) != 2:
        raise ValueError("date_range must contain exactly [start, end]")
    if dates.isna().any() or dates[1] < dates[0]:
        raise ValueError("date_range must contain a valid start before end")

    margin_degrees = max(1.0, max_distance_km / 111.0 + 0.25)
    domain = [
        max(-180.0, float(x - margin_degrees)),
        min(180.0, float(x + margin_degrees)),
        max(-90.0, float(y - margin_degrees)),
        min(90.0, float(y + margin_degrees)),
    ]
    nominal = NominalTrack(mission)
    tracks = nominal.select_in_area(domain, extend_to_latitude_bounds=True)
    tracks = tracks.groupby("track").filter(lambda group: len(group) > 1)

    output_columns = [
        "x",
        "y",
        "time",
        "track",
        "cycle",
        "direction",
        "distance_to_nadir_km",
    ]
    if tracks.empty:
        return pd.DataFrame(columns=output_columns)

    distances = {
        number: distance_to_track(x, y, track)
        for number, track in tracks.groupby("track")
    }
    nearby = {
        number: distance
        for number, distance in distances.items()
        if distance <= max_distance_km * 1000.0
    }
    if not nearby:
        return pd.DataFrame(columns=output_columns)

    passages = sat_pass(nominal, dates, domain)
    passages = passages[passages["track"].isin(nearby)].copy()
    passages["distance_to_nadir_km"] = passages["track"].map(nearby) / 1000.0
    passages.insert(0, "y", float(y))
    passages.insert(0, "x", float(x))
    return passages.sort_values("time").reset_index(drop=True)


def colocate_xyt(
    x,
    y,
    t,
    *,
    n_passes=1,
    max_distance_km=65.0,
    search_window_days=10.0,
    mission="sw",
):
    """Associate fixed or moving ``(x, y, t)`` observations with SWOT.

    Unlike :func:`colocate_xy`, this operation starts from observation times
    and ranks the closest SWOT passage for each observation.
    """
    return colocate_swot_passages(
        longitude=x,
        latitude=y,
        time=t,
        n_passes=n_passes,
        max_distance_km=max_distance_km,
        search_window_days=search_window_days,
        mission=mission,
    )


def colocate_swot_passages(
    longitude,
    latitude,
    time,
    *,
    n_passes=1,
    max_distance_km=65.0,
    search_window_days=10.0,
    mission="sw",
):
    """Associate observations with their nearest nominal SWOT passages.

    ``longitude`` and ``latitude`` may describe a moving platform, or each may
    be a scalar for a fixed mooring.  For every observation, candidate passes
    whose nominal nadir track lies within ``max_distance_km`` are ranked by
    absolute time separation.

    Parameters
    ----------
    longitude, latitude : scalar or array-like
        Observation coordinates in degrees east and north.
    time : scalar or array-like
        Observation timestamps. Naive timestamps are interpreted as UTC.
    n_passes : int, default 1
        Number of closest-in-time passage matches to return per observation.
    max_distance_km : float, default 65
        Maximum distance from the nominal nadir track. The default represents
        one side of the approximate SWOT KaRIn swath, excluding the nadir gap.
    search_window_days : float, default 10
        Passes are generated this many days before and after the observations.
    mission : {"sw", "swi"}, default "sw"
        SWOT science or calibration/validation nominal orbit.

    Returns
    -------
    pandas.DataFrame
        Original position/time plus ``swot_time``, ``time_offset_days``,
        ``track``, ``cycle``, ``direction``, and ``distance_to_nadir_km``.
        When ``n_passes`` is greater than one, match columns receive ``_1``,
        ``_2``, ... suffixes. Missing input rows are retained with null matches.
    """
    if mission not in {"sw", "swi"}:
        raise ValueError("mission must be 'sw' or 'swi'")
    if not isinstance(n_passes, (int, np.integer)) or n_passes < 1:
        raise ValueError("n_passes must be a positive integer")
    if max_distance_km <= 0:
        raise ValueError("max_distance_km must be positive")
    if search_window_days < 0:
        raise ValueError("search_window_days cannot be negative")

    longitude, latitude, observation_time = _broadcast_observations(
        longitude, latitude, time
    )
    result = pd.DataFrame(
        {
            "longitude": longitude,
            "latitude": latitude,
            "observation_time": observation_time,
        }
    )

    valid = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & observation_time.notna()
        & (latitude >= -90)
        & (latitude <= 90)
    )
    suffixes = [""] if n_passes == 1 else [f"_{i}" for i in range(1, n_passes + 1)]
    for suffix in suffixes:
        result[f"swot_time{suffix}"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
        result[f"time_offset_days{suffix}"] = np.nan
        result[f"track{suffix}"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result[f"cycle{suffix}"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result[f"direction{suffix}"] = pd.Series(
            None, index=result.index, dtype="object"
        )
        result[f"distance_to_nadir_km{suffix}"] = np.nan

    if not valid.any():
        return result

    valid_longitude = longitude[valid]
    valid_latitude = latitude[valid]
    valid_time = observation_time[valid]
    margin_degrees = max(1.0, max_distance_km / 111.0 + 0.25)
    domain = [
        max(-180.0, float(np.min(valid_longitude) - margin_degrees)),
        min(180.0, float(np.max(valid_longitude) + margin_degrees)),
        max(-90.0, float(np.min(valid_latitude) - margin_degrees)),
        min(90.0, float(np.max(valid_latitude) + margin_degrees)),
    ]
    date_range = [
        valid_time.min() - pd.Timedelta(days=search_window_days),
        valid_time.max() + pd.Timedelta(days=search_window_days),
    ]

    nominal = NominalTrack(mission)
    tracks = nominal.select_in_area(domain, extend_to_latitude_bounds=True)
    tracks = tracks.groupby("track").filter(lambda group: len(group) > 1)
    if tracks.empty:
        return result

    passages = sat_pass(nominal, date_range, domain).reset_index(drop=True)
    if passages.empty:
        return result

    track_groups = {number: group for number, group in tracks.groupby("track")}
    valid_indices = np.flatnonzero(valid)
    for row_index, x, y, observed_at in zip(
        valid_indices, valid_longitude, valid_latitude, valid_time
    ):
        distances = {
            number: distance_to_track(x, y, track)
            for number, track in track_groups.items()
        }
        nearby = {
            number: distance
            for number, distance in distances.items()
            if distance <= max_distance_km * 1000.0
        }
        if not nearby:
            continue

        candidates = passages[passages["track"].isin(nearby)].copy()
        candidates["distance_to_nadir_km"] = candidates["track"].map(nearby) / 1000.0
        candidates["time_offset_days"] = (
            candidates["time"] - observed_at
        ).dt.total_seconds() / 86_400.0
        candidates = candidates.loc[
            candidates["time_offset_days"].abs().sort_values().index
        ].head(n_passes)

        for match_number, (_, candidate) in enumerate(candidates.iterrows(), start=1):
            suffix = "" if n_passes == 1 else f"_{match_number}"
            result.at[row_index, f"swot_time{suffix}"] = candidate["time"]
            result.at[row_index, f"time_offset_days{suffix}"] = candidate[
                "time_offset_days"
            ]
            result.at[row_index, f"track{suffix}"] = int(candidate["track"])
            result.at[row_index, f"cycle{suffix}"] = int(candidate["cycle"])
            result.at[row_index, f"direction{suffix}"] = candidate["direction"]
            result.at[row_index, f"distance_to_nadir_km{suffix}"] = candidate[
                "distance_to_nadir_km"
            ]

    return result
