#!/usr/bin/python3
# -*- coding: utf8 -*-
"""
This module contains tools to compute satellite past and future passes on a given area.
Originally developed by C. Chupin under the PyGoat toolbox (CC, YTT, VB, LT)
01/2024 (YTT): Adaptation to SWOT altimetry and extraction from the PyGoat toolbox as an independent module

.. Warning:: This code is still under development, and significant changes may take place in future versions.

.. todo::
    - Address the issue with Sentinel tracks!!
    - Add CFOSAt
    - Optimize
    - Check time time_unit
"""

from importlib.resources import as_file, files
import numpy as np
import pandas as pd
import logging

# Locate resources bundled with the installed satpass package.
_DATA_ROOT = files("satpass.data")

def _read_data_csv(relative_path, **kwargs):
    """Read a CSV resource bundled with the satpass package."""
    resource = _DATA_ROOT.joinpath(*relative_path.split("/"))
    if not resource.is_file():
        raise FileNotFoundError(f"Missing satpass data file: {relative_path}")

    with as_file(resource) as path:
        return pd.read_csv(path, **kwargs)

# Define data types for metadata CSV columns
dtypes = {'short_name': str, 'full_name': str, 'start_date_cycle_min': str, 'end_of_mission': str,
          'repeatability': float, 'cycle_min': pd.Int64Dtype(), 'cycle_max': pd.Int64Dtype(),
          'nominal_track_filename': str}
parse_dates = ['start_date_cycle_min', 'end_of_mission']

# Read metadata CSV
meta = _read_data_csv(
    "altimeters.meta",
    sep=',',
    comment='#',
    skipinitialspace=True,
    dtype=dtypes,
    index_col=['short_name'],
)

for col in parse_dates:
    meta[col] = pd.to_datetime(meta[col])

# Define logger
logger = logging.getLogger(__name__)

# Define NominalTrack class
class NominalTrack(object):
    """
    Nominal Track object

    :param mission: satellite mission name (default is 'en', could be 'sw', 's3a', 'j2', ...)
    :type mission: str

    >>> j3 = NominalTrack(mission='j3')
    """

    def __init__(self, mission='en', **kwargs):
        # Set paths and attributes
        filename = meta.loc[mission].nominal_track_filename
        self.source_file = _DATA_ROOT.joinpath("nominal_tracks", filename)
        self.mission = mission
        self.meta = meta.loc[mission]

        # Read nominal track data from CSV
        if self.mission in ['s3a', 's3b', 'cfo']:
            self.nominal = _read_data_csv(
                f"nominal_tracks/{filename}",
                header=None,
                names=['track', 'lon', 'lat', 'time_c06'],
            )
        else:
            self.nominal = _read_data_csv(
                f"nominal_tracks/{filename}",
                header=None,
                names=['track', 'lon', 'lat'],
            )

    def select_in_area(self, area, extend_to_latitude_bounds=True):
        """
        Select nominal tracks intersecting a geographic area.

        :param area: given area (format: [lon_min, lon_max, lat_min, lat_max] in degree in [-180°:180°]/[-90°:90°])
        :type area: list
        :param extend_to_latitude_bounds: if True, identify tracks intersecting
            the full area and return their complete segments between the area's
            latitude bounds. If False, return only points inside the full area.
        :type extend_to_latitude_bounds: bool
        :returns: DataFrame with track number, lon, and lat
        """

        lon_min, lon_max, lat_min, lat_max = area
        df = self.nominal

        in_box = (
            (df.lon >= lon_min) & (df.lon <= lon_max) &
            (df.lat >= lat_min) & (df.lat <= lat_max)
        )

        if extend_to_latitude_bounds:
            # Identify tracks using the full box, then retain their complete
            # segments between the requested latitude bounds.
            intersecting_tracks = in_box.groupby(df.track).any()
            intersecting_tracks = intersecting_tracks[intersecting_tracks].index
            mask = (
                df.track.isin(intersecting_tracks) &
                (df.lat >= lat_min) & (df.lat <= lat_max)
            )
        else:
            mask = in_box

        tracks = df.loc[mask].copy()

        # Check longitudinal progression within each track.
        dlon = tracks.groupby("track")["lon"].diff()

        dlon = (dlon + 180) % 360 - 180
        mean_diff = dlon.groupby(tracks["track"]).mean()

        valid_tracks = mean_diff[mean_diff > 0].index
        tracks = tracks.loc[tracks.track.isin(valid_tracks)].copy()

        lon0_360 = tracks.groupby("track")["lon"].apply(circular_mean_lon_360)
        tracks["lon0_360"] = tracks["track"].map(lon0_360)

        # if not extend_to_latitude_bounds:
        #     return tracks

        # Extended selections also carry the properties used by pass and
        # subcycle calculations.
        tracks["subcycle"] = np.nan
        tracks["direction"] = pd.Series(index=tracks.index, dtype="object")

        if tracks.empty:
            return tracks

        direction = {}

        for track_number, track in tracks.groupby('track'):
            if len(track) > 1:
                if np.diff(track.lat)[0] > 0:
                    direction[track_number] = 'asc'
                else:
                    direction[track_number] = 'des'
            else:
                direction[track_number] = np.nan

        tracks["direction"] = tracks["track"].map(direction)

        if self.mission == 'sw':

            tracks = assign_subcycles_to_tracks(tracks)

        return tracks

    def create_tracks_dict(self, numbers, nominal_cut):
        """
        Create a dict with satellite_tracks_number & list of coordinates

        :param numbers: list of satellite tracks numbers
        :param nominal_cut: DataFrame with points on the given area

        :returns: dict {track_number: list of coordinates}
        """

        track_dict = {}

        for num in numbers:
            _dt = pd.Timedelta(self.meta["repeatability"], unit='D') / len(set(self.nominal.track)) / \
                  len(self.nominal.lon[self.nominal.track == num])

            _points = [(lon, lat, t) for lon, lat, t in zip(nominal_cut.lon[nominal_cut.track == num],
                                                            nominal_cut.lat[nominal_cut.track == num],
                                                            nominal_cut.lon[
                                                                nominal_cut.track == num].index * _dt)]
            track_dict[num] = _points

        return track_dict

    def extract(self, area, ocean=True):
        """
        Extract the tracks in a given area

        :param area: a area in the [lon_min, lon_max, lat_min, lat_max] in decimal degree [-180°,180°,-90.,90.])
        :type area: list
        :param ocean: set to True to select only the ocean tracks (default: True)
        :type ocean: bool
        :return: a dict with the track number as key and its list of coordinates in the area
        """
        selection = self.select_in_area(area=area)
        tracks_num = list(set(selection['track']))
        track_dict = self.create_tracks_dict(numbers=tracks_num, nominal_cut=selection)

        if ocean:
            return track_dict
        else:
            return track_dict

    def get_tracks(self, area):
        """
        For a given area, select tracks numbers and create dict with associated list of coordinates

        :param area: given area (format: [lon_min, lon_max, lat_min, lat_max] in degree in [-180°:180°]/[-90°:90°])

        :returns: dict {track_number: list of coordinates}
        """
        nominal_masked_cut = self.select_in_area(
            area=area,
            extend_to_latitude_bounds=True,
        )
        tracks_num = list(set(nominal_masked_cut['track']))
        _points = self.create_tracks_dict(numbers=tracks_num, nominal_cut=nominal_masked_cut)

        return tracks_num, _points
    
def circular_mean_lon_360(lon):
    lon = np.asarray(lon, dtype=float)

    # convert to radians
    ang = np.deg2rad(lon)

    # circular mean
    mean_ang = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    mean_lon = np.rad2deg(mean_ang)

    # convert to [0, 360)
    mean_lon_360 = mean_lon % 360
    return mean_lon_360

# Ephemeris computation
def ephemeris(nt, date, **kwargs):
    """
    Compute complete ephemeris depending on the NominalTrack object.

    :param nt: NominalTrack object
    :param date: list of 1/2 dates (e.g., ['2020-02-30', '2020-06-21'])
    :rtype: pandas.DatetimeIndex
    """
    date = pd.DatetimeIndex(date)

    if isinstance(nt.meta["end_of_mission"], pd.Timestamp):
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           end=nt.meta["end_of_mission"]).to_frame()
    elif isinstance(nt.meta["cycle_max"], np.int64):
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           periods=(nt.meta["cycle_max"] - nt.meta["cycle_min"] + 1)).to_frame()
    else:
        complete_ephemeris = pd.date_range(start=nt.meta["start_date_cycle_min"],
                                           freq=pd.Timedelta(nt.meta["repeatability"], unit='D'),
                                           end=date[-1]).to_frame()

    if len(date) == 1:
        selected_ephemeris = complete_ephemeris[date[0].strftime('%Y-%m'):date[0].strftime('%Y-%m')].index.tz_localize("UTC")
    elif len(date) == 2:
        selected_ephemeris = complete_ephemeris[(date[0] - pd.Timedelta(nt.meta["repeatability"], unit='D')):date[1]].index
    else:
        print('--- Error date interval format => return complete ephemeris')
        selected_ephemeris = complete_ephemeris

    return selected_ephemeris

# SatPass main launcher
# SatPass main launcher
# def sat_pass(mission, date, area=None, **kwargs):
#     """
#     Function to compute satellite ephemeris for a given mission and date interval.

#     :param mission: list of missions to consider
#     :param date: list of dates (e.g., [datetime(2020, 3, 2), datetime(2020, 6, 2)])
#     :param area: a 4-number list [xmin, xmax, ymin, ymax]
#     :return: DataFrame of the tracks and time in the area
#     """

#     if len(date)> 2:
#         date = [date[0], date[-1]]

#     if isinstance(mission, str):
#         nt = NominalTrack(mission=mission)
#     elif isinstance(mission, NominalTrack):
#         nt = mission

#     dct_eph = {}

#     eph = ephemeris(nt, date)

#     DF = []
    
#     if area:
#         n_tracks, dct = nt.get_tracks(area=area)

#         for n, trace in dct.items():
#             _x, _y, _t = np.array(trace).T

#             if len(_x) > 1:
#                 if np.diff(_y)[0] > 0:
#                     direction = 'asc'
#                 else:
#                     direction = 'des'

#                 _i = int(len(_x) / 2)
#                 _x, _y, _t = _x[_i], _y[_i], (_t[_i] + eph).round('60s')

#                 cycle = (_t - nt.meta['start_date_cycle_min']).total_seconds() / 86400 // nt.meta[
#                     'repeatability'] + nt.meta['cycle_min']

#                 df = pd.DataFrame(np.array([[n] * len(eph), [direction] * len(eph), cycle]).T, index=_t,
#                                 columns=['track', 'direction', 'cycle'])

#                 DF.append(df)
#     else:
#         n_tracks, dct, dct_masked = nt.get_tracks(area=None)

#     DF = pd.concat(DF).sort_index()
#     DF.index = pd.DatetimeIndex(DF.index)
#     DF = DF[date[0]:date[-1]]
#     DF.track = DF.track.astype(int)
#     DF['time'] = DF.index

#     if (mission == 'sw')|(mission == 'swi'):
#         DF = DF.rename(columns = dict(time = 't_swot'))

#     DF.cycle = DF.cycle.astype(float) 

#     return DF


# def assign_subcycles(swot_passes, domain, tol = 5):
    
#     swot_tracks = NominalTrack('sw').select_in_area(domain)

#     direction = []

#     for idx, df in swot_tracks.groupby('track'):
#         if len(df) > 1:
#             if np.diff(df.lat)[0] > 0:
#                 _direction = 'asc'
#             else:
#                 _direction = 'des'
#         else:
#             _direction = np.nan

#         direction.append(_direction)

#     direction = pd.Series(direction, index = np.unique(swot_tracks.track))

#     swot_tracks = swot_tracks.copy()
#     swot_tracks["subcycle"] = np.nan
#     lon0 = swot_tracks.groupby('track').median().lon

#     swot_tracks["lon0"] = swot_tracks["track"].map(lon0)
#     swot_tracks["direction"] = swot_tracks["track"].map(direction)

#     for i, direction in enumerate(["asc", "des"]):

#         DF = swot_tracks[swot_tracks.direction == direction].copy()

#         # Mean lon0 per track (compute ONCE)
#         g = DF.groupby("track")["lon0"].mean()     # index = track, values = lon0
#         trk = g.index.to_numpy()
#         lon = g.to_numpy()

#         # rotate so it starts at maximum lon0
#         i0 = int(np.argmax(lon))
#         lon_rot = np.hstack([lon[i0:], lon[:i0]])
#         trk_rot = np.hstack([trk[i0:], trk[:i0]])

#         # split into two subcycles (returns indices in lon_rot/trk_rot)
#         idx0, idx1 = disentangle_two_subcycles(lon_rot, tol=tol)

#         tracks_0 = trk_rot[idx0]
#         tracks_1 = trk_rot[idx1]

#         # assign: asc -> 0,1 ; des -> 0.5,1.5  (your scheme)
#         swot_tracks.loc[swot_tracks["track"].isin(tracks_0), "subcycle"] = 0 + 2*i
#         swot_tracks.loc[swot_tracks["track"].isin(tracks_1), "subcycle"] = 1 + 2*i

#     # return swot_tracks
#     track_to_subcycle = (
#     swot_tracks.dropna(subset=["subcycle"])
#               .drop_duplicates("track")
#               .set_index("track")["subcycle"]
# )

#     swot_passes = swot_passes.copy()
#     swot_passes["subcycle"] = swot_passes["track"].map(track_to_subcycle)

#     occ = swot_passes.groupby("track").cumcount()
#     swot_passes["subcycle_global"] = swot_passes["subcycle"] + 4 * occ

#     return swot_passes

# def assign_subcycles_old(df, threshold = pd.Timedelta(days = 2), remove_incomplete = True):

#     df = df.sort_values(by=['direction', 'time'])

#     df['time_diff'] = df.groupby('direction')['time'].diff()

#     def assign_subcycle(group):
#         subcycle_numbers = (group['time_diff'] > threshold).cumsum()
#         if group['direction'].iloc[0] == 'desc':
#             subcycle_numbers += subcycle_numbers.max() + 1
#         return subcycle_numbers

#     # Apply the custom function within each direction group
#     df['subcycle'] = df.groupby('direction', group_keys=False).apply(assign_subcycle)

#     # Drop the temporary time_diff column if needed
#     df = df.drop(columns=['time_diff'])
# #     subcycle[np.isnan(subcycle)] = 0.
#     # df.loc[:, 'subcycle'] = df.loc[:, 'subcycle']%2

#     if remove_incomplete:
#         df = remove_incomplete_subcycles(df)
#     df.index = df.time

#     df.cycle = df.cycle.astype(float)
#     return df

def sat_pass(mission, date, area=None, **kwargs):
    """
    Function to compute satellite ephemeris for a given mission and date interval.

    :param mission: list of missions to consider
    :param date: list of dates (e.g., [datetime(2020, 3, 2), datetime(2020, 6, 2)])
    :param area: a 4-number list [xmin, xmax, ymin, ymax]
    :return: DataFrame of the tracks and time in the area
    """

    if len(date)> 2:
        date = [date[0], date[-1]]

    if isinstance(mission, str):
        nt = NominalTrack(mission=mission)
    elif isinstance(mission, NominalTrack):
        nt = mission

    dct_eph = {}

    eph = ephemeris(nt, date)

    DF = []
    
    if area:
        n_tracks, dct = nt.get_tracks(area=area)

        for n, trace in dct.items():
            _x, _y, _t = np.array(trace).T
            lon0_360 = circular_mean_lon_360(_x)

            if len(_x) > 1:
                if np.diff(_y)[0] > 0:
                    direction = 'asc'
                else:
                    direction = 'des'

                _i = int(len(_x) / 2)
                _x, _y, _t = _x[_i], _y[_i], (_t[_i] + eph).round('60s')

                cycle = (_t - nt.meta['start_date_cycle_min']).total_seconds() / 86400 // nt.meta[
                    'repeatability'] + nt.meta['cycle_min']
                df = pd.DataFrame(np.array([[n] * len(eph), [direction] * len(eph), cycle, [lon0_360]* len(eph)]).T, index=_t,
                                columns=['track', 'direction', 'cycle', 'lon0_360'])

                DF.append(df)
    else:
        n_tracks, dct, dct_masked = nt.get_tracks(area=None)

    DF = pd.concat(DF).sort_index()
    DF.index = pd.DatetimeIndex(DF.index)
    DF = DF[date[0]:date[-1]]
    DF.track = DF.track.astype(int)
    DF['time'] = DF.index

    is_swot = nt.mission in ('sw', 'swi')
    
    DF.cycle = DF.cycle.astype(float) 

    DF.lon0_360 = DF.lon0_360.astype(float)

    if is_swot:
        DF = assign_subcycles_to_passes(DF, area)

    return DF

def disentangle_two_subcycles(lon, tol=5, tol_step=1):

    lon = np.asarray(lon, dtype=float)

    idx0 = []
    idx1 = []
    last0 = lon[0] + 1e-3

    for i in range(len(lon)):
        if lon[i] < last0:
        
            if np.abs(lon[i] - last0) < tol:

                
                if len(idx0) <= 1:
                    idx0.append(i)
                    last0 = lon[i]
                else:
                    if np.abs(np.abs(lon[i] - last0) - np.abs(lon[idx0[1]] - lon[idx0[0]])) < tol_step:
                        idx0.append(i)
                        last0 = lon[i]
                    else:
                        idx1.append(i)
            else:
                idx1.append(i)
        elif lon[i] > last0:
            idx1.append(i)

    idx0 = list(idx0)
    idx1 = list(idx1)

    # final sorting
    idx0 = np.array(sorted(idx0, key=lambda i: lon[i], reverse=True), dtype=int)
    idx1 = np.array(sorted(idx1, key=lambda i: lon[i], reverse=True), dtype=int)

    return idx0, idx1

def assign_subcycles_to_tracks(tracks, tol = 5, tol_step = 1):

    for i, direction in enumerate(["asc", "des"]):
    
        DF = tracks[tracks.direction == direction].copy()

        # Mean lon0 per track (compute ONCE)
        g = DF.groupby("track")["lon0_360"].mean()     # index = track, values = lon0
        trk = g.index.to_numpy()
        lon = g.to_numpy()

        # rotate so it starts at maximum lon0
        i0 = int(np.argmax(lon))
        lon_rot = np.hstack([lon[i0:], lon[:i0]])
        trk_rot = np.hstack([trk[i0:], trk[:i0]])

        # split into two subcycles (returns indices in lon_rot/trk_rot)
        # idx0, idx1 = satpass.disentangle_two_subcycles(lon_rot, tol=5)
        idx0, idx1 = disentangle_two_subcycles(lon_rot, tol=5, tol_step=1)

        tracks_0 = trk_rot[idx0]
        tracks_1 = trk_rot[idx1]

        lon0_0 = lon_rot[idx0]
        lon0_1 = lon_rot[idx1]

        tracks.loc[tracks["track"].isin(tracks_0), "subcycle"] = 0 + 2*i
        tracks.loc[tracks["track"].isin(tracks_1), "subcycle"] = 1 + 2*i

    return tracks

def assign_subcycle_global_from_rank(df):
    # df = df.sort_values("time").copy()
    df = df.copy()

    new_cycle = df["subcycle_rank"].diff() < 0
    new_cycle.iloc[0] = True

    cycle_number = new_cycle.cumsum() - 1
    df["subcycle_global"] = df["subcycle"] + 4 * cycle_number

    return df

def assign_subcycles_to_passes(swot_passes, domain, tol = 5, tol_step = 1):

    swot_tracks = NominalTrack('sw').select_in_area(
        domain,
        extend_to_latitude_bounds=True,
    )
    swot_tracks = assign_subcycles_to_tracks(swot_tracks, tol, tol_step)


    track_seq = (
        swot_tracks
        .dropna(subset=["subcycle", "lon0_360"])
        .groupby(["subcycle", "track"], as_index=False)["lon0_360"]
        .mean()
        .sort_values(["subcycle", "lon0_360"], ascending=[True, False])
    ).reset_index()

    track_seq["subcycle_rank"] = track_seq.groupby("subcycle").cumcount()

    track_rank_map = track_seq.set_index("track")["subcycle_rank"]
    subcycle_map = track_seq.set_index("track")["subcycle"]

    swot_passes["subcycle_rank"] = swot_passes["track"].map(track_rank_map)
    swot_passes["subcycle"] = swot_passes["track"].map(subcycle_map)
    # A decrease in rank marks the start of another occurrence of the same
    # subcycle.  Compute this directly instead of using GroupBy.apply: pandas
    # 3 excludes grouping columns from the frame passed to apply, which made
    # ``assign_subcycle_global_from_rank`` fail when reading ``subcycle``.
    grouped = swot_passes.groupby("subcycle", dropna=False)
    new_cycle = grouped["subcycle_rank"].diff().lt(0)
    new_cycle |= grouped.cumcount().eq(0)
    cycle_number = new_cycle.groupby(swot_passes["subcycle"], dropna=False).cumsum() - 1
    swot_passes["subcycle_global"] = (
        swot_passes["subcycle"] + 4 * cycle_number
    )

    return swot_passes

def remove_incomplete_subcycles(df):
        
    counts = df.groupby(['direction', 'cycle', 'subcycle']).size()
    max_counts = counts.groupby(['direction', 'subcycle']).transform('max')
    # Identify the combinations to remove
    combinations_to_remove = counts[counts != max_counts].index
    
    # Remove the identified combinations
    df = df.set_index(['direction', 'cycle', 'subcycle']).drop(combinations_to_remove).reset_index()
    return df
