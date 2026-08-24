import pandas as pd

def assign_filepath(satpass_swot, swot_files):
    """
    Match satpass output to existing SWOT NetCDF files.
    """

    rows = []

    for file in swot_files:
        parts = file.split('SWOT_')[-1].split('_')
        t0 = pd.Timestamp(parts[6])
        t1 = pd.Timestamp(parts[7])
        t = pd.DatetimeIndex([t0, t1]).mean()
        cycle = int(parts[4])
        track = int(parts[5])
        rows.append(pd.DataFrame({'cycle': [cycle], 'track': [track], 'filepath': [file]}, index=[t]))

    if not rows:
        return satpass_swot.assign(filepath=np.nan)

    file_table = pd.concat(rows).sort_index()
    file_table.index = file_table.index.round('60s')

    out = pd.merge(satpass_swot, file_table, how='left')
    out = out.sort_values('time')
    out.index = pd.DatetimeIndex(out.time.values)

    return out