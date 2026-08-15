"""
Shared helper functions for GTFS scenario construction.
"""

from pathlib import Path
import ast
import zipfile
import pandas as pd
import numpy as np


def to_sec(t):
    """Returns seconds after midnight for a given HH:MM:SS string."""

    h, m, s = t.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def to_gtfs_time(sec):
    """Returns HH:MM:SS string from seconds after midnight."""

    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def compress_trip(group):
    """Compresses inter-stop journey time for one trip's stop_times by 15%."""

    group = group.sort_values('stop_sequence').copy()
    dep = group['dep_sec'].tolist()

    new_dep = dep.copy()

    for i in range(1, len(group)):
        inter_stop = dep[i] - dep[i - 1]
        new_inter = inter_stop * 0.85
        new_dep[i] = new_dep[i - 1] + new_inter

    group['alt_dep_time'] = new_dep
    group['alt_dep_time'] = group['alt_dep_time'].apply(to_gtfs_time)
    return group


def median_segment_times(pattern_stoptimes):
    """Returns the median stop-to-stop duration (seconds) at each position
    in a trip pattern, computed across all of its morning-peak trips."""

    df = pattern_stoptimes.sort_values(['trip_id', 'stop_sequence']).copy()
    df['pos'] = df.groupby('trip_id').cumcount()
    pivot = df.pivot(index='trip_id', columns='pos', values='dep_sec')
    segment_durations = pivot.diff(axis=1).iloc[:, 1:]
    return segment_durations.median(axis=0).to_numpy()


def select_template_trip_id(pattern_stoptimes):
    """Returns the trip_id within a pattern's stop_times slice that has the
    MOST stop_times rows recorded that day --> the most complete
    observed coverage of the pattern's full stop sequence.

    Using any arbitrary trip (whichever trip happens to be "trip_a" in a
    gap-injection loop) as the template instead causes a length mismatch
    in build_synthetic_stoptimes whenever that trip has fewer recorded
    stops than the pattern's most complete trip - which happens whenever
    GPS proximity-matching misses a stop for some but not all trips
    sharing a pattern.
    """

    return pattern_stoptimes.groupby('trip_id').size().idxmax()


def build_synthetic_stoptimes(trip_a_stoptimes, median_segments, start_sec, synth_id):
    """Builds a synthetic trip's stop_times rows off a pattern's median
    segment durations, starting at start_sec."""

    st = trip_a_stoptimes.sort_values('stop_sequence').copy()
    cum_offsets = np.concatenate(([0], np.cumsum(median_segments)))
    new_times = [to_gtfs_time(start_sec + off) for off in cum_offsets]
    st['trip_id'] = synth_id
    st['arrival_time'] = new_times
    st['departure_time'] = new_times
    return st


def filter_bee_network_patterns(patterns_df, city):
    """Manchester-only helper. Restricts a patterns DataFrame to rows where 
    agency_name == 'Bee Network'. No-op for any other city."""

    if city != 'Manchester':
        return patterns_df

    filtered = patterns_df[patterns_df['agency_name'] == 'Bee Network']
    if filtered.empty and not patterns_df.empty:
        raise ValueError(
            "Filtering morning_patterns.csv to agency_name == 'Bee Network' "
            f"returned zero patterns. Available agency_name values: "
            f"{patterns_df['agency_name'].unique().tolist()}. "
            "Update the exact-match string in filter_bee_network_patterns to match."
        )
    return filtered


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_morning_patterns(post_processed_root, city):
    """Reads {city}_retro_morning_patterns.csv, restores the `pattern` and
    `id_list` columns from their stringified list form, and applies the
    Manchester-only Bee Network restriction when applicable."""

    patterns = pd.read_csv(post_processed_root / f'{city}_retro_morning_patterns.csv')
    patterns['pattern'] = patterns['pattern'].apply(ast.literal_eval)
    patterns['id_list'] = patterns['id_list'].apply(ast.literal_eval)
    return filter_bee_network_patterns(patterns, city)


def load_baseline_gtfs(baseline_root, city, target_date):
    """Reads one city/date's retrospective baseline GTFS stop_times.txt and
    trips.txt. Adds a `dep_sec` column (seconds-after-midnight departure) to
    stop_times, used throughout the speed-up and injection logic.

    Returns (stop_times, trips, source_dir) - source_dir is the folder the
    unchanged GTFS text files (agency/calendar/feed_info/routes/shapes/stops)
    get copied from when the scenario zip is written."""
    
    source_dir = baseline_root / city / target_date
    stop_times = pd.read_csv(source_dir / 'stop_times.txt').sort_values('stop_sequence')
    stop_times['dep_sec'] = stop_times['departure_time'].apply(to_sec)
    trips = pd.read_csv(source_dir / 'trips.txt')
    return stop_times, trips, source_dir


# ---------------------------------------------------------------------------
# Pattern filters
# ---------------------------------------------------------------------------

def make_pattern_filter(frequency_values, area=None):
    """Builds a boolean-mask function over morning_patterns rows.

    frequency_values: which `frequent?` values are eligible.
        - ['frequent', 'not_frequent']  -> speed-up stage eligibility (2a/2b/4a/4b)
        - ['not_frequent']              -> injection stage eligibility (3a/3b/4a/4b)

    area: None for the entire-study-area "a" scenarios, or 
    area: 'inbound'/'deprived' to further restrict to that "b" scenario area flag"""

    def _filter(patterns):
        mask = patterns['frequent?'].isin(frequency_values)
        if area is not None:
            mask = mask & (patterns[f'{area}?'] == area)
        return mask
    return _filter


# ---------------------------------------------------------------------------
# Core scenario logic
# ---------------------------------------------------------------------------

def build_scenario_output(patterns, stop_times, trips, speedup_filter=None, inject_filter=None):
    """Applies an optional speed-up stage (original Scenario 1 logic: compress
    inter-stop times by 15% for trips in speedup_filter-eligible patterns)
    followed by an optional injection stage (original Scenario 2 logic:
    synthetic duplicate trips into the gaps of inject_filter-eligible
    patterns) to one city/date/suffix combination.

    Passing only one filter reproduces 1a/1b (speedup_filter only) or 2a/2b
    (inject_filter only). Passing both reproduces 3a/3b.

    Returns (new_trips, new_stoptimes, trips_modified, trips_injected,
    patterns_injected) ready for write_gtfs_zip(). trips_injected and
    patterns_injected are None when inject_filter is None."""

    stop_times = stop_times.copy()
    if 'dep_sec' not in stop_times.columns:
        stop_times['dep_sec'] = stop_times['departure_time'].apply(to_sec)

    # ===== Stage 1: speed-up (Scenario 1 logic) =====
    if speedup_filter is not None:
        speedup_patterns = patterns[speedup_filter(patterns)]
        speedup_trips = {
            trip_id
            for id_list in speedup_patterns['id_list']
            for trip_id in id_list
        }
        peak_stmask = stop_times['trip_id'].isin(speedup_trips)

        st_spedup = stop_times[peak_stmask].groupby(
            'trip_id', group_keys=False
        ).apply(compress_trip)

        # true count of trips actually modified for THIS date/suffix
        trips_modified = st_spedup['trip_id'].nunique()

        st_spedup['arrival_time'] = st_spedup['alt_dep_time']
        st_spedup['departure_time'] = st_spedup['alt_dep_time']
        st_spedup = st_spedup.drop(columns=['alt_dep_time'])

        st_remain = stop_times[~peak_stmask]

        amended_stoptimes = pd.concat([st_spedup, st_remain]).sort_index()
        # recompute dep_sec off the (possibly now-compressed) departure_time -
        # needed as the injection stage's gap-timing basis when both stages run
        amended_stoptimes['dep_sec'] = amended_stoptimes['departure_time'].apply(to_sec)
    else:
        trips_modified = 0
        amended_stoptimes = stop_times

    # ===== Stage 2: injection (Scenario 2 logic) =====
    if inject_filter is not None:
        eligible_patterns = patterns[inject_filter(patterns)]

        first_stop_deps = amended_stoptimes.groupby(
            'trip_id'
        ).first().reset_index()[['trip_id', 'dep_sec']]
        trips_dated = trips.merge(first_stop_deps, on='trip_id', how='left')

        synth_trips = []
        synth_stoptimes = []

        for _, pattern_row in eligible_patterns.iterrows():

            pattern_trip_ids = pattern_row['id_list']

            group_sorted = trips_dated[
                trips_dated['trip_id'].isin(pattern_trip_ids)
            ].sort_values('dep_sec').reset_index(drop=True)

            seg_stoptimes = amended_stoptimes[amended_stoptimes['trip_id'].isin(pattern_trip_ids)]
            median_segments = median_segment_times(seg_stoptimes)

            # structural template for synthetic rows: the most-complete trip in
            # this pattern today, sized to match median_segments (see
            # select_template_trip_id's docstring for why)
            if len(group_sorted) >= 2:
                template_trip_id = select_template_trip_id(seg_stoptimes)
                template_st = amended_stoptimes[amended_stoptimes['trip_id'] == template_trip_id]

            for i in range(len(group_sorted) - 1):
                trip_a_id = group_sorted.loc[i, 'trip_id']
                dep_a = group_sorted.loc[i, 'dep_sec']
                dep_b = group_sorted.loc[i + 1, 'dep_sec']
                gap = dep_b - dep_a
                start_sec = dep_a + round(gap / 2)

                synth_id = f"{trip_a_id}_S{i}"

                orig_trip = trips_dated[trips_dated['trip_id'] == trip_a_id].iloc[0].to_dict()
                orig_trip['trip_id'] = synth_id
                synth_trips.append(orig_trip)

                synth_stoptimes.append(
                    build_synthetic_stoptimes(template_st, median_segments, start_sec, synth_id)
                )

        synthtrips_df = pd.DataFrame(synth_trips) if synth_trips else pd.DataFrame(columns=trips_dated.columns)
        new_trips = pd.concat([trips_dated, synthtrips_df], ignore_index=True).drop(columns=['dep_sec'])

        synthst_df = pd.concat(synth_stoptimes, ignore_index=True) if synth_stoptimes else pd.DataFrame(columns=amended_stoptimes.columns)
        new_stoptimes = pd.concat([amended_stoptimes, synthst_df], ignore_index=True).drop(columns=['dep_sec'])

        trips_injected = len(synthtrips_df)
        patterns_injected = len(eligible_patterns)
    else:
        new_trips = trips.copy()
        new_stoptimes = amended_stoptimes.drop(columns=['dep_sec'])
        trips_injected = None
        patterns_injected = None

    return new_trips, new_stoptimes, trips_modified, trips_injected, patterns_injected


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_gtfs_zip(output_path, source_gtfs_dir, trips_df, stoptimes_df):
    """Writes one scenario GTFS zip. trips.txt and stop_times.txt are written
    from the DataFrames passed in kwargs; the remaining GTFS
    text files are copied through unchanged from source_gtfs_dir (that city/
    date's baseline retrospective GTFS)."""

    unchanged_files = [
        'agency.txt', 'calendar.txt', 'feed_info.txt',
        'routes.txt', 'shapes.txt', 'stops.txt'
    ]
    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for file in unchanged_files:
            z.write(Path(source_gtfs_dir) / file, arcname=file)
        for df, fname in [(trips_df, 'trips.txt'), (stoptimes_df, 'stop_times.txt')]:
            z.writestr(fname, df.to_csv(index=False))


def summarize_scenario_run(city, target_date, suffix, speedup_active, inject_active,
                            trips_modified, trips_injected, patterns_injected):
    """Builds the console summary line for one (scenario, city, date, suffix)
    run."""

    tag = f"{city} {target_date}" + (f" {suffix}" if suffix else "")
    pattern_desc = "not-frequent" + (f"+{suffix}" if suffix else "")

    if speedup_active and inject_active:
        return (
            f"{tag}: compounded GTFS generated - {trips_modified} trip records "
            f"sped up on this date plus {trips_injected} trips injected across "
            f"{patterns_injected} {pattern_desc} patterns!"
        )
    elif speedup_active:
        return (
            f"{tag}: reduced-journey GTFS generated - {trips_modified} trip "
            f"records sped up on this date "
        )
    else:
        return (
            f"{tag}: more-frequent GTFS generated, injecting {trips_injected} "
            f"trips across {patterns_injected} {pattern_desc} trip patterns!"
        )
