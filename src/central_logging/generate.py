"""Generate a plant alarm log in the export format of the original project.

The plant data is under NDA, so the repository builds its own log: 75 PLCs on
three lines, background faults, error cascades that travel to neighbouring
stations, and devices that repeat one message in quick succession. The
generator knows which alerts belong to which cascade and writes that to a
separate file. The pipeline never reads it; only the scoring does.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .vocab import FAMILIES, FAMILY, LINES, ROLE_FAMILIES, ROLES

EXPORT_COLUMNS = [
    "alertId", "alertBegin", "alertEnd", "alertTotalDuration", "alertIsBandStop",
    "alertIsTpm", "alertCategory", "alertText", "alertPlc",
]

DEFAULTS = dict(
    n_plcs=75,
    days=14,
    start="2024-04-08",          # a Monday, no clock change in the window
    utc_offset_hours=2,
    seed=42,
    background_per_day=11_000,   # ordinary alerts on a production day
    cascades_per_day=480,
    chain_length=(3, 12),        # alerts per cascade, inclusive
    link_delay_ms=(5, 250),      # gap between one alert of a cascade and the next
    same_station=0.55,           # chance the next alert stays on the same PLC
    downstream=0.75,             # otherwise: chance it moves downstream
    bursts_per_day=120,          # a device repeating one message
    burst_length=(4, 30),
    burst_interval_ms=(20, 400),
    weekend_factor=0.08,
    zipf=1.1,
    tpm_rate=1e-4,
)

BREAKS = [(9.0, 9.25), (11.5, 12.0), (17.0, 17.25), (19.5, 20.0), (1.0, 1.25), (3.5, 4.0)]


def build_plant(n_plcs, rng, zipf):
    per_line = int(np.ceil(n_plcs / len(LINES)))
    rows, counters = [], {}
    for i in range(n_plcs):
        line, pos = LINES[i // per_line], i % per_line
        role, section = ROLES[pos % len(ROLES)]
        k = counters.get((line, role), 0) + 1
        counters[(line, role)] = k
        rows.append(dict(alertPlc=f"{line}{role}{k:02d}", line=line, position=pos, role=role, section=section))
    plant = pd.DataFrame(rows)
    weights = 1.0 / np.arange(1, n_plcs + 1) ** zipf
    plant["volume_share"] = (weights / weights.sum())[rng.permutation(n_plcs)]
    return plant


def build_inventory(plant, rng):
    """Each station gets a fixed set of messages per fault family."""
    inventory = {}
    for plc, role in zip(plant.alertPlc, plant.role):
        per_family = {}
        for key in ROLE_FAMILIES[role]:
            fam = FAMILY[key]
            messages = set()
            while len(messages) < int(rng.integers(3, 7)):
                device, template = fam.messages[int(rng.integers(len(fam.messages)))]
                messages.add(f"{device}{int(rng.integers(1, 10))} {template.format(n=int(rng.integers(1, 21)))}")
            msgs = sorted(messages)
            w = rng.dirichlet(np.ones(len(msgs)) * 0.8)
            per_family[key] = (msgs, w)
        inventory[plc] = per_family
    return inventory


def minute_weights(start, days, weekend_factor):
    idx = pd.date_range(start, periods=days * 1440, freq="min")
    hour = idx.hour.values + idx.minute.values / 60
    dow = idx.dayofweek.values
    production = ((dow <= 4) & (hour >= 6)) | ((dow >= 1) & (dow <= 5) & (hour < 6))
    shift = np.where((hour >= 6) & (hour < 14), 1.0, np.where((hour >= 14) & (hour < 22), 0.95, 0.7))
    in_break = np.zeros(len(idx), dtype=bool)
    for a, b in BREAKS:
        in_break |= (hour >= a) & (hour < b)
    w = np.where(production, shift * np.where(in_break, 0.3, 1.0), weekend_factor)
    one_day = w[1440:2880].sum()           # Tuesday, a full production day
    return idx, w, w.sum() / one_day


def sample_times(n, idx, w, rng):
    minutes = rng.choice(len(idx), size=n, p=w / w.sum())
    base = idx.values.astype("datetime64[ns]").astype(np.int64)[minutes]
    return base + rng.integers(0, 60_000_000_000, size=n)


def generate(**overrides):
    cfg = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(cfg["seed"])
    plant = build_plant(cfg["n_plcs"], rng, cfg["zipf"])
    inventory = build_inventory(plant, rng)
    idx, w, day_equiv = minute_weights(cfg["start"], cfg["days"], cfg["weekend_factor"])
    plcs = plant.alertPlc.values
    share = plant.volume_share.values
    role_of = dict(zip(plant.alertPlc, plant.role))
    lines = {line: g.sort_values("position").alertPlc.tolist() for line, g in plant.groupby("line")}
    where = {plc: (line, i) for line, members in lines.items() for i, plc in enumerate(members)}

    def pick_family(plc, preferred=None):
        fams = ROLE_FAMILIES[role_of[plc]]
        if preferred:
            options = [f for f in preferred if f in fams]
            if options:
                return str(rng.choice(options))
        keys = list(fams)
        p = np.array([fams[k] for k in keys])
        return str(rng.choice(keys, p=p / p.sum()))

    def pick_message(plc, family):
        msgs, mw = inventory[plc][family]
        return msgs[rng.choice(len(msgs), p=mw)]

    rows = []

    def add(t, plc, family, text, band_p, cascade=-1, role="none", burst=-1, dur=None):
        if dur is None:
            dur = float(np.clip(rng.lognormal(0.6, 2.0), 0.05, 86_400))
        rows.append((t, int(dur * 1e9), plc, family, text, int(rng.random() < band_p), cascade, role, burst))

    # Background alerts
    n_bg = int(round(cfg["background_per_day"] * day_equiv))
    for t, plc in zip(sample_times(n_bg, idx, w, rng), rng.choice(plcs, size=n_bg, p=share)):
        fam = pick_family(plc)
        add(int(t), plc, fam, pick_message(plc, fam), 0.03 + 0.15 * FAMILY[fam].severity)

    # Error cascades: a root fault, then consequent faults on the same or a neighbouring PLC
    n_casc = int(round(cfg["cascades_per_day"] * day_equiv))
    lo, hi = cfg["chain_length"]
    d_lo, d_hi = cfg["link_delay_ms"]
    for cid, (t, plc) in enumerate(zip(sample_times(n_casc, idx, w, rng),
                                       rng.choice(plcs, size=n_casc, p=share))):
        t = int(t)
        fam = pick_family(plc, [f.key for f in FAMILIES if f.triggers])
        add(t, plc, fam, pick_message(plc, fam), 0.3 + 0.5 * FAMILY[fam].severity, cid, "root",
            dur=float(np.clip(rng.lognormal(2.5, 1.5), 1, 86_400)))
        for _ in range(int(rng.integers(lo, hi + 1)) - 1):
            t += int(rng.uniform(d_lo, d_hi) * 1e6)
            if rng.random() >= cfg["same_station"]:
                line, i = where[plc]
                step = int(rng.integers(1, 3)) * (1 if rng.random() < cfg["downstream"] else -1)
                plc = lines[line][int(np.clip(i + step, 0, len(lines[line]) - 1))]
            triggers = FAMILY[fam].triggers
            fam = pick_family(plc, list(triggers) if triggers and rng.random() < 0.7 else [fam])
            add(t, plc, fam, pick_message(plc, fam), 0.1, cid, "link")

    # Repeating devices: one message fired again and again within seconds
    n_burst = int(round(cfg["bursts_per_day"] * day_equiv))
    b_lo, b_hi = cfg["burst_length"]
    i_lo, i_hi = cfg["burst_interval_ms"]
    for bid, (t, plc) in enumerate(zip(sample_times(n_burst, idx, w, rng),
                                       rng.choice(plcs, size=n_burst, p=share))):
        t = int(t)
        fam = pick_family(plc)
        text = pick_message(plc, fam)
        gap = rng.uniform(i_lo, i_hi)
        for _ in range(int(rng.integers(b_lo, b_hi + 1))):
            add(t, plc, fam, text, 0.02, role="repeat", burst=bid,
                dur=float(np.clip(rng.lognormal(-0.5, 0.8), 0.02, 60)))
            t += int(gap * rng.uniform(0.8, 1.2) * 1e6)

    df = pd.DataFrame(rows, columns=["begin_ns", "dur_ns", "alertPlc", "family", "alertText",
                                     "alertIsBandStop", "cascade_id", "cascade_role", "burst_id"])
    df = df.sort_values("begin_ns", kind="stable").reset_index(drop=True)
    df["alertId"] = np.arange(10_000_001, 10_000_001 + len(df))
    df["alertCategory"] = df.family.map({f.key: f.category for f in FAMILIES})
    df["alertIsTpm"] = (rng.random(len(df)) < cfg["tpm_rate"]).astype(int)

    suffix = f" +{cfg['utc_offset_hours']:02d}:00"            # local plant time
    begin = pd.to_datetime(df.begin_ns, unit="ns")
    df["alertBegin"] = _fmt(begin, suffix)
    df["alertEnd"] = _fmt(begin + pd.to_timedelta(df.dur_ns, unit="ns"), suffix)
    secs = (df.dur_ns // 1_000_000_000).astype(int)
    df["alertTotalDuration"] = (secs // 3600).map("{:02d}".format) + ":" + \
        (secs % 3600 // 60).map("{:02d}".format) + ":" + (secs % 60).map("{:02d}".format)

    alerts = df[EXPORT_COLUMNS].copy()
    truth = df[["alertId", "alertPlc", "cascade_id", "cascade_role", "burst_id", "family"]].copy()
    return alerts, truth, plant.drop(columns="volume_share")


def _fmt(ts, suffix):
    frac = (ts.dt.nanosecond.values // 100 + ts.dt.microsecond.values * 10)
    return ts.dt.strftime("%Y-%m-%d %H:%M:%S") + "." + pd.Series(frac, index=ts.index).map("{:07d}".format) + suffix


def write_dataset(out_dir, **overrides):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    alerts, truth, plant = generate(**overrides)
    alerts.to_csv(out / "alerts.csv", index=False)
    truth.to_csv(out / "ground_truth.csv", index=False)
    plant.to_csv(out / "plant.csv", index=False)
    return alerts, truth, plant


def alert_day(alerts):
    return alerts.alertBegin.str.slice(0, 10)


def export_daily_files(alerts, day, single_dir=None, multi_dir=None):
    """Write one day's export: one file per PLC, and/or one plant-wide file."""
    day_df = alerts[alert_day(alerts) == day]
    tag = day.replace("-", "")
    written = []
    if single_dir is not None:
        Path(single_dir).mkdir(parents=True, exist_ok=True)
        for plc, g in day_df.groupby("alertPlc"):
            path = Path(single_dir) / f"{plc}_{tag}.csv"
            g.to_csv(path, index=False)
            written.append(path)
    if multi_dir is not None:
        Path(multi_dir).mkdir(parents=True, exist_ok=True)
        path = Path(multi_dir) / f"Multi_PLC_{tag}.csv"
        day_df.to_csv(path, index=False)
        written.append(path)
    return written
