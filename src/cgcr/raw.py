"""Stream the supplied ABC2026 ZIP into documented, second-level wide tables.

MAC-to-number assignment is external: the archive has no such mapping. The
numbered columns are only valid after an authoritative mapping is supplied.
"""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import re
import zipfile
from zoneinfo import ZoneInfo


LABEL_NAME = "Dataset/5f_label_loc_train.csv"
BLE_PREFIX = "Dataset/BLE Data/"
MAC_RE = re.compile(r"^(?:[0-9A-F]{2}:){5}[0-9A-F]{2}$")
FILE_DAY_RE = re.compile(r"user-ble-id_90_(\d{4}-\d{2}-\d{2})T")
EXPERIMENT_START = datetime.fromisoformat("2023-04-10T13:00:00")
EXPERIMENT_END = datetime.fromisoformat("2023-04-13T17:29:59.999")
TOKYO = ZoneInfo("Asia/Tokyo")


def _ble_members(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    groups: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for member in archive.infolist():
        if not (member.filename.startswith(BLE_PREFIX) and member.filename.endswith(".csv")):
            continue
        match = FILE_DAY_RE.search(member.filename)
        if match:
            groups[match.group(1)].append(member)
    return groups


def inspect_macs(archive_path: Path, report_path: Path) -> int:
    """Inventory observed MACs, keeping data in the ZIP and streaming each file."""
    counts: Counter[str] = Counter()
    days: dict[str, Counter[str]] = defaultdict(Counter)
    with zipfile.ZipFile(archive_path) as archive:
        for day, members in _ble_members(archive).items():
            for member in members:
                with archive.open(member) as raw:
                    for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
                        if len(row) != 6:
                            raise ValueError(f"Malformed BLE row in {member.filename}: {row[:2]}")
                        mac = row[3].upper()
                        if MAC_RE.fullmatch(mac):
                            counts[mac] += 1
                            days[mac][day] += 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        day_columns = sorted({day for tally in days.values() for day in tally})
        writer.writerow(["mac", "observations", *day_columns])
        for mac, count in counts.most_common():
            writer.writerow([mac, count, *(days[mac][day] for day in day_columns)])
    return len(counts)


def load_beacon_map(path: Path) -> dict[str, int]:
    """Read a JSON object mapping MAC addresses to physical beacon IDs 1..25."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Beacon map must be a nonempty JSON object")
    result: dict[str, int] = {}
    for address, beacon in raw.items():
        mac = address.upper()
        if not MAC_RE.fullmatch(mac) or type(beacon) is not int or not 1 <= beacon <= 25:
            raise ValueError(f"Invalid beacon map entry: {address!r}: {beacon!r}")
        if mac in result:
            raise ValueError(f"Duplicate MAC: {mac}")
        result[mac] = beacon
    if len(set(result.values())) != len(result):
        raise ValueError("Each physical beacon ID must map to exactly one MAC")
    return result


def _labels(archive: zipfile.ZipFile) -> tuple[dict[datetime, str | None], int]:
    """Expand clean user-97 label intervals to complete, unambiguous seconds."""
    seconds: dict[datetime, str | None] = {}
    eligible = 0
    with archive.open(LABEL_NAME) as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
            if (row["user_id"] != "97" or row.get("activity") != "Location"
                    or row.get("deleted_at") or not row["room"]
                    or not row["started_at"] or not row["finished_at"]):
                continue
            start = datetime.fromisoformat(row["started_at"]).astimezone(TOKYO).replace(tzinfo=None)
            end = datetime.fromisoformat(row["finished_at"]).astimezone(TOKYO).replace(tzinfo=None)
            if end <= start:
                continue
            eligible += 1
            tick = start.replace(microsecond=0)
            if tick < start:
                tick += timedelta(seconds=1)
            while tick + timedelta(seconds=1) <= end:
                if tick not in seconds:
                    seconds[tick] = row["room"]
                elif seconds[tick] != row["room"]:
                    seconds[tick] = None
                tick += timedelta(seconds=1)
    return seconds, eligible


def prepare_wide(archive_path: Path, beacon_map_path: Path, output_dir: Path) -> dict[str, int]:
    """Aggregate per-second max RSSI and attach only conflict-free labels.

    All timestamps remain local Japan time without timezone suffix to match the
    historical notebook's naive timestamps. Unobserved beacons are zero.
    """
    beacon_map = load_beacon_map(beacon_map_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"rssi_{i}" for i in range(1, 26)]
    stats = Counter()
    stats["mapped_beacons"] = len(beacon_map)
    with zipfile.ZipFile(archive_path) as archive:
        labels, stats["label_intervals"] = _labels(archive)
        with (output_dir / "labeled_wide.csv").open("w", newline="", encoding="utf-8") as labeled_out, \
             (output_dir / "unlabeled_wide.csv").open("w", newline="", encoding="utf-8") as unlabeled_out:
            labeled_writer = csv.writer(labeled_out)
            unlabeled_writer = csv.writer(unlabeled_out)
            labeled_writer.writerow(["timestamp", "room", *columns])
            unlabeled_writer.writerow(["timestamp", *columns])
            for day, members in sorted(_ble_members(archive).items()):
                # At most one day's matched observations are held in memory.
                aggregate: dict[datetime, dict[int, int]] = defaultdict(dict)
                for member in members:
                    with archive.open(member) as raw:
                        for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
                            if len(row) != 6:
                                raise ValueError(f"Malformed BLE row in {member.filename}")
                            stats["ble_rows"] += 1
                            if row[0] != "90":
                                continue
                            beacon = beacon_map.get(row[3].upper())
                            if beacon is None:
                                continue
                            timestamp = datetime.fromisoformat(row[1]).astimezone(TOKYO)
                            timestamp = timestamp.replace(microsecond=0, tzinfo=None)
                            if timestamp.date().isoformat() != day:
                                raise ValueError(f"BLE timestamp/file date mismatch: {member.filename}")
                            if not EXPERIMENT_START <= timestamp <= EXPERIMENT_END:
                                continue
                            try:
                                rssi = int(row[4])
                            except ValueError as exc:
                                raise ValueError(f"Invalid RSSI in {member.filename}") from exc
                            if not -120 <= rssi <= 0:
                                stats["invalid_rssi_rows"] += 1
                                continue
                            stats["mapped_ble_rows"] += 1
                            old = aggregate[timestamp].get(beacon, -120)
                            aggregate[timestamp][beacon] = max(old, rssi)
                for timestamp, signals in sorted(aggregate.items()):
                    values = [signals.get(i, 0) for i in range(1, 26)]
                    label = labels.get(timestamp)
                    if label is None:
                        unlabeled_writer.writerow([timestamp.isoformat(sep=" "), *values])
                        stats["unlabeled_seconds"] += 1
                    else:
                        labeled_writer.writerow([timestamp.isoformat(sep=" "), label, *values])
                        stats["labeled_seconds"] += 1
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(dict(stats), indent=2) + "\n", encoding="utf-8"
    )
    return dict(stats)
