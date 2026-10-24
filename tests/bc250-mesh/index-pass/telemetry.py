"""Read-only low-rate sysfs diagnostics; never writes GPU configuration."""
import json
from pathlib import Path
import threading
import time


def devices(root=Path('/sys/class/drm')):
    found = []
    for card in sorted(root.glob('card[0-9]*')):
        if not card.name[4:].isdigit():
            continue
        device = card / 'device'
        if not (device / 'gpu_busy_percent').exists():
            continue
        files = dict(busy_percent=device / 'gpu_busy_percent')
        for hwmon in sorted((device / 'hwmon').glob('hwmon*')):
            for name, field in (('freq1_input', 'clock_hz'), ('freq2_input', 'memory_clock_hz'),
                                ('temp1_input', 'temperature_millidegrees')):
                if (hwmon / name).exists():
                    files[field] = hwmon / name
        found.append((card.name, files))
    return found


def snapshot(found):
    rows = []
    for card, files in found:
        values, errors = {}, []
        for key, path in files.items():
            try:
                values[key] = int(path.read_text().strip())
            except (OSError, ValueError):
                errors.append(key)
        rows.append(dict(card=card, values=values, unavailable=errors))
    return dict(monotonic_ns=time.monotonic_ns(), cards=rows)


class Recorder:
    def __init__(self, path, found=None):
        self.path = path
        self.found = devices() if found is None else found
        self.stop_event = threading.Event()
        self.errors = []

    def __enter__(self):
        self.file = self.path.open('x')
        self.write()
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        return self

    def write(self):
        self.file.write(json.dumps(snapshot(self.found)) + '\n')
        self.file.flush()

    def collect(self):
        try:
            while not self.stop_event.wait(.05):
                self.write()
        except Exception as exc:
            self.errors.append(type(exc).__name__)

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        try:
            self.write()
        finally:
            self.file.close()
        if self.errors and args[0] is None:
            raise RuntimeError('telemetry recording failed: ' + str(self.errors))


def summarize(path, begin, end):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows and all(a['monotonic_ns'] <= b['monotonic_ns'] for a, b in zip(rows, rows[1:]))
    selected = [r for r in rows if begin <= r['monotonic_ns'] <= end]
    clocks, busy = [], []
    for row in selected:
        for card in row['cards']:
            values = card['values']
            if 'clock_hz' in values:
                clocks.append(values['clock_hz'])
            if 'busy_percent' in values:
                busy.append(values['busy_percent'])
    return dict(samples_in_window=len(selected), clock_hz_range=[min(clocks), max(clocks)] if clocks else None,
                busy_percent_range=[min(busy), max(busy)] if busy else None,
                interpretation='50 ms read-only snapshots; short measurement windows may have no sample; '
                               'this does not prove clocks stayed constant between samples')
