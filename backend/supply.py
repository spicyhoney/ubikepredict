"""Fixed-time, observed station inventory for supply recommendations.

Contains no future outcome or demand forecast. Packaged in the private Lambda
image; only neighborhoods of the current prediction are returned to clients.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math

SNAPSHOT_TIME = '2026-06-29T09:00:00+08:00'

@lru_cache(maxsize=1)
def _snapshot():
    path = Path(__file__).with_name('supply_snapshot.json')
    raw = path.read_bytes()
    expected = path.with_suffix('.sha256').read_text().strip()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError('供車快照完整性驗證失敗')
    data = json.loads(raw)
    assert data['decision_time'] == SNAPSHOT_TIME
    stations = data['stations']
    assert len(stations) == 1562 and len({s['station_id'] for s in stations}) == len(stations)
    for s in stations:
        assert all(isinstance(s[k], int) and s[k] >= 0 for k in ('bikes', 'docks', 'capacity'))
        assert s['bikes'] + s['docks'] <= s['capacity']
        assert math.isfinite(s['latitude']) and math.isfinite(s['longitude'])
    return stations

def _distance(a, b):
    x, y = math.radians(a['latitude']), math.radians(b['latitude'])
    z = math.radians(b['longitude'] - a['longitude'])
    h = math.sin((y-x)/2)**2 + math.cos(x)*math.cos(y)*math.sin(z/2)**2
    return 6371.0088 * 2 * math.asin(math.sqrt(min(1, max(0, h))))

def supply_context(decision_time, mode, target_ids):
    if decision_time != SNAPSHOT_TIME or mode != 'empty':
        return None
    stations = _snapshot()
    by_id = {s['station_id']: s for s in stations}
    included = {}; neighborhoods = {}
    for station_id in target_ids:
        target = by_id.get(station_id)
        if not target or target['bikes'] != 0 or target['docks'] <= 0:
            continue
        included[station_id] = target
        nearest = sorted(((_distance(target, s), s['station_id']) for s in stations if s['station_id'] != station_id))[:20]
        neighborhoods[str(station_id)] = [sid for _, sid in nearest]
        for _, sid in nearest:
            included[sid] = by_id[sid]
    return {'decision_time': SNAPSHOT_TIME, 'snapshot_station_count': len(stations),
            'stations': list(included.values()), 'neighborhoods': neighborhoods}
