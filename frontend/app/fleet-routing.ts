import type { MapStation } from './road-map';
import { type FleetProblem, type FleetPlan } from './fleet-planning';
import { type SupplyContext, type SupplyPolicy, supplyCandidates, ROUTING_URL } from './supply-routing';
import { fetchRoadData, roadRouteUrl, parseRoadRoute, hasCoordinates, type RoadRoute } from './road-routing';

export type FleetGeometry = { vehicleId: number; route: RoadRoute };
export type FleetProgress = { plan: FleetPlan; started: boolean; geometryReady: boolean; solveMs: number };
type Matrix = { version: number; decisionTime: string; stations: { station_id: number; latitude: number; longitude: number }[]; travel: FleetProblem['travel'] };

export function fleetTargets(context: SupplyContext | null | undefined, stations: MapStation[], limit: number, includeLowRisk = false): MapStation[] {
  if (!context) return [];
  const stock = new Map(context.stations.map(s => [s.station_id, s]));
  // Use the server's alert decision to match its exact threshold, before limiting the pool.
  return stations.filter(s => (includeLowRisk || s.is_alert === true) && s.status === 'scored' && s.risk_probability !== null && Number.isFinite(s.risk_probability) && s.risk_probability >= 0 && s.risk_probability <= 1
    && stock.get(s.station_id)?.bikes === 0 && (stock.get(s.station_id)?.docks ?? 0) > 0)
    .sort((a, b) => b.risk_probability! - a.risk_probability! || a.station_id - b.station_id).slice(0, limit);
}

export function problemFromMatrix(context: SupplyContext, targets: MapStation[], vehicles: number, policy: SupplyPolicy, handlingMinutes: number, matrix: Matrix, vehicleCapacity = 20): FleetProblem {
  if (matrix.version !== 1 || new Date(matrix.decisionTime).getTime() !== new Date(context.decision_time).getTime()
    || !Array.isArray(matrix.stations) || !matrix.travel) throw new Error('這個時間點尚未準備道路矩陣，請使用 6/29 09:00 情境。');
  const indexed = new Map(matrix.stations.map(s => [s.station_id, s]));
  const check = (id: number) => {
    const s = context.stations.find(x => x.station_id === id), cached = indexed.get(id);
    if (!s || !cached || !hasCoordinates(s) || !hasCoordinates(cached)
      || Math.abs(cached.latitude - s.latitude) > .000001 || Math.abs(cached.longitude - s.longitude) > .000001)
      throw new Error('站點座標與道路快取不符，此區域尚不能規劃車隊。');
  };
  const edge = (from: number, to: number) => {
    const value = matrix.travel[`${from}:${to}`];
    if (value === undefined) throw new Error('道路矩陣缺少必要方向，無法確認可行路線。');
    if (value !== null && (!Number.isFinite(value.duration) || value.duration < 0 || !Number.isFinite(value.distance) || value.distance < 0)) throw new Error('道路矩陣包含無效時間。');
    return value;
  };
  const prepared = targets.map(t => {
    check(t.station_id);
    const group = supplyCandidates(context, t.station_id, policy);
    if (!group || t.risk_probability === null) throw new Error('目標站庫存或風險資料不完整。');
    const ranked = group.candidates.flatMap(s => {
      check(s.station_id);
      const road = edge(s.station_id, t.station_id);
      return road ? [{ s, road }] : [];
    }).sort((a, b) => a.road.duration - b.road.duration || a.road.distance - b.road.distance || b.s.available - a.s.available || a.s.station_id - b.s.station_id).slice(0, 3);
    return { station: group.target, risk: t.risk_probability, need: group.need, suppliers: ranked.map(({s}) => ({ station_id: s.station_id, available: s.available })) };
  });
  const donors = new Set(prepared.flatMap(t => t.suppliers.map(s => s.station_id)));
  for (const target of prepared) for (const donor of donors) edge(target.station.station_id, donor);
  for (const target of prepared) for (const next of prepared) edge(target.station.station_id, next.station.station_id);
  return { targets: prepared, stations: context.stations, travel: matrix.travel, vehicles, handlingMinutes, vehicleCapacity, horizonSeconds: 1800 };
}

export async function prepareFleetProblem(context: SupplyContext, targets: MapStation[], vehicles: number, policy: SupplyPolicy, handlingMinutes: number, vehicleCapacity = 20): Promise<FleetProblem> {
  const response = await fetch('/fleet-road-matrix.json', { signal: AbortSignal.timeout(12000) });
  if (!response.ok) throw new Error('無法讀取道路時間快取，請稍後重新規劃。');
  return problemFromMatrix(context, targets, vehicles, policy, handlingMinutes, await response.json(), vehicleCapacity);
}

export async function fetchFleetGeometry(problem: FleetProblem, plan: FleetPlan): Promise<FleetGeometry[]> {
  const byId = new Map(problem.stations.map(s => [s.station_id, s]));
  const results: FleetGeometry[] = [];
  // Each visit remains a waypoint, including repeated visits to one donor.
  for (const vehicle of plan.routes) {
    const ids = vehicle.steps.flatMap(s => s.pickup === 0 ? [s.targetId] : [s.supplierId, s.targetId]);
    const points = ids.map(id => byId.get(id)!);
    const parts: RoadRoute[] = [];
    for (let offset = 0; offset < points.length - 1; offset += 24) {
      const chunk = points.slice(offset, offset + 25);
      const url = `${roadRouteUrl(chunk, ROUTING_URL)}&continue_straight=false`;
      parts.push(await fetchRoadData(url, body => parseRoadRoute(body, chunk.length)));
    }
    const route: RoadRoute = {
      coordinates: parts.flatMap((p, i) => i ? p.coordinates.slice(1) : p.coordinates),
      distance: parts.reduce((sum, p) => sum + p.distance, 0),
      duration: parts.reduce((sum, p) => sum + p.duration, 0),
      legs: parts.flatMap(p => p.legs),
      maxSnapDistance: Math.max(...parts.map(p => p.maxSnapDistance)),
    };
    const drive = vehicle.steps.reduce((sum, s) => sum + s.driveSeconds, 0);
    if (Math.abs(route.duration - drive) > 2 || route.duration + vehicle.steps.length * problem.handlingMinutes * 60 > (problem.horizonSeconds ?? 1800) + .01)
      throw new Error(`第 ${vehicle.vehicleId} 車道路時間與快取不一致，尚不能開始運送。請稍後重試。`);
    results.push({ vehicleId: vehicle.vehicleId, route });
  }
  return results;
}
