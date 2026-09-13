import { fetchRoadData, hasCoordinates, type RoadPoint } from './road-routing';

export type InventoryStation = RoadPoint & { station_name: string; district: string; bikes: number; docks: number; capacity: number };
export type SupplyContext = { decision_time: string; snapshot_station_count: number; stations: InventoryStation[]; neighborhoods: Record<string, number[]> };
export type SupplyPolicy = 'fixed5' | 'hybrid20';
export type SupplyCandidate = InventoryStation & { reserve: number; available: number; after: number };
export type RankedSupply = SupplyCandidate & { duration: number; distance: number };
export const ROUTING_URL = process.env.NEXT_PUBLIC_ROUTING_BASE_URL ?? 'https://routing.openstreetmap.de/routed-car';
export function stockTarget(station: InventoryStation, policy: SupplyPolicy): number {
  return Math.min(station.capacity, Math.max(5, policy === 'hybrid20' ? Math.ceil(station.capacity * .2) : 5));
}
export function supplyCandidates(context: SupplyContext, targetId: number, policy: SupplyPolicy) {
  const byId = new Map(context.stations.map(s => [s.station_id, s]));
  const target = byId.get(targetId);
  if (!target || target.bikes !== 0 || target.docks <= 0) return null;
  const need = Math.max(0, stockTarget(target, policy) - target.bikes);
  const nearest = context.neighborhoods[String(targetId)] ?? [];
  const candidates: SupplyCandidate[] = nearest.flatMap(id => {
    const s = byId.get(id);
    if (!s || id === targetId || !hasCoordinates(s)) return [];
    const reserve = stockTarget(s, policy), available = Math.max(0, s.bikes - reserve);
    return available >= need && target.docks >= need ? [{ ...s, reserve, available, after: s.bikes - need }] : [];
  });
  return { target, need, candidates, searched: nearest.length };
}
export function parseSupplyTable(body: unknown, candidates: SupplyCandidate[]): RankedSupply[] {
  const b = body as { code?: unknown; durations?: unknown[][]; distances?: unknown[][]; sources?: { distance?: unknown }[]; destinations?: { distance?: unknown }[] };
  if (!b || b.code !== 'Ok' || !Array.isArray(b.durations) || !Array.isArray(b.distances) || b.durations.length !== candidates.length || b.distances.length !== candidates.length || b.sources?.length !== candidates.length || b.destinations?.length !== 1) throw new Error('道路時間表回應不完整，尚無法排序。');
  const validNumber = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  const targetSnap = b.destinations[0].distance;
  if (!validNumber(targetSnap) || targetSnap > 150) throw new Error('目標站離可行車道路過遠。');
  return candidates.flatMap((s, i) => {
    if (!Array.isArray(b.durations![i]) || b.durations![i].length !== 1 || !Array.isArray(b.distances![i]) || b.distances![i].length !== 1) throw new Error('道路時間表欄位不完整。');
    const duration = b.durations![i][0], distance = b.distances![i][0], snap = b.sources![i].distance;
    if (duration === null && distance === null) return [];
    if (!validNumber(duration) || !validNumber(distance) || !validNumber(snap)) throw new Error('道路時間表包含無效數值。');
    return snap <= 150 ? [{ ...s, duration, distance }] : [];
  }).sort((a, b) => a.duration - b.duration || a.distance - b.distance || b.available - a.available || a.station_id - b.station_id);
}
export function fetchSupplyRanking(target: InventoryStation, candidates: SupplyCandidate[]): Promise<RankedSupply[]> {
  const points = [target, ...candidates];
  if (!candidates.length || candidates.length > 20 || !points.every(hasCoordinates)) throw new Error('供車站座標不完整。');
  const coords = points.map(p => `${p.longitude.toFixed(6)},${p.latitude.toFixed(6)}`).join(';');
  const url = `${ROUTING_URL}/table/v1/driving/${coords}?sources=${candidates.map((_, i) => i + 1).join(';')}&destinations=0&annotations=duration,distance&radiuses=${points.map(() => 150).join(';')}`;
  // Cache the response, not inventory-dependent ranking, so policy changes retain current stock rules.
  return fetchRoadData(url, value => value).then(value => parseSupplyTable(value, candidates));
}
