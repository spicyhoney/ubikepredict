export type RoadPoint = { station_id: number; latitude: number | null; longitude: number | null };
export type RoadRoute = {
  coordinates: [number, number][];
  distance: number;
  duration: number;
  legs: { distance: number; duration: number }[];
  maxSnapDistance: number;
};

export function hasCoordinates<T extends RoadPoint>(point: T): point is T & { latitude: number; longitude: number } {
  return typeof point.latitude === 'number' && Number.isFinite(point.latitude) && Math.abs(point.latitude) <= 90
    && typeof point.longitude === 'number' && Number.isFinite(point.longitude) && Math.abs(point.longitude) <= 180;
}

export function roadRouteUrl(points: RoadPoint[], endpoint: string): string {
  if (points.length < 2 || points.length > 25 || !points.every(hasCoordinates)) {
    throw new Error('站點座標不完整或站數不適用，無法規劃完整道路路線。');
  }
  const coordinates = points.map(p => `${p.longitude.toFixed(6)},${p.latitude.toFixed(6)}`).join(';');
  return `${endpoint.replace(/\/+$/, '')}/route/v1/driving/${coordinates}?overview=full&geometries=geojson&steps=false&alternatives=false&generate_hints=false&radiuses=${points.map(() => '150').join(';')}`;
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object') throw new Error('道路服務回應格式不正確。');
  return value as Record<string, unknown>;
}
function nonnegative(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error('道路距離或時間不正確。');
  return value;
}

export function parseRoadRoute(value: unknown, stopCount: number): RoadRoute {
  const body = object(value);
  if (body.code !== 'Ok') throw new Error(body.code === 'NoSegment' || body.code === 'NoRoute'
    ? '找不到串連所有站點的汽車道路；請確認站點附近是否有可通行道路。'
    : '道路服務暫時無法提供路線。');
  if (!Array.isArray(body.routes) || !body.routes.length) throw new Error('道路服務沒有回傳路線。');
  const route = object(body.routes[0]);
  const geometry = object(route.geometry);
  if (geometry.type !== 'LineString' || !Array.isArray(geometry.coordinates) || geometry.coordinates.length < 2) throw new Error('道路線形不完整。');
  const coordinates: [number, number][] = geometry.coordinates.map(value => {
    if (!Array.isArray(value) || value.length !== 2 || !hasCoordinates({ station_id: 0, longitude: value[0], latitude: value[1] })) throw new Error('道路座標不正確。');
    // OSRM GeoJSON uses longitude/latitude; Leaflet uses latitude/longitude.
    return [value[1], value[0]];
  });
  if (!Array.isArray(route.legs) || route.legs.length !== stopCount - 1 || !Array.isArray(body.waypoints) || body.waypoints.length !== stopCount) throw new Error('道路服務未涵蓋全部站點。');
  const legs = route.legs.map(value => { const leg = object(value); return { distance: nonnegative(leg.distance), duration: nonnegative(leg.duration) }; });
  const maxSnapDistance = Math.max(...body.waypoints.map(value => nonnegative(object(value).distance)));
  if (maxSnapDistance > 150) throw new Error('路線距離站點過遠，無法視為可用的站點路線。');
  return { coordinates, legs, distance: nonnegative(route.distance), duration: nonnegative(route.duration), maxSnapDistance };
}

// Single-user demo: bounded in-memory cache and serialized sends, >=1.1s apart.
// For shared/high-volume operation, use a dedicated routing service with a server-side limiter.
const cache = new Map<string, { expires: number; promise: Promise<unknown> }>();
let queue: Promise<unknown> = Promise.resolve();
let lastStarted = 0;
let pending = 0;
export function fetchRoadRoute(points: RoadPoint[], endpoint: string): Promise<RoadRoute> {
  return fetchRoadData(roadRouteUrl(points, endpoint), value => parseRoadRoute(value, points.length));
}
export function fetchRoadData<T>(url: string, parse: (value: unknown) => T): Promise<T> {
  const hit = cache.get(url);
  if (hit && hit.expires > Date.now()) return hit.promise as Promise<T>;
  if (pending >= 4) return Promise.reject(new Error('道路查詢忙碌中，請稍後再試。'));
  pending++;
  const promise = queue.catch(() => undefined).then(async () => {
    await new Promise(resolve => setTimeout(resolve, Math.max(0, 1100 - (Date.now() - lastStarted))));
    lastStarted = Date.now();
    const response = await fetch(url, { signal: AbortSignal.timeout(12000) });
    if (!response.ok) throw new Error('道路服務暫時忙碌，請稍後再試。');
    return parse(await response.json());
  }).catch(error => { cache.delete(url); throw error; }).finally(() => { pending--; });
  queue = promise;
  cache.set(url, { expires: Date.now() + 10 * 60 * 1000, promise });
  if (cache.size > 16) cache.delete(cache.keys().next().value!);
  return promise;
}
