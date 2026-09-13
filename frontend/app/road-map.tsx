'use client';

import { useEffect, useRef, useState } from 'react';
import type * as Leaflet from 'leaflet';
import type { InventoryStation } from './supply-routing';
import { spreadMapMarkers } from './map-layout';
import { hasCoordinates, type RoadRoute } from './road-routing';

export type MapStation = {
  station_id: number; station_name: string; district: string;
  latitude: number | null; longitude: number | null;
  red_duration_display: string; status: 'scored' | 'insufficient_history';
  risk_probability: number | null; risk_rank: number | null;
  is_alert: boolean; in_action_list: boolean; route_order: number | null;
  distance_from_previous_km: number | null; fleetLabel?: string; fleetColor?: string;
};
type Instance = { L: typeof Leaflet; map: Leaflet.Map; markers: Leaflet.LayerGroup; route: Leaflet.LayerGroup };
export type RoadMapProps = {
  stations: MapStation[]; actions: MapStation[]; selectedStationId: number | null;
  onSelectStation: (id: number) => void;
  outcomes: ReadonlyMap<number, boolean | null>;
  simulatedStationIds: ReadonlySet<number>; revealed: boolean;
  suppliers?: (InventoryStation & { label: string })[]; selectedSupplierId?: number | null;
  onSelectSupplier?: (id: number) => void; supplyRoute?: RoadRoute | null; confirmedRoutes?: RoadRoute[];
  fleetRoutes?: { vehicleId: number; route: RoadRoute; color: string }[];
};
const TILE_URL = process.env.NEXT_PUBLIC_MAP_TILE_URL ?? 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

export default function RoadMap({ stations, selectedStationId, onSelectStation, outcomes, simulatedStationIds, suppliers = [], selectedSupplierId, onSelectSupplier, supplyRoute, confirmedRoutes = [], fleetRoutes }: RoadMapProps) {
  const container = useRef<HTMLDivElement>(null);
  const [instance, setInstance] = useState<Instance | null>(null);
  const [mapError, setMapError] = useState(false);
  const [tileError, setTileError] = useState(false);
  const [mapZoom, setMapZoom] = useState(0);
  const onSelect = useRef(onSelectStation);
  const stationsRef = useRef<(MapStation | InventoryStation)[]>(stations);
  useEffect(() => { stationsRef.current = [...stations, ...suppliers]; }, [stations, suppliers]);
  useEffect(() => { onSelect.current = onSelectStation; }, [onSelectStation]);
  const hasDispatch = fleetRoutes !== undefined || onSelectSupplier !== undefined;
  const selected = stations.find(s => s.station_id === selectedStationId);

  useEffect(() => {
    let disposed = false; let map: Leaflet.Map | undefined; let observer: ResizeObserver | undefined;
    void import('leaflet').then(L => {
      if (disposed || !container.current) return;
      map = L.map(container.current, { center: [25.06, 121.48], zoom: 13, scrollWheelZoom: false });
      map.createPane('targetRoutes').style.zIndex = '350';
      L.tileLayer(TILE_URL, { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors' })
        .on('tileerror', () => { if (!disposed) setTileError(true); }).addTo(map);
      const value = { L, map, markers: L.layerGroup().addTo(map), route: L.layerGroup().addTo(map) };
      setInstance(value);
      map.on('zoomend', () => { if (!disposed) setMapZoom(v => v + 1); });
      let previousWidth = 0; let previousHeight = 0;
      observer = new ResizeObserver(entries => {
        const { width, height } = entries[0].contentRect;
        if (!map || width <= 0 || height <= 0) { previousWidth = width; previousHeight = height; return; }
        map.invalidateSize({ pan: false });
        if (width !== previousWidth || height !== previousHeight) {
          const points = stationsRef.current.filter(hasCoordinates).map(p => [p.latitude, p.longitude] as Leaflet.LatLngTuple);
          if (points.length) map.fitBounds(L.latLngBounds(points), { padding: [50, 50], maxZoom: 16, animate: false });
        }
        previousWidth = width; previousHeight = height;
      });
      observer.observe(container.current);
    }).catch(() => { if (!disposed) setMapError(true); });
    return () => { disposed = true; observer?.disconnect(); map?.remove(); };
  }, []);

  useEffect(() => {
    if (!instance) return;
    const positions = [...stations, ...suppliers].filter(hasCoordinates).map(p => [p.latitude, p.longitude] as Leaflet.LatLngTuple);
    if (positions.length) instance.map.fitBounds(instance.L.latLngBounds(positions), { padding: [38, 38], maxZoom: 16 });
  }, [instance, stations, suppliers]);

  useEffect(() => {
    if (!instance) return;
    const { L, markers } = instance; markers.clearLayers();
    const located = stations.filter(hasCoordinates);
    const anchors = located.map(station => instance.map.latLngToLayerPoint([station.latitude, station.longitude]));
    const supplyAnchors = suppliers.filter(hasCoordinates).map(s => instance.map.latLngToLayerPoint([s.latitude, s.longitude]));
    const positions = spreadMapMarkers([...anchors.map((point, index) => ({ x: point.x, y: point.y, radius: located[index].in_action_list ? 15 : 9 })), ...supplyAnchors.map(point => ({ x: point.x, y: point.y, radius: 16 }))]);
    located.forEach((station, index) => {
      const shifted = positions[index];
      const markerLocation = instance.map.layerPointToLatLng([shifted.x, shifted.y]);
      if (Math.hypot(shifted.x - anchors[index].x, shifted.y - anchors[index].y) > 3) {
        L.polyline([[station.latitude, station.longitude], markerLocation], { color: '#264f63', weight: 1.5, opacity: 0.8, interactive: false }).addTo(markers);
        L.circleMarker([station.latitude, station.longitude], { radius: 2, color: '#264f63', fillOpacity: 1, interactive: false }).addTo(markers);
      }
      const outcome = outcomes.get(station.station_id);
      const color = station.in_action_list ? outcome === true ? 'persistent' : outcome === false ? 'recovered' : 'action'
        : station.status !== 'scored' ? 'unknown' : station.is_alert ? 'alert' : 'below';
      const button = document.createElement('button'); button.type = 'button';
      button.className = `road-station ${color}${station.station_id === selectedStationId ? ' selected' : ''}${simulatedStationIds.has(station.station_id) ? ' simulated' : ''}`;
      button.textContent = station.fleetLabel ?? (station.in_action_list ? String(station.route_order ?? '') : '');
      if (station.fleetColor) { button.style.borderColor = station.fleetColor; button.style.fontSize = '11px'; if (outcome === undefined || outcome === null) { button.style.backgroundColor = station.fleetColor; button.style.color = '#fff'; } }
      const label = `${station.station_name}，${station.risk_probability === null ? '資料不足' : `風險${(station.risk_probability * 100).toFixed(1)}%`}${station.fleetLabel ? `，車輛與站序${station.fleetLabel}` : station.in_action_list ? `，目標第${station.route_order}站` : ''}`;
      button.setAttribute('aria-label', label); button.title = label; button.dataset.stationId = String(station.station_id);
      button.disabled = station.status !== 'scored';
      button.onclick = e => { e.stopPropagation(); onSelect.current(station.station_id); };
      const tooltip = document.createElement('span'); tooltip.textContent = label;
      L.marker(markerLocation, { icon: L.divIcon({ html: button, className: 'road-marker', iconSize: [30, 30], iconAnchor: [15, 15] }), keyboard: false, zIndexOffset: station.station_id === selectedStationId ? 1000 : station.in_action_list ? 500 : 0 })
        .bindTooltip(tooltip, { direction: 'top', offset: [0, -15] }).addTo(markers);
    });
    return () => { markers.clearLayers(); };
  }, [instance, stations, outcomes, simulatedStationIds, selectedStationId, mapZoom, suppliers]);

  useEffect(() => {
    if (!instance) return;
    const { L, map } = instance;
    const layer = L.layerGroup().addTo(map);
    const located = stations.filter(hasCoordinates);
    const supplyLocated = suppliers.filter(hasCoordinates);
    const anchors = [...located, ...supplyLocated].map(s => map.latLngToLayerPoint([s.latitude, s.longitude]));
    const positions = spreadMapMarkers(anchors.map((p, i) => ({ x: p.x, y: p.y, radius: i < located.length ? (located[i].in_action_list ? 15 : 9) : 16 })));
    supplyLocated.forEach((s, i) => {
      const index = located.length + i, shifted = positions[index];
      const location = map.layerPointToLatLng([shifted.x, shifted.y]);
      if (Math.hypot(shifted.x - anchors[index].x, shifted.y - anchors[index].y) > 3) {
        L.polyline([[s.latitude, s.longitude], location], { color: '#8538af', weight: 1.5, interactive: false }).addTo(layer);
        L.circleMarker([s.latitude, s.longitude], { radius: 2, color: '#8538af', fillOpacity: 1, interactive: false }).addTo(layer);
      }
      const button = document.createElement('button'); button.type = 'button';
      button.className = `road-supplier${s.station_id === selectedSupplierId ? ' selected' : ''}`;
      button.textContent = s.label;
      const supplyLabel = s.label === '取' ? `取車站：${s.station_name}` : `供車方案${s.label}：${s.station_name}`;
      button.setAttribute('aria-label', supplyLabel);
      button.title = supplyLabel;
      button.onclick = () => onSelectSupplier?.(s.station_id);
      L.marker(location, { icon: L.divIcon({ html: button, className: 'road-marker', iconSize: [32, 32], iconAnchor: [16, 16] }), keyboard: false, zIndexOffset: 1500 }).addTo(layer);
    });
    if (supplyRoute) {
      L.polyline(supplyRoute.coordinates, { color: '#fff', weight: 9, interactive: false }).addTo(layer);
      L.polyline(supplyRoute.coordinates, { color: '#8538af', weight: 5, dashArray: '9 5', interactive: false, className: 'supply-driving-route' }).addTo(layer);
    }
    return () => { layer.remove(); };
  }, [instance, stations, suppliers, supplyRoute, selectedSupplierId, onSelectSupplier, mapZoom]);

  useEffect(() => {
    if (!instance || !selected || !hasCoordinates(selected)) return;
    instance.map.panInside([selected.latitude, selected.longitude], { padding: [40, 40] });
  }, [instance, selected]);

  useEffect(() => {
    if (!instance) return;
    instance.route.clearLayers();
    for (const route of confirmedRoutes) {
      instance.L.polyline(route.coordinates, { pane: 'targetRoutes', color: '#fff', weight: 8, opacity: 0.9, interactive: false }).addTo(instance.route);
      instance.L.polyline(route.coordinates, { pane: 'targetRoutes', color: '#007d8b', weight: 5, interactive: false, className: 'confirmed-driving-route' }).addTo(instance.route);
    }
    for (const item of fleetRoutes ?? []) {
      instance.L.polyline(item.route.coordinates, { pane: 'targetRoutes', color: '#fff', weight: 9, opacity: 0.9, interactive: false }).addTo(instance.route);
      instance.L.polyline(item.route.coordinates, { pane: 'targetRoutes', color: item.color, weight: 5, interactive: false, className: `fleet-driving-route fleet-vehicle-${item.vehicleId}` }).addTo(instance.route);
    }
  }, [instance, confirmedRoutes, fleetRoutes]);

  return <div className="road-map-content">
    <div className="road-map-stage">
      <div ref={container} className="road-map-canvas" aria-label="OpenStreetMap 真實道路與站點地圖" />
      {!instance && <output className="road-map-message">{mapError ? '地圖載入失敗，請重新整理。' : '正在載入道路地圖…'}</output>}
      {tileError && <output className="road-tile-warning">部分底圖未載入；站點與道路線形仍可查看。</output>}
      <button className="road-fit-button" type="button" onClick={() => {
        if (!instance) return;
        const points = [...stations, ...suppliers].filter(hasCoordinates).map(p => [p.latitude, p.longitude] as Leaflet.LatLngTuple);
        for (const route of confirmedRoutes) points.push(...route.coordinates);
        for (const item of fleetRoutes ?? []) points.push(...item.route.coordinates);
        if (supplyRoute) points.push(...supplyRoute.coordinates);
        if (points.length) instance.map.fitBounds(instance.L.latLngBounds(points), { padding: [38, 38], maxZoom: 16 });
      }}>{hasDispatch ? '顯示完整取送路線' : '顯示全部站點'}</button>
    </div>
    <section className="road-route-summary" aria-label={hasDispatch ? "取送路線圖例" : "風險地圖說明"}>
      {!hasDispatch ? <>
        <strong>點選站點，查看 30 分鐘後仍缺車的風險與模型原因</strong>
        <small>圓圈數字對應右側關注清單；密集標記的細線指向站點原座標。</small>
      </> : fleetRoutes !== undefined ? <>
        <strong>各色實線：各車的完整取送路線 · 圓圈「1-2」代表車 1 的第 2 個送車站</strong>
        <small>紫色「取」：共用取車站，可能由多台車多次拜訪；細線指向站點原座標。路線是否啟用請看上方規劃狀態。</small>
      </> : <>
        <strong>紫色虛線：目前選取的取送路線 · 藍綠實線：已確認的連續取送</strong>
        <small>紫色「取」：這筆任務選定的取車站。未確認前不連接缺車目標；密集標記的細線指向站點原座標。</small>
      </>}
      {hasDispatch && <small>OSRM／OpenStreetMap 現行路網估算；無即時交通、歷史路況或貨車限高限重資訊。停靠位置須現場確認。</small>}
      {supplyRoute && supplyRoute.maxSnapDistance > 30 && <small>部分站點對應至附近道路，最遠 {Math.round(supplyRoute.maxSnapDistance)} 公尺。</small>}
      <a href="https://www.openstreetmap.org/fixthemap" target="_blank" rel="noopener noreferrer">回報底圖問題</a>
    </section>
  </div>;
}
