'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import RoadMap, { type MapStation } from './road-map';
import type { SupplyContext, SupplyPolicy } from './supply-routing';
import { solveFleetGreedy, validateFleetPlan, type FleetPlan } from './fleet-planning';
import { prepareFleetProblem, fetchFleetGeometry, type FleetGeometry, type FleetProgress } from './fleet-routing';

export const FLEET_COLORS = ['#007d8b', '#c76400', '#345fd1', '#923ab9', '#ba3860'];
type FleetMapProps = {
  context: SupplyContext; stations: MapStation[]; targets: MapStation[]; vehicles: number;
  policy: SupplyPolicy; handlingMinutes: number; vehicleCapacity?: number; selectedStationId: number | null;
  onSelectStation: (id: number) => void; outcomes: Record<string, boolean | null>;
  onProgress: (progress: FleetProgress | null) => void;
};
const minutes = (seconds: number) => (seconds / 60).toFixed(1);

export default function FleetMap({ context, stations, targets, vehicles, policy, handlingMinutes, vehicleCapacity = 20, selectedStationId, onSelectStation, outcomes, onProgress }: FleetMapProps) {
  const [plan, setPlan] = useState<FleetPlan | null>(null);
  const [geometry, setGeometry] = useState<FleetGeometry[]>([]);
  const [phase, setPhase] = useState<'idle' | 'matrix' | 'geometry' | 'ready' | 'started' | 'error'>('idle');
  const [error, setError] = useState('');
  const [solveMs, setSolveMs] = useState(0);
  const [hiddenVehicles, setHiddenVehicles] = useState<number[]>([]);
  const [selectedSupplierId, setSelectedSupplierId] = useState<number | null>(null);
  const generation = useRef(0), progress = useRef(onProgress);
  useEffect(() => { progress.current = onProgress; }, [onProgress]);
  useEffect(() => () => { generation.current++; }, []);
  const busy = phase === 'matrix' || phase === 'geometry';
  const byId = useMemo(() => new Map(context.stations.map(s => [s.station_id, s])), [context]);
  const name = (id: number) => byId.get(id)?.station_name ?? `站點 ${id}`;
  const reset = () => {
    generation.current++; setHiddenVehicles([]); setPlan(null); setGeometry([]); setPhase('idle'); setError(''); setSolveMs(0); setSelectedSupplierId(null); progress.current(null);
  };
  async function arrange() {
    const request = ++generation.current;
    setHiddenVehicles([]); setPlan(null); setGeometry([]); setPhase('matrix'); setError(''); progress.current(null);
    try {
      const problem = await prepareFleetProblem(context, targets, vehicles, policy, handlingMinutes, vehicleCapacity);
      if (request !== generation.current) return;
      const start = performance.now(), result = solveFleetGreedy(problem), elapsed = performance.now() - start;
      const validation = validateFleetPlan(problem, result);
      if (!validation.valid) throw new Error(validation.errors.join('；'));
      setPlan(result); setSolveMs(elapsed);
      progress.current({ plan: result, started: false, geometryReady: false, solveMs: elapsed });
      if (!result.routes.length) { setPhase('ready'); return; }
      setPhase('geometry');
      const roads = await fetchFleetGeometry(problem, result);
      if (request !== generation.current) return;
      if (roads.length !== result.routes.length || result.routes.some(route => !roads.some(road => road.vehicleId === route.vehicleId))) throw new Error('部分車輛道路尚未取得，請重新規劃後再開始運送。');
      setGeometry(roads); setPhase('ready');
      progress.current({ plan: result, started: false, geometryReady: true, solveMs: elapsed });
    } catch (failure) {
      if (request !== generation.current) return;
      setError(failure instanceof Error ? failure.message : '路線規劃失敗，請稍後再試。'); setPhase('error');
    }
  }
  const visibleRoutes = useMemo(() => (plan?.routes ?? []).filter(route => !hiddenVehicles.includes(route.vehicleId)), [plan, hiddenVehicles]);
  const visibleTargets = useMemo(() => new Set(visibleRoutes.flatMap(route => route.steps.map(step => step.targetId))), [visibleRoutes]);
  const visibleDonors = useMemo(() => new Set(visibleRoutes.flatMap(route => route.steps.filter(step => (step.pickup ?? step.need) > 0).map(step => step.supplierId))), [visibleRoutes]);
  const fleetRoutes = useMemo(() => geometry.filter(item => !hiddenVehicles.includes(item.vehicleId)).map(item => ({ ...item, color: FLEET_COLORS[item.vehicleId - 1] })), [geometry, hiddenVehicles]);
  const onlyVehicle = (id: number) => setHiddenVehicles((plan?.routes ?? []).filter(route => route.vehicleId !== id).map(route => route.vehicleId));
  const assigned = useMemo(() => {
    const labels = new Map<number, { fleetLabel: string; fleetColor: string; order: number }>();
    plan?.routes.forEach(route => route.steps.forEach((step, index) => labels.set(step.targetId, { fleetLabel: `${route.vehicleId}-${index + 1}`, fleetColor: FLEET_COLORS[route.vehicleId - 1], order: index + 1 })));
    return labels;
  }, [plan]);
  const targetIds = useMemo(() => new Set(targets.map(t => t.station_id)), [targets]);
  const mapStations = useMemo(() => stations.filter(station => !plan?.reserved[station.station_id] && (!assigned.has(station.station_id) || visibleTargets.has(station.station_id))).map(station => {
    const assignment = assigned.get(station.station_id);
    return { ...station, in_action_list: Boolean(assignment), route_order: assignment?.order ?? null, fleetLabel: assignment?.fleetLabel, fleetColor: assignment?.fleetColor };
  }), [stations, assigned, plan, visibleTargets]);
  const suppliers = useMemo(() => Object.keys(plan?.reserved ?? {}).map(Number).flatMap(id => {
    const station = byId.get(id); return station ? [{ ...station, label: '取' }] : [];
  }), [plan, byId]);
  const visibleSuppliers = useMemo(() => suppliers.filter(station => visibleDonors.has(station.station_id)), [suppliers, visibleDonors]);
  const outcomeMap = useMemo(() => new Map(Object.entries(outcomes).map(([id, value]) => [Number(id), value])), [outcomes]);
  const simulated = useMemo(() => new Set(phase === 'started' ? plan?.servedIds ?? [] : []), [phase, plan]);
  const startDelivery = () => {
    if (!plan?.routes.length || phase !== 'ready' || geometry.length !== plan.routes.length) return;
    setPhase('started'); progress.current({ plan, started: true, geometryReady: true, solveMs });
  };
  return <div className="fleet-map">
    <section className="fleet-toolbar" aria-label="自動分配運補車">
      <div><strong>{phase === 'started' ? '已開始運送' : plan ? '建議分工預覽' : '建立多車取送方案'}</strong>
        <p>候選 {targetIds.size} 站 · 可用 {vehicles} 台車 · 每車最多 30 分鐘</p></div>
      <div className="fleet-controls">
        <button type="button" onClick={() => void arrange()} disabled={busy || phase === 'started' || !targets.length}>{busy ? '規劃中…' : plan || error ? '重新規劃' : '自動安排取送'}</button>
        <button type="button" onClick={startDelivery} disabled={phase !== 'ready' || !plan?.routes.length || geometry.length !== plan.routes.length}>開始運送</button>
        {(plan || busy || error) && <button type="button" onClick={reset}>{phase === 'started' ? '返回規劃並清除' : '清除方案'}</button>}
      </div>
      <output aria-live="polite">{phase === 'matrix' ? '正在取得候選站道路時間並安排分工；首次查詢可能較久。' : phase === 'geometry' ? '已算出可行分工，正在取得各車道路線形。' : phase === 'started' ? '本次方案已啟用；揭曉時只評估這些已安排的站。' : plan?.routes.length ? '目前為預覽，按「開始運送」才納入本次派送評估。' : plan ? '目前候選庫存與 30 分鐘限制下沒有可安排路線。' : '每站先保留 Top 3 供車候選，再依風險由高至低逐站選擇可行取送，共用同一份庫存。'}</output>
      {error && <p className="fleet-error" role="alert">{error}</p>}
    </section>
    {Boolean(plan?.routes.length) && <fieldset className="fleet-visibility">
      <legend>地圖顯示的派送員</legend>
      <div className="fleet-visibility-options">{plan!.routes.map(route => <label key={route.vehicleId}>
        <input type="checkbox" checked={!hiddenVehicles.includes(route.vehicleId)} onChange={e => setHiddenVehicles(current => e.target.checked ? current.filter(id => id !== route.vehicleId) : [...current, route.vehicleId])} />
        <i style={{ backgroundColor: FLEET_COLORS[route.vehicleId - 1] }} />派送員 {route.vehicleId}<small>（{route.steps.length} 站）</small>
      </label>)}</div>
      <div className="fleet-visibility-actions">{plan!.routes.map(route => <button key={route.vehicleId} type="button" onClick={() => onlyVehicle(route.vehicleId)}>只看派送員 {route.vehicleId}</button>)}<button type="button" onClick={() => setHiddenVehicles([])}>全部顯示</button><button type="button" onClick={() => setHiddenVehicles(plan!.routes.map(route => route.vehicleId))}>全部隱藏</button></div>
      <output aria-live="polite">已顯示 {visibleRoutes.length}／{plan!.routes.length} 條路線；只切換地圖顯示，不改變派送計畫或命中率。</output>
    </fieldset>}
    <RoadMap stations={mapStations} actions={mapStations.filter(s => s.in_action_list)} selectedStationId={selectedStationId} onSelectStation={onSelectStation} outcomes={outcomeMap} simulatedStationIds={simulated} revealed={Object.values(outcomes).some(value => value !== null)} suppliers={visibleSuppliers} selectedSupplierId={selectedSupplierId} onSelectSupplier={setSelectedSupplierId} fleetRoutes={fleetRoutes} />
    {plan && <section className="fleet-results" aria-label="各車分工與共用庫存">
      <div className="fleet-summary"><strong>建議 {plan.routes.length} 台車 · 安排 {plan.servedIds.length}／{targetIds.size} 站</strong><span>預期持續缺車站數 {plan.score.toFixed(2)} · 求解 {(solveMs / 1000).toFixed(2)} 秒</span></div>
      <p className="fleet-note">預期值為已安排站點的風險機率加總；依風險排序，逐站選擇可行取送；這是 greedy 分工，非全域最佳。道路查詢時間另計。</p>
      <div className="fleet-route-list">{plan.routes.map(route => <article className="fleet-route-card" key={route.vehicleId} style={{ borderColor: FLEET_COLORS[route.vehicleId - 1] }}>
        <h3 style={{ color: FLEET_COLORS[route.vehicleId - 1] }}>車 {route.vehicleId} · {route.steps.length} 站 · {minutes(route.duration)} 分鐘 · {(route.distance / 1000).toFixed(2)} 公里</h3>
        <button type="button" className="fleet-focus-button" onClick={() => onlyVehicle(route.vehicleId)}>只看派送員 {route.vehicleId}</button>
        {hiddenVehicles.includes(route.vehicleId) && <small className="fleet-hidden-note">此路線在地圖上已隱藏</small>}
        <p className="fleet-itinerary">{route.steps.flatMap(step => step.pickup === 0 ? [name(step.targetId)] : [name(step.supplierId), name(step.targetId)]).join(' → ')}</p>
        <ol>{route.steps.map((step, index) => <li key={step.targetId}>
          <button type="button" className="fleet-target-button" onClick={() => onSelectStation(step.targetId)}>{route.vehicleId}-{index + 1}　{name(step.targetId)}</button>
          <span>{step.pickup === 0 ? `沿用車上來自 ${name(step.supplierId)} 的車輛` : `向 ${name(step.supplierId)} 取 ${step.pickup ?? step.need} 台`} · 本站送 {step.need} 台 · 送完車上 {step.onboardAfter ?? 0} 台 · 累計 {minutes(step.endSeconds)} 分鐘</span>
        </li>)}</ol>
      </article>)}</div>
      {plan.unservedIds.length > 0 && <p className="fleet-unserved">本方案未安排 {plan.unservedIds.length} 站：{plan.unservedIds.map(name).join('、')}。這不代表已證明無法供應，可調整人力或條件後重算。</p>}
      <details className="fleet-ledger"><summary>共用庫存預留 · {suppliers.length} 個取車站</summary>
        <p>相同「取」站只畫一個標記；各車取車數共用扣帳；同車連續向同站取車時，在容量內合併為一次取車、多站送達。每站仍計入作業時間。</p>
        <ul>{suppliers.map(station => <li key={station.station_id}>{station.station_name}：原有 {station.bikes} 台，所有車合計預留 {plan.reserved[station.station_id]} 台，取完剩 {station.bikes - plan.reserved[station.station_id]} 台。</li>)}</ul>
      </details>
    </section>}
  </div>;
}
