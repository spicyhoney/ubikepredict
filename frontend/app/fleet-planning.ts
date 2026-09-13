import type { InventoryStation } from './supply-routing';

export type FleetEdge = { duration: number; distance: number };
export type FleetTarget = { station: InventoryStation; risk: number; need: number; suppliers: { station_id: number; available: number }[] };
export type FleetProblem = { targets: FleetTarget[]; stations: InventoryStation[]; travel: Record<string, FleetEdge | null>; vehicles: number; handlingMinutes: number; horizonSeconds?: number; vehicleCapacity?: number };
export type FleetStep = { targetId: number; supplierId: number; need: number; driveSeconds: number; distance: number; endSeconds: number; pickup?: number; onboardAfter?: number };
export type FleetRoute = { vehicleId: number; steps: FleetStep[]; duration: number; distance: number };
export type FleetPlan = { routes: FleetRoute[]; score: number; servedIds: number[]; reserved: Record<string, number>; unservedIds: number[]; diagnostics: { algorithm: string; evaluations: number; expandedStates: number; depths: number; evaluationLimit: number; beamWidth: number; truncated: boolean; optimal: false } };

const EVALUATION_LIMIT = 240_000;
const BEAM_WIDTH = 24;
const EPS = 1e-7;
const finite = (n: number) => Number.isFinite(n) && n >= 0;

function prepare(problem: FleetProblem) {
  if (!Number.isInteger(problem.vehicles) || problem.vehicles < 1 || problem.vehicles > 5) throw new Error('運補車數必須介於 1 至 5。');
  if (!finite(problem.handlingMinutes) || problem.handlingMinutes > 60) throw new Error('作業時間必須介於 0 至 60 分鐘。');
  const capacity = problem.vehicleCapacity ?? 20;
  if (!Number.isInteger(capacity) || capacity < 1 || capacity > 100) throw new Error('車輛容量必須介於 1 至 100 台。');
  const horizon = problem.horizonSeconds ?? 1800;
  if (!finite(horizon) || horizon <= 0) throw new Error('規劃時間必須大於零。');
  const stations = new Map<number, InventoryStation>();
  for (const station of problem.stations) {
    if (!Number.isInteger(station.station_id) || stations.has(station.station_id) || !Number.isInteger(station.bikes) || station.bikes < 0) throw new Error('站點 ID 或庫存無效。');
    stations.set(station.station_id, station);
  }
  const ids = new Set<number>(), capacities: Record<string, number> = {};
  const targets = [...problem.targets].sort((a, b) => a.station.station_id - b.station.station_id);
  for (const target of targets) {
    const id = target.station.station_id, canonical = stations.get(id);
    if (!canonical || ids.has(id) || !finite(target.risk) || target.risk > 1 || !Number.isInteger(target.need) || target.need <= 0 || !Number.isInteger(canonical.docks) || canonical.docks < target.need) throw new Error('缺車目標、風險或需求量無效。');
    ids.add(id);
    const suppliers = new Set<number>();
    for (const supplier of target.suppliers) {
      const station = stations.get(supplier.station_id);
      if (!station || supplier.station_id === id || suppliers.has(supplier.station_id) || !Number.isInteger(supplier.available) || supplier.available < 0 || supplier.available > station.bikes) throw new Error('供車候選或可調庫存無效。');
      suppliers.add(supplier.station_id);
      // Repeated physical stations share one conservative capacity, never one capacity per task.
      capacities[supplier.station_id] = Math.min(capacities[supplier.station_id] ?? supplier.available, supplier.available);
    }
  }
  const edge = (from: number, to: number): FleetEdge | null => {
    if (from === to) return { duration: 0, distance: 0 };
    const value = problem.travel[`${from}:${to}`];
    return value && finite(value.duration) && finite(value.distance) ? value : null;
  };
  return { targets, capacities, horizon, capacity, handling: problem.handlingMinutes * 60, edge };
}

type State = { routes: FleetRoute[]; served: Set<number>; reserved: Record<string, number>; score: number; elapsed: number; key: string };
function compare(a: State, b: State): number {
  return b.score - a.score || b.served.size - a.served.size || a.elapsed - b.elapsed || (a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
}
function stateKey(routes: FleetRoute[]) {
  // Vehicle labels are interchangeable; canonical keys suppress duplicate assignments.
  return routes.map(r => r.steps.map(s => `${s.supplierId}>${s.targetId}`).join(',')).sort().join('|');
}

/** Deterministic bounded beam search, not an optimality proof. No roads or model calls occur here. */
export function solveFleet(problem: FleetProblem): FleetPlan {
  const prepared = prepare(problem);
  const initial: State = { routes: [], served: new Set(), reserved: {}, score: 0, elapsed: 0, key: '' };
  let best = initial, beam = [initial], evaluations = 0, expandedStates = 0, depths = 0, truncated = false;
  for (let depth = 0; depth < prepared.targets.length && beam.length; depth++) {
    const next = new Map<string, State>();
    outer: for (const state of beam) {
      expandedStates++;
      for (const target of prepared.targets) {
        if (state.served.has(target.station.station_id) || target.need > prepared.capacity) continue;
        for (const supplier of [...target.suppliers].sort((a, b) => a.station_id - b.station_id)) {
          if ((state.reserved[supplier.station_id] ?? 0) + target.need > prepared.capacities[supplier.station_id]) continue;
          const delivery = prepared.edge(supplier.station_id, target.station.station_id);
          if (!delivery) continue;
          // Only open the next unused vehicle, eliminating equivalent empty-vehicle branches.
          for (let vehicle = 0; vehicle < Math.min(problem.vehicles, state.routes.length + 1); vehicle++) {
            if (evaluations >= EVALUATION_LIMIT) { truncated = true; break outer; }
            evaluations++;
            const old = state.routes[vehicle];
            const approach = old ? prepared.edge(old.steps[old.steps.length - 1].targetId, supplier.station_id) : { duration: 0, distance: 0 };
            if (!approach) continue;
            const driveSeconds = approach.duration + delivery.duration, distance = approach.distance + delivery.distance;
            const endSeconds = (old?.duration ?? 0) + driveSeconds + prepared.handling;
            if (endSeconds > prepared.horizon) continue;
            const step: FleetStep = { targetId: target.station.station_id, supplierId: supplier.station_id, need: target.need, driveSeconds, distance, endSeconds };
            const route = { vehicleId: vehicle + 1, steps: [...(old?.steps ?? []), step], duration: endSeconds, distance: (old?.distance ?? 0) + distance };
            const routes = [...state.routes]; routes[vehicle] = route;
            const key = stateKey(routes);
            if (next.has(key)) continue;
            const candidate: State = { routes, served: new Set([...state.served, step.targetId]), reserved: { ...state.reserved, [supplier.station_id]: (state.reserved[supplier.station_id] ?? 0) + target.need }, score: state.score + target.risk, elapsed: state.elapsed + driveSeconds + prepared.handling, key };
            next.set(key, candidate);
            if (compare(candidate, best) < 0) best = candidate;
          }
        }
      }
    }
    depths = depth + 1;
    if (truncated) break;
    const states = [...next.values()];
    // Keep both high-score and time-efficient partial plans; neither alone sees future opportunities.
    const selected = new Map<string, State>();
    for (const state of states.sort(compare).slice(0, BEAM_WIDTH / 2)) selected.set(state.key, state);
    states.sort((a, b) => b.score / (b.elapsed + 1) - a.score / (a.elapsed + 1) || compare(a, b));
    for (const state of states) { if (selected.size >= BEAM_WIDTH) break; selected.set(state.key, state); }
    beam = [...selected.values()];
  }
  const servedIds = [...best.served].sort((a, b) => a - b);
  const result: FleetPlan = { routes: best.routes, score: prepared.targets.filter(t => best.served.has(t.station.station_id)).reduce((sum, t) => sum + t.risk, 0), servedIds, reserved: best.reserved, unservedIds: prepared.targets.filter(t => !best.served.has(t.station.station_id)).map(t => t.station.station_id), diagnostics: { algorithm: 'deterministic-diverse-beam-v1', evaluations, expandedStates, depths, evaluationLimit: EVALUATION_LIMIT, beamWidth: BEAM_WIDTH, truncated, optimal: false } };
  const validation = validateFleetPlan(problem, result);
  if (!validation.valid) throw new Error(`派車規劃驗證失敗：${validation.errors.join('；')}`);
  return result;
}

/** Risk-first greedy with consecutive same-donor deliveries loaded together at the batch origin. */
export function solveFleetGreedy(problem: FleetProblem): FleetPlan {
  const p = prepare(problem), routes: FleetRoute[] = [], served = new Set<number>(), reserved: Record<string, number> = {};
  let evaluations = 0;
  const ordered = [...p.targets].sort((a, b) => b.risk - a.risk || a.station.station_id - b.station.station_id);
  for (const target of ordered) {
    if (target.need > p.capacity) continue;
    const choices: { vehicle: number; supplierId: number; driveSeconds: number; distance: number; endSeconds: number; mergeAt: number }[] = [];
    for (const supplier of target.suppliers) {
      if ((reserved[supplier.station_id] ?? 0) + target.need > p.capacities[supplier.station_id]) continue;
      for (let vehicle = 0; vehicle < Math.min(problem.vehicles, routes.length + 1); vehicle++) {
        evaluations++;
        const route = routes[vehicle], last = route?.steps[route.steps.length - 1];
        let mergeAt = -1;
        if (last?.supplierId === supplier.station_id) {
          let index = route.steps.length - 1;
          while (index > 0 && route.steps[index].pickup === 0) index--;
          if ((route.steps[index].pickup ?? route.steps[index].need) + target.need <= p.capacity) mergeAt = index;
        }
        const approach = mergeAt >= 0 ? { duration: 0, distance: 0 } : last ? p.edge(last.targetId, supplier.station_id) : { duration: 0, distance: 0 };
        const delivery = p.edge(mergeAt >= 0 ? last.targetId : supplier.station_id, target.station.station_id);
        if (!approach || !delivery) continue;
        const driveSeconds = approach.duration + delivery.duration, endSeconds = (route?.duration ?? 0) + driveSeconds + p.handling;
        if (endSeconds <= p.horizon) choices.push({ vehicle, supplierId: supplier.station_id, driveSeconds, distance: approach.distance + delivery.distance, endSeconds, mergeAt });
      }
    }
    choices.sort((a, b) => a.driveSeconds - b.driveSeconds || a.endSeconds - b.endSeconds || a.vehicle - b.vehicle || a.supplierId - b.supplierId);
    const choice = choices[0];
    if (!choice) continue;
    const old = routes[choice.vehicle], steps = (old?.steps ?? []).map(step => ({ ...step }));
    if (choice.mergeAt >= 0) {
      const first = steps[choice.mergeAt]; first.pickup = (first.pickup ?? first.need) + target.need;
      for (let i = choice.mergeAt; i < steps.length; i++) steps[i].onboardAfter = (steps[i].onboardAfter ?? 0) + target.need;
    }
    steps.push({ targetId: target.station.station_id, supplierId: choice.supplierId, need: target.need, driveSeconds: choice.driveSeconds, distance: choice.distance, endSeconds: choice.endSeconds, pickup: choice.mergeAt >= 0 ? 0 : target.need, onboardAfter: 0 });
    routes[choice.vehicle] = { vehicleId: choice.vehicle + 1, steps, duration: choice.endSeconds, distance: (old?.distance ?? 0) + choice.distance };
    served.add(target.station.station_id); reserved[choice.supplierId] = (reserved[choice.supplierId] ?? 0) + target.need;
  }
  const result: FleetPlan = { routes, servedIds: [...served].sort((a, b) => a - b), unservedIds: p.targets.filter(t => !served.has(t.station.station_id)).map(t => t.station.station_id), reserved, score: p.targets.filter(t => served.has(t.station.station_id)).reduce((sum, t) => sum + t.risk, 0), diagnostics: { algorithm: 'risk-first-batch-greedy-v1', evaluations, expandedStates: ordered.length, depths: ordered.length, evaluationLimit: ordered.reduce((sum, t) => sum + t.suppliers.length * problem.vehicles, 0), beamWidth: 1, truncated: false, optimal: false } };
  const validation = validateFleetPlan(problem, result);
  if (!validation.valid) throw new Error(`派車規劃驗證失敗：${validation.errors.join('；')}`);
  return result;
}

/** Replay every task from the immutable input, independent of the search state's ledger. */
export function validateFleetPlan(problem: FleetProblem, plan: FleetPlan): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  let prepared: ReturnType<typeof prepare>;
  try { prepared = prepare(problem); } catch (error) { return { valid: false, errors: [String(error)] }; }
  const targets = new Map(prepared.targets.map(t => [t.station.station_id, t]));
  const served = new Set<number>(), vehicles = new Set<number>(), reserved: Record<string, number> = {};
  let score = 0;
  const close = (a: number, b: number) => finite(a) && Math.abs(a - b) <= EPS;
  for (const route of plan.routes) {
    if (!Number.isInteger(route.vehicleId) || route.vehicleId < 1 || route.vehicleId > problem.vehicles || vehicles.has(route.vehicleId) || !route.steps.length) errors.push('車輛編號或空路線無效');
    vehicles.add(route.vehicleId);
    let elapsed = 0, totalDistance = 0, previous: number | null = null, onboard = 0, batchSupplier: number | null = null;
    for (const step of route.steps) {
      const target = targets.get(step.targetId);
      if (!target || served.has(step.targetId) || !target.suppliers.some(s => s.station_id === step.supplierId) || target.need !== step.need) { errors.push('目標重複、候選不符或需求量不符'); continue; }
      served.add(step.targetId); score += target.risk;
      const pickup = step.pickup ?? step.need;
      if (!Number.isInteger(pickup) || pickup < 0) errors.push('取車量無效');
      if (pickup > 0) {
        if (onboard !== 0) errors.push('上一批車尚未送完，不可切換取車批次');
        onboard += pickup; batchSupplier = step.supplierId;
        reserved[step.supplierId] = (reserved[step.supplierId] ?? 0) + pickup;
        if (reserved[step.supplierId] > prepared.capacities[step.supplierId]) errors.push('共用供車庫存超支');
        if (onboard > prepared.capacity) errors.push('車輛載重超過容量');
      } else if (previous === null || batchSupplier !== step.supplierId) errors.push('未先取車或供車批次不符');
      if (onboard < step.need) errors.push('車上庫存不足以送車');
      onboard -= step.need;
      if (step.onboardAfter !== undefined && step.onboardAfter !== onboard) errors.push('車上剩餘數量不符');
      const approach = pickup === 0 || previous === null ? { duration: 0, distance: 0 } : prepared.edge(previous, step.supplierId);
      const delivery = pickup === 0 ? previous === null ? null : prepared.edge(previous, step.targetId) : prepared.edge(step.supplierId, step.targetId);
      if (!approach || !delivery) { errors.push('路段無法通行或缺少道路時間'); continue; }
      const driving = approach.duration + delivery.duration, distance = approach.distance + delivery.distance;
      elapsed += driving + prepared.handling; totalDistance += distance;
      if (!close(step.driveSeconds, driving) || !close(step.distance, distance) || !close(step.endSeconds, elapsed)) errors.push('取送時間或距離不符');
      if (elapsed > prepared.horizon) errors.push('路線超過時間上限');
      previous = step.targetId;
    }
    if (onboard !== 0) errors.push('路線結束仍有未送完的車');
    if (!close(route.duration, elapsed) || !close(route.distance, totalDistance)) errors.push('路線合計不符');
  }
  if (!close(plan.score, score)) errors.push('預期效益不符');
  const sameIds = (actual: number[], expected: number[]) => actual.length === expected.length && [...actual].sort((a, b) => a - b).every((id, i) => id === expected[i]);
  if (!sameIds(plan.servedIds, [...served].sort((a, b) => a - b))) errors.push('已服務站點清單不符');
  if (!sameIds(plan.unservedIds, prepared.targets.filter(t => !served.has(t.station.station_id)).map(t => t.station.station_id))) errors.push('未服務站點清單不符');
  const ledgerIds = new Set([...Object.keys(reserved), ...Object.keys(plan.reserved)]);
  for (const id of ledgerIds) if (plan.reserved[id] !== reserved[id]) errors.push('庫存預留合計不符');
  return { valid: errors.length === 0, errors };
}
