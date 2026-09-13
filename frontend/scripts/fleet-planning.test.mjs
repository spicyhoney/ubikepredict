import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';
const code = ts.transpileModule(fs.readFileSync(new URL('../app/fleet-planning.ts', import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { solveFleet, solveFleetGreedy, validateFleetPlan } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
const station = (id, bikes = 0) => ({ station_id: id, station_name: String(id), district: 'test', latitude: 25, longitude: 121, bikes, capacity: 40, docks: 40 - bikes });
function problem(count = 3, vehicles = 1, available = 15) {
  const targets = Array.from({ length: count }, (_, i) => ({ station: station(i + 1), risk: .8, need: 5, suppliers: [{ station_id: 100, available }] }));
  const stations = [...targets.map(t => t.station), station(100, 30)];
  const travel = {};
  for (const a of stations) for (const b of stations) travel[`${a.station_id}:${b.station_id}`] = { duration: a.station_id === b.station_id ? 0 : 60, distance: a.station_id === b.station_id ? 0 : 500 };
  return { targets, stations, travel, vehicles, handlingMinutes: 3 };
}
test('same donor may recur on one route; each visit repositions and spends a fresh task handling period', () => {
  const p = problem(); const original = JSON.stringify(p); const plan = solveFleet(p);
  assert.equal(plan.routes.length, 1); assert.deepEqual(plan.servedIds, [1, 2, 3]);
  assert.equal(plan.reserved[100], 15); assert.equal(plan.routes[0].duration, 840);
  assert.deepEqual(plan.routes[0].steps.map(s => s.driveSeconds), [60, 120, 120]);
  assert.equal(JSON.stringify(p), original); assert.equal(validateFleetPlan(p, plan).valid, true);
});
test('all vehicles share one donor ledger, so additional vehicles cannot manufacture inventory', () => {
  const p = problem(4, 5, 10); const plan = solveFleet(p);
  assert.equal(plan.servedIds.length, 2); assert.equal(plan.reserved[100], 10);
  assert.equal(new Set(plan.servedIds).size, 2); assert.equal(plan.unservedIds.length, 2);
  assert.equal(validateFleetPlan(p, plan).valid, true);
  const insufficient = problem(4, 5, 4); assert.equal(solveFleet(insufficient).servedIds.length, 0);
});
test('30 minute horizon includes handling, allows exact boundary and rejects even slightly late', () => {
  const p = problem(1); p.travel['100:1'].duration = 1620;
  assert.equal(solveFleet(p).routes[0].duration, 1800);
  p.travel['100:1'].duration = 1620.01; assert.equal(solveFleet(p).servedIds.length, 0);
  p.handlingMinutes = 0; assert.equal(solveFleet(p).servedIds.length, 1);
});
test('missing or invalid directed approach cannot be replaced by reverse edge or straight distance', () => {
  const p = problem(2); p.travel['1:100'] = null; p.travel['2:100'] = null;
  assert.equal(solveFleet(p).servedIds.length, 1);
  p.vehicles = 2; assert.equal(solveFleet(p).servedIds.length, 2); // both start at donor
  p.travel['100:1'] = { duration: NaN, distance: 20 }; delete p.travel['100:2'];
  assert.equal(solveFleet(p).servedIds.length, 0);
});
test('joint planning can skip a costly highest-risk target and choose several useful nearby targets', () => {
  const p = problem(4); p.targets[0].risk = .99;
  p.travel['100:1'].duration = 1500; p.travel['1:100'].duration = 1500;
  const plan = solveFleet(p);
  assert.deepEqual(plan.servedIds, [2, 3, 4]); assert.ok(Math.abs(plan.score - 2.4) < 1e-9);
  assert.ok(plan.score > .99); assert.equal(plan.diagnostics.optimal, false);
});
test('alternative donor choice avoids shared-stock bottleneck while preserving fixed candidate pools', () => {
  const p = problem(2, 1, 5); const alt = station(101, 20); p.stations.push(alt);
  p.targets[0].suppliers.push({ station_id: 101, available: 5 });
  p.travel['101:1'] = { duration: 70, distance: 600 };
  p.travel['2:101'] = { duration: 60, distance: 500 };
  const plan = solveFleet(p); assert.equal(plan.servedIds.length, 2);
  assert.equal(plan.reserved[100], 5); assert.equal(plan.reserved[101], 5);
  assert.equal(plan.routes[0].steps.find(s => s.targetId === 1).supplierId, 101);
});
test('inconsistent donor availability is conservatively shared and bad inputs are rejected', () => {
  const p = problem(2, 2, 10); p.targets[1].suppliers[0].available = 5;
  assert.equal(solveFleet(p).servedIds.length, 1);
  for (const vehicles of [0, 6, 1.5]) assert.throws(() => solveFleet({ ...p, vehicles }));
  for (const handlingMinutes of [-1, NaN, 61]) assert.throws(() => solveFleet({ ...p, handlingMinutes }));
  const duplicate = structuredClone(p); duplicate.targets.push(duplicate.targets[0]); assert.throws(() => solveFleet(duplicate));
});
test('independent replay catches tampered reservations, duplicated targets, time and summaries', () => {
  const p = problem(3, 2), original = solveFleet(p);
  for (const corrupt of [
    plan => { plan.reserved[100] = 0; },
    plan => { plan.routes[0].steps.push(plan.routes[0].steps[0]); },
    plan => { plan.routes[0].steps[0].endSeconds = 1; },
    plan => { plan.score = 100; },
    plan => { plan.servedIds = []; },
  ]) { const plan = structuredClone(original); corrupt(plan); assert.equal(validateFleetPlan(p, plan).valid, false); }
});
test('bounded 30-target / five-vehicle search is deterministic including diagnostics', () => {
  const p = problem(30, 5, 30); p.stations.find(s => s.station_id === 100).bikes = 40;
  const original = JSON.stringify(p); const a = solveFleet(p), b = solveFleet(p);
  assert.deepEqual(a, b); assert.ok(a.diagnostics.evaluations <= a.diagnostics.evaluationLimit);
  assert.ok(a.routes.length <= 5); assert.equal(validateFleetPlan(p, a).valid, true);
  assert.equal(JSON.stringify(p), original);
});
test('five-donor candidate pool exercises bounded search without losing physical feasibility', () => {
  const p = problem(30, 5, 30);
  for (let id = 101; id <= 104; id++) p.stations.push(station(id, 30));
  p.targets.forEach((target, i) => {
    target.risk = .5 + (i % 5) * .1;
    target.suppliers = [100, 101, 102, 103, 104].map(station_id => ({ station_id, available: 30 }));
  });
  for (const a of p.stations) for (const b of p.stations) p.travel[`${a.station_id}:${b.station_id}`] = { duration: a.station_id === b.station_id ? 0 : 60 + (a.station_id * 7 + b.station_id * 13) % 120, distance: 500 };
  const a = solveFleet(p), b = solveFleet(p);
  assert.deepEqual(a, b); assert.equal(validateFleetPlan(p, a).valid, true);
  assert.ok(a.diagnostics.evaluations <= a.diagnostics.evaluationLimit);
  assert.ok(a.servedIds.length >= 15);
});

test('batch greedy loads C once, delivers C->1->2 and records onboard bikes and exact ledger', () => {
  const p = problem(2); p.targets[0].risk=.9;
  const original=JSON.stringify(p), plan=solveFleetGreedy(p), steps=plan.routes[0].steps;
  assert.equal(plan.routes.length,1); assert.deepEqual(steps.map(s=>s.pickup),[10,0]);
  assert.deepEqual(steps.map(s=>s.onboardAfter),[5,0]); assert.deepEqual(steps.map(s=>s.driveSeconds),[60,60]);
  assert.equal(plan.routes[0].duration,480); assert.equal(plan.reserved[100],10);
  assert.equal(validateFleetPlan(p,plan).valid,true); assert.equal(JSON.stringify(p),original);
  assert.deepEqual(plan,solveFleetGreedy(p));
});
test('capacity five forces a second donor visit; capacity ten creates bounded separate batches', () => {
  const p=problem(3); p.vehicleCapacity=5;
  const small=solveFleetGreedy(p); assert.deepEqual(small.routes[0].steps.map(s=>s.pickup),[5,5,5]);
  assert.deepEqual(small.routes[0].steps.map(s=>s.driveSeconds),[60,120,120]);
  p.vehicleCapacity=10;
  const medium=solveFleetGreedy(p); assert.deepEqual(medium.routes[0].steps.map(s=>s.pickup),[10,0,5]);
  assert.deepEqual(medium.routes[0].steps.map(s=>s.onboardAfter),[5,0,0]);
  assert.equal(validateFleetPlan(p,medium).valid,true);
});
test('risk-first greedy commits highest risk first, skips infeasible targets, and shares all vehicle inventory', () => {
  const p=problem(4,5,10); p.targets[0].risk=1;p.travel['100:1']=null;p.targets[3].risk=.95;
  const plan=solveFleetGreedy(p);assert.deepEqual(plan.servedIds,[2,4]);assert.equal(plan.routes[0].steps[0].targetId,4);
  assert.equal(plan.reserved[100],10);assert.equal(validateFleetPlan(p,plan).valid,true);
  const costly=problem(4);costly.targets[0].risk=.99;costly.travel['100:1'].duration=1500;costly.travel['1:100'].duration=1500;
  assert.deepEqual(solveFleetGreedy(costly).servedIds,[1]);assert.deepEqual(solveFleet(costly).servedIds,[2,3,4]);
});
test('batch replay rejects stolen cargo, incorrect origin, overcapacity and leftover bikes', () => {
  const p=problem(2),plan=solveFleetGreedy(p);
  for(const mutate of [
    x=>{x.routes[0].steps[0].pickup=5;},
    x=>{x.routes[0].steps[0].pickup=25;},
    x=>{x.routes[0].steps[0].onboardAfter=0;},
    x=>{x.routes[0].steps[1].pickup=5;},
    x=>{x.routes[0].steps[0].pickup=11;},
  ]) {const bad=structuredClone(plan);mutate(bad);assert.equal(validateFleetPlan(p,bad).valid,false);}
  const newTarget=problem(1);newTarget.vehicleCapacity=4;assert.equal(solveFleetGreedy(newTarget).servedIds.length,0);
  assert.throws(()=>solveFleetGreedy({...p,vehicleCapacity:0}));assert.throws(()=>solveFleetGreedy({...p,vehicleCapacity:NaN}));
});
test('batch delivery uses directed target-to-target edge and includes handling within thirty minutes', () => {
  const p=problem(2);p.targets[0].risk=.9;p.travel['1:2']={duration:1380,distance:500};
  const exact=solveFleetGreedy(p);assert.equal(exact.routes[0].duration,1800);assert.equal(exact.servedIds.length,2);
  p.travel['1:2'].duration=1380.01;assert.equal(solveFleetGreedy(p).servedIds.length,1);
  p.travel['1:2']=null;assert.equal(solveFleetGreedy(p).servedIds.length,1);
});
