import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';
const encoded = new Map();
function moduleUrl(name) {
  if (encoded.has(name)) return encoded.get(name);
  let code = ts.transpileModule(fs.readFileSync(new URL(`../app/${name}.ts`, import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  code = code.replace(/from ['"]\.\/([^'"]+)['"]/g, (_, dependency) => `from '${moduleUrl(dependency)}'`);
  const url = 'data:text/javascript;base64,' + Buffer.from(code).toString('base64'); encoded.set(name, url); return url;
}
const { fleetTargets, problemFromMatrix } = await import(moduleUrl('fleet-routing'));
function fixture() {
  const station = (id, bikes = 0) => ({ station_id: id, station_name: String(id), district: 'demo', latitude: 25 + id / 10000, longitude: 121, bikes, docks: 40 - bikes, capacity: 40 });
  const inventory = [station(1), station(2), station(101, 15), station(102, 30), station(103, 20), station(104, 40)];
  const context = { decision_time: '2026-06-29T09:00:00+08:00', snapshot_station_count: inventory.length, stations: inventory, neighborhoods: { '1': [104,102,103,101], '2': [104,102,103,101] } };
  const targets = inventory.slice(0,2).map((s,i) => ({ ...s, risk_probability: i ? .9 : .7, status: 'scored', is_alert: false, in_action_list: false }));
  const travel = {};
  for (const from of inventory) for (const to of inventory) travel[`${from.station_id}:${to.station_id}`] = { duration: from.station_id === to.station_id ? 0 : 60, distance: from.station_id === to.station_id ? 0 : 500 };
  for (const target of targets) for (const [id, duration] of [[101,10],[102,30],[103,20],[104,40]]) travel[`${id}:${target.station_id}`] = { duration, distance: 500 };
  const matrix = { version: 1, decisionTime: context.decision_time, stations: inventory.map(({station_id,latitude,longitude}) => ({station_id,latitude,longitude})), travel };
  return { context, targets, matrix };
}
test('fleet targets include below-threshold scored empty stations and rank by risk then station ID', () => {
  const {context,targets} = fixture();
  const tied = {...targets[0],station_id:3,risk_probability:.9}; context.stations.push({...context.stations[0],station_id:3});
  const stocked = {...targets[0],station_id:101,risk_probability:.99};
  const unavailable = {...targets[0],station_id:4,status:'insufficient_history',risk_probability:null}; context.stations.push({...context.stations[0],station_id:4});
  assert.deepEqual(fleetTargets(context,[targets[0],tied,stocked,targets[1],unavailable],30,true).map(s=>s.station_id),[2,3,1]);
  assert.deepEqual(fleetTargets(context,[targets[0],tied,targets[1]],2,true).map(s=>s.station_id),[2,3]);
  assert.deepEqual(fleetTargets(null,targets,30),[]);
  assert.deepEqual(fleetTargets(context,[{...targets[0],status:'insufficient_history',risk_probability:.99},{...targets[1],risk_probability:1.1}],30),[]);
});
test('Top 3 donors rank by directed road seconds rather than maximum inventory', () => {
  const {context,targets,matrix}=fixture(); const p=problemFromMatrix(context,targets,2,'fixed5',3,matrix,20);
  assert.deepEqual(p.targets[0].suppliers.map(s=>s.station_id),[101,103,102]);
  assert.equal(p.targets[0].suppliers[0].available,10); assert.equal(p.vehicleCapacity,20); assert.equal(p.horizonSeconds,1800);
});
test('reject mismatched decision time and coordinate metadata', () => {
  const {context,targets,matrix}=fixture(); const badTime=structuredClone(matrix);badTime.decisionTime='2026-06-29T09:30:00+08:00';
  assert.throws(()=>problemFromMatrix(context,targets,2,'fixed5',3,badTime),/時間點/);
  const badCoords=structuredClone(matrix);badCoords.stations[0].latitude+=.01;
  assert.throws(()=>problemFromMatrix(context,targets,2,'fixed5',3,badCoords),/座標/);
});
test('reject missing metadata coordinates, not silently comparing undefined as NaN', () => {
  const {context,targets,matrix}=fixture();delete matrix.stations[0].latitude;
  assert.throws(()=>problemFromMatrix(context,targets,2,'fixed5',3,matrix),/座標/);
});
test('required directed approaches and direct target-to-target legs cannot use reverse fallbacks', () => {
  const {context,targets,matrix}=fixture();delete matrix.travel['2:101'];
  assert.throws(()=>problemFromMatrix(context,targets,2,'fixed5',3,matrix),/必要方向/);
  const f=fixture();delete f.matrix.travel['1:2'];
  assert.throws(()=>problemFromMatrix(f.context,f.targets,2,'fixed5',3,f.matrix),/必要方向/);
});
test('null unreachable deliveries are excluded, malformed durations fail closed', () => {
  const {context,targets,matrix}=fixture();matrix.travel['101:1']=null;
  assert.deepEqual(problemFromMatrix(context,targets,2,'fixed5',3,matrix).targets[0].suppliers.map(s=>s.station_id),[103,102,104]);
  matrix.travel['101:1']={duration:-1,distance:10};
  assert.throws(()=>problemFromMatrix(context,targets,2,'fixed5',3,matrix),/無效時間/);
});


test('default fleet pool follows server alert decision before Top K; low-risk inclusion is explicit', () => {
  const {context,targets}=fixture();
  const alert={...targets[0],is_alert:true};
  const below={...targets[1],is_alert:false};
  // Alert flags are authoritative, including threshold-boundary and rounding decisions.
  assert.deepEqual(fleetTargets(context,[below,alert],1).map(s=>s.station_id),[alert.station_id]);
  assert.deepEqual(fleetTargets(context,[below,alert],30).map(s=>s.station_id),[alert.station_id]);
  assert.deepEqual(fleetTargets(context,[below],30),[]);
  assert.deepEqual(fleetTargets(context,[below,alert],30,true).map(s=>s.station_id),[below.station_id,alert.station_id]);
});
