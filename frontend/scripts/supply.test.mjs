import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import ts from 'typescript';
const source = fs.readFileSync(new URL('../app/supply-routing.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText.replace("'./road-routing'", JSON.stringify(new URL('../app/road-routing.ts', import.meta.url).href));
const {stockTarget,supplyCandidates,parseSupplyTable}=await import('data:text/javascript;base64,'+Buffer.from(compiled).toString('base64'));
const station=(id,bikes,capacity=20,docks=capacity-bikes)=>({station_id:id,station_name:String(id),district:'test',latitude:25,longitude:121,bikes,capacity,docks});
test('capacity floor and donor reserve are enforced without changing snapshot',()=>{
 const target=station(1,0,60), donor=station(2,25,60), small=station(3,8,20);
 const context={stations:[target,donor,small],neighborhoods:{1:[2,3]}};
 const hybrid=supplyCandidates(context,1,'hybrid20');assert.equal(hybrid.need,12);assert.equal(hybrid.candidates.length,1);assert.equal(hybrid.candidates[0].after,13);assert.equal(hybrid.candidates[0].reserve,12);
 assert.equal(stockTarget(target,'fixed5'),5);assert.equal(donor.bikes,25);
 assert.equal(supplyCandidates({stations:[station(1,0,20,0)],neighborhoods:{1:[]}},1,'fixed5'),null);
});
test('road ranking uses directed source travel time, filters unreachable, and breaks ties deterministically',()=>{
 const candidates=[station(1,10),station(2,10),station(3,10)].map(s=>({...s,reserve:5,available:5,after:5}));
 const body={code:'Ok',durations:[[200],[100],[null]],distances:[[500],[900],[null]],sources:[{distance:1},{distance:2},{distance:3}],destinations:[{distance:1}]};
 const ranked=parseSupplyTable(body,candidates);assert.deepEqual(ranked.map(s=>s.station_id),[2,1]);
 assert.throws(()=>parseSupplyTable({...body,durations:[[100]]},candidates));
 assert.throws(()=>parseSupplyTable({...body,destinations:[{distance:151}]},candidates));
 assert.throws(()=>parseSupplyTable({...body,durations:[[NaN],[100],[null]]},candidates));
});
