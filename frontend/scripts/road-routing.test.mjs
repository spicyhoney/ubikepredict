import { test } from 'node:test';
import assert from 'node:assert/strict';
import { roadRouteUrl, parseRoadRoute, fetchRoadRoute } from '../app/road-routing.ts';
const points=[{station_id:1,latitude:25.1,longitude:121.4},{station_id:2,latitude:25.2,longitude:121.5},{station_id:3,latitude:25.3,longitude:121.6}];
function body(){return {code:'Ok',routes:[{geometry:{type:'LineString',coordinates:[[121.4,25.1],[121.41,25.11],[121.5,25.2]]},distance:1000,duration:180,legs:[{distance:400,duration:60},{distance:600,duration:120}]}],waypoints:points.map(()=>({distance:5}))};}
test('request preserves action order and uses longitude/latitude with bounded snapping',()=>{
 const url=new URL(roadRouteUrl(points,'https://routing.example/routed-car'));
 assert.ok(url.pathname.endsWith('121.400000,25.100000;121.500000,25.200000;121.600000,25.300000'));
 assert.equal(url.searchParams.get('radiuses'),'150;150;150');
 assert.equal(url.searchParams.get('geometries'),'geojson');
 assert.throws(()=>roadRouteUrl([{...points[0],latitude:null},points[1]],'https://routing.example'));
 assert.throws(()=>roadRouteUrl([points[0]],'https://routing.example'));
});
test('road geometry is converted to Leaflet order and preserves actual road bends/leg durations',()=>{
 const route=parseRoadRoute(body(),3);assert.deepEqual(route.coordinates,[[25.1,121.4],[25.11,121.41],[25.2,121.5]]);
 assert.equal(route.distance,1000);assert.deepEqual(route.legs.map(l=>l.duration),[60,120]);
});
test('no road, partial response, invalid coordinates and excessive snapping are not presented as routes',()=>{
 assert.throws(()=>parseRoadRoute({code:'NoRoute'},3));
 const partial=body();partial.routes[0].legs.pop();assert.throws(()=>parseRoadRoute(partial,3));
 const invalid=body();invalid.routes[0].geometry.coordinates[1]=[Infinity,25];assert.throws(()=>parseRoadRoute(invalid,3));
 const distant=body();distant.waypoints[0].distance=151;assert.throws(()=>parseRoadRoute(distant,3));
});
test('same-route requests share one fetch, distinct routes are paced, failures can be retried',async()=>{
 const original=global.fetch;const times=[];let fail=true;
 global.fetch=async url=>{times.push(Date.now());if(url.includes('retry.example')&&fail){fail=false;return {ok:false};}return {ok:true,json:async()=>body()};};
 try {
  const a=fetchRoadRoute(points,'https://cache.example');const b=fetchRoadRoute(points,'https://cache.example');
  assert.equal(a,b);await Promise.all([a,b]);assert.equal(times.length,1);
  await fetchRoadRoute([...points].reverse(),'https://cache.example');assert.ok(times[1]-times[0]>=1090);
  await assert.rejects(fetchRoadRoute(points,'https://retry.example'));
  await fetchRoadRoute(points,'https://retry.example');assert.equal(times.length,4);
 } finally {global.fetch=original;}
});

test('dense station markers remain separately clickable without moving isolated stations', async()=>{
 const {spreadMapMarkers}=await import('../app/map-layout.ts');
 const points=[{x:100,y:100,radius:15},{x:102,y:101,radius:15},{x:500,y:500,radius:9}];
 const result=spreadMapMarkers(points);
 assert.ok(Math.hypot(result[1].x-result[0].x,result[1].y-result[0].y)>=36.9);
 assert.deepEqual(result[2],points[2]);assert.deepEqual(points[0],{x:100,y:100,radius:15});
});
