import unittest
from backend.supply import supply_context, SNAPSHOT_TIME, _snapshot

class SupplyTest(unittest.TestCase):
    def test_time_boundary(self):
        self.assertIsNone(supply_context('2026-06-29T09:30:00+08:00','empty',[1526]))
        self.assertIsNone(supply_context(SNAPSHOT_TIME,'full_dock',[1526]))
    def test_snapshot_and_neighborhood(self):
        rows=_snapshot()
        self.assertEqual(len(rows),1562)
        target=next(s for s in rows if s['station_name']=='河邊北街100巷口')
        c=supply_context(SNAPSHOT_TIME,'empty',[target['station_id']])
        ids=c['neighborhoods'][str(target['station_id'])]
        self.assertEqual(len(ids),20)
        self.assertNotIn(target['station_id'],ids)
        self.assertEqual(len(c['stations']),21)
        self.assertTrue(all(set(s)=={'station_id','station_name','district','latitude','longitude','capacity','bikes','docks'} for s in c['stations']))
    def test_unavailable_stations_excluded(self):
        rows=_snapshot();bad=[s['station_id'] for s in rows if s['bikes']==0 and s['docks']==0]
        self.assertEqual(len(bad),7)
        self.assertEqual(supply_context(SNAPSHOT_TIME,'empty',bad)['neighborhoods'],{})

if __name__=='__main__':unittest.main()
