import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from precision_search import prune
class SearchTests(unittest.TestCase):
    def test_noncontiguous_required_layers(self):
        result=prune(['a','b','c','d'],lambda xs: {'b','d'}<=set(xs),20)
        self.assertEqual(result['best'],['b','d'])
    def test_failed_seed_never_claims_success(self):
        self.assertIsNone(prune(['a'],lambda xs:False,3)['best'])
    def test_budget_keeps_only_verified_best(self):
        calls=[]
        def check(xs):calls.append(list(xs));return True
        r=prune(['a','b','c'],check,1)
        self.assertEqual(len(calls),1);self.assertEqual(r['best'],['a','b','c'])
        self.assertTrue(r['budget_exhausted'])
    def test_infrastructure_failure_propagates(self):
        def check(xs):raise RuntimeError('build failed')
        with self.assertRaises(RuntimeError):prune(['a'],check,2)

class SupervisorTests(unittest.TestCase):
    def test_infrastructure_error_stops_prepared_command(self):
        import tempfile,json
        from polygraphy_supervise import supervise
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory)
            command=[sys.executable,'-u','-c',"import time; print('ERROR: inference failed'); time.sleep(10)"]
            (work/'command.json').write_text(json.dumps({'command':command,'env':{}}))
            with self.assertRaises(RuntimeError):supervise(work)
            self.assertIn('ERROR: inference failed',(work/'search.log').read_text())
            with self.assertRaises(FileExistsError):supervise(work)

if __name__=='__main__':unittest.main()
