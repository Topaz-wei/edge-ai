import sys
from pathlib import Path
import unittest
import tempfile
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import precision_check as pc

class CheckTests(unittest.TestCase):
    def sample(self):
        a=np.zeros((1,84,8400),np.float32)
        a[0,:4,0]=[100,100,50,80];a[0,4,0]=.8
        return a
    def test_identity_and_duplicate_nms(self):
        a=self.sample(); a[0,:,1]=a[0,:,0];a[0,4,1]=.75
        self.assertTrue(pc.evaluate(a,a)['pass'])
        self.assertEqual(len(pc.detections(a,.7,.7)),1)
    def test_score_drop_and_missing_detection_fail(self):
        a=self.sample();b=a.copy();b[0,4,0]=.65
        self.assertFalse(pc.evaluate(a,b)['pass'])
    def test_box_shift_fails(self):
        a=self.sample();b=a.copy();b[0,0,0]+=200
        self.assertFalse(pc.evaluate(a,b)['pass'])
    def test_class_argmax_precedes_person_filter(self):
        a=self.sample();b=a.copy();b[0,5,0]=.9
        self.assertFalse(pc.evaluate(a,b)['pass'])
    def test_extra_detection_fails(self):
        a=self.sample();b=a.copy();b[0,:4,3]=[400,400,50,80];b[0,4,3]=.9
        self.assertFalse(pc.evaluate(a,b)['pass'])
    def test_invalid_inputs_rejected(self):
        a=self.sample()
        for b in [np.zeros((1,84,1)),np.full_like(a,np.nan)]:
            with self.assertRaises(ValueError):pc.evaluate(a,b)
        with self.assertRaises(ValueError):pc.evaluate(np.zeros_like(a),a)
        with self.assertRaises(ValueError):pc.evaluate(a,a,score_error=-1)
    def test_report_cannot_overwrite_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'reference.bin';self.sample().tofile(source)
            before=source.read_bytes()
            command=[sys.executable,str(Path(pc.__file__)), '--reference',str(source),
                     '--output',str(source),'--report',str(source)]
            result=subprocess.run(command,capture_output=True)
            self.assertEqual(result.returncode,2)
            self.assertEqual(source.read_bytes(),before)
    def test_matching_is_one_to_one(self):
        self.assertFalse(pc.match_detections([(0,.8,[0,0,20,20]),(0,.8,[0,0,20,20])],[(0,.8,[0,0,20,20])],.85,.05))
if __name__=='__main__':unittest.main()
