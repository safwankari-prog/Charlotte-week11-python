"""Smoke checks: expected to fail until you implement starter tasks."""
import unittest
from pathlib import Path
from preprocess_iris import prepare_dataset

class StudentSmokeTest(unittest.TestCase):
 def test_usable_outputs(self):
  _tr, train, test, report = prepare_dataset(Path(__file__).with_name("iris_preprocessing_case_PUBLIC_DERIVED.csv"))
  self.assertEqual(len(train),120)
  self.assertEqual(len(test),30)
  self.assertEqual(list(train.columns),list(test.columns))
  self.assertNotIn("species",train.columns)
  self.assertNotIn("sample_id",train.columns)

if __name__ == '__main__': unittest.main()
