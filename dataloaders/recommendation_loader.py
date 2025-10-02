import dspy
import polars as pl
from typing import List

class RecommendationLoader():
    def __init__(self, cfg):
        self.cfg = cfg
        self.train_path = cfg.train_path
        self.val_path = cfg.val_path
        self.test_path = cfg.test_path
        self.predict_path = cfg.predict_path

    def train_dataloader(self) -> List[dspy.Example]:
        df = pl.read_csv(self.train_path).limit(self.cfg.limit)
        trainset = []
        for row in df.iter_rows(named=True):
            trainset.append(dspy.Example(
                **row
            ).with_inputs("patient_question", "doctor_response"))
        return trainset
        
    def val_dataloader(self) -> List[dspy.Example]:
        df = pl.read_csv(self.val_path).limit(self.cfg.limit)
        valset = []
        for row in df.iter_rows(named=True):
            valset.append(dspy.Example(
                **row
            ).with_inputs("patient_question", "doctor_response"))
        return valset

    def test_dataloader(self) -> List[dspy.Example]:
        pass

    def predict_dataloader(self) -> List[dspy.Example]:
        predict_df = pl.read_csv(self.predict_path).limit(self.cfg.limit)
        predict_set = []
        for row in predict_df.iter_rows(named=True):
            predict_set.append(dspy.Example(
                **row
                # base_id=row['base_id'],
                # patient_question=row['patient_question'],
                # doctor_response=row['doctor_response'],
            ).with_inputs("patient_question", "doctor_response"))
        return predict_set
