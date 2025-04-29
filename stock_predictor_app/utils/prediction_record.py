from datetime import datetime
import json
import os
from pathlib import Path

class PredictionRecord:
    def __init__(self):
        self.records_dir = Path.home() / '.stock_predictor' / 'predictions'
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.records_dir / 'prediction_records.json'
        self.load_records()

    def load_records(self):
        if self.records_file.exists():
            with open(self.records_file, 'r') as f:
                self.records = json.load(f)
        else:
            self.records = []
            self.save_records()

    def save_records(self):
        with open(self.records_file, 'w') as f:
            json.dump(self.records, f, indent=4)

    def add_record(self, symbol, model_name, days, prediction_data, mse=None):
        record = {
            'id': len(self.records) + 1,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': symbol,
            'model': model_name,
            'days': days,
            'prediction': prediction_data.tolist() if hasattr(prediction_data, 'tolist') else prediction_data,
            'mse': mse,
            'status': 'Completed'
        }
        self.records.append(record)
        self.save_records()
        return record

    def get_records(self, limit=None):
        records = sorted(self.records, key=lambda x: x['timestamp'], reverse=True)
        return records[:limit] if limit else records

    def get_record_by_id(self, record_id):
        for record in self.records:
            if record['id'] == record_id:
                return record
        return None 