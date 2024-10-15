

import os
import pandas as pd

class DynamicDataLoader:
    def __init__(self, data_icu, labels_path='./data/csv/labels.csv', data_dir='./data/csv/', num_patients=None):
        self.data_icu = data_icu
        self.labels_path = labels_path
        self.data_dir = data_dir
        self.num_patients = num_patients
    
    def load_data(self):
        # Load labels
        labels = pd.read_csv(self.labels_path, header=0)
        
        # Get a list of patient IDs
        if self.data_icu:
            patient_ids = labels['stay_id']
        else:
            patient_ids = labels['hadm_id']
        
        if self.num_patients is not None:
            patient_ids = patient_ids[:self.num_patients]  # Limit to specified number of patients
        
        X_df, y_df = pd.DataFrame(), pd.DataFrame()

        for patient_id in patient_ids:
            dynamic_path = os.path.join(self.data_dir, f'{patient_id}/dynamic.csv')
            static_path = os.path.join(self.data_dir, f'{patient_id}/static.csv')
            demo_path = os.path.join(self.data_dir, f'{patient_id}/demo.csv')
            
            # Load dynamic, static, and demographic data
            dyn = pd.read_csv(dynamic_path, header=[0, 1])
            stat = pd.read_csv(static_path, header=[0, 1])
            demo = pd.read_csv(demo_path, header=0)
            
            # Example of how you might combine these; modify based on actual data structure
            if X_df.empty:
                X_df = pd.concat([dyn, stat, demo], axis=1)
            else:
                X_df = pd.concat([X_df, pd.concat([dyn, stat, demo], axis=1)], axis=0)
            
            # Append labels
            y = labels[labels['stay_id'] == patient_id]['label'] if self.data_icu else labels[labels['hadm_id'] == patient_id]['label']
            y_df = pd.concat([y_df, y])

        print("X_df shape:", X_df.shape)
        print("y_df shape:", y_df.shape)

        return X_df, y_df