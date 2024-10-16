

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
        static_demo_df = pd.DataFrame()  # DataFrame for static and demo data

        for patient_id in patient_ids:
            dynamic_path = os.path.join(self.data_dir, f'{patient_id}/dynamic.csv')
            static_path = os.path.join(self.data_dir, f'{patient_id}/static.csv')
            demo_path = os.path.join(self.data_dir, f'{patient_id}/demo.csv')
            
            # Load dynamic, static, and demographic data
            dyn = pd.read_csv(dynamic_path, header=[0, 1])
            stat = pd.read_csv(static_path, header=[0, 1])
            demo = pd.read_csv(demo_path, header=0)
            
            # Add patient ID to all dataframes
            dyn['id'] = patient_id
            stat['id'] = patient_id
            demo['id'] = patient_id

            # Add timepoint column to dynamic data
            dyn = dyn.reset_index()  # Reset index to make the timepoint accessible
            dyn['time'] = dyn.index  # Assuming each row corresponds to a different timepoint

            # Combine dynamic data for this patient
            patient_data = dyn

            # Append patient's dynamic data to X_df
            if X_df.empty:
                X_df = patient_data
            else:
                X_df = pd.concat([X_df, patient_data], axis=0)

            # Prepare static and demo data for a separate DataFrame
            # Combine static and demo data
            combined_static_demo = stat
            # combined_static_demo = pd.concat([stat, demo], axis=1)
            # Keep only the patient_id column and values, dropping any time-related indices
            # combined_static_demo['id'] = patient_id
            # combined_static_demo = combined_static_demo[[col for col in combined_static_demo.columns if col != 'index']]

            # Append to static_demo_df
            if static_demo_df.empty:
                static_demo_df = combined_static_demo
            else:
                static_demo_df = pd.concat([static_demo_df, combined_static_demo], axis=0)

            # Append labels
            y = labels[labels['stay_id'] == patient_id]['label'] if self.data_icu else labels[labels['hadm_id'] == patient_id]['label']
            y_df = pd.concat([y_df, y])

        # Drop the 'index' column from X_df if it exists
        X_df = X_df.drop('index', axis=1, errors='ignore')
        X_df['id']   =  X_df['id'].astype(int)
        static_demo_df['id']   =  static_demo_df['id'].astype(int)

        print("X_df shape:", X_df.shape)
        print("y_df shape:", y_df.shape)
        print("static_demo_df shape:", static_demo_df.shape)

        return X_df, y_df, static_demo_df  # Return the new DataFrame
    
    # def load_data(self):
    #     # Load labels
    #     labels = pd.read_csv(self.labels_path, header=0)
        
    #     # Get a list of patient IDs
    #     if self.data_icu:
    #         patient_ids = labels['stay_id']
    #     else:
    #         patient_ids = labels['hadm_id']
        
    #     if self.num_patients is not None:
    #         patient_ids = patient_ids[:self.num_patients]  # Limit to specified number of patients
        
    #     X_df, y_df = pd.DataFrame(), pd.DataFrame()

    #     for patient_id in patient_ids:
    #         dynamic_path = os.path.join(self.data_dir, f'{patient_id}/dynamic.csv')
    #         static_path = os.path.join(self.data_dir, f'{patient_id}/static.csv')
    #         demo_path = os.path.join(self.data_dir, f'{patient_id}/demo.csv')
            
    #         # Load dynamic, static, and demographic data
    #         dyn = pd.read_csv(dynamic_path, header=[0, 1])
    #         stat = pd.read_csv(static_path, header=[0, 1])
    #         demo = pd.read_csv(demo_path, header=0)
            
    #         # Add patient ID to all dataframes
    #         dyn['patient_id'] = patient_id
    #         stat['patient_id'] = patient_id
    #         demo['patient_id'] = patient_id

    #         # Add timepoint column to dynamic data
    #         dyn = dyn.reset_index()  # Reset index to make the timepoint accessible
    #         dyn['timepoint'] = dyn.index  # Assuming each row corresponds to a different timepoint

    #         # Combine dynamic, static, and demographic data for this patient
    #         # patient_data = pd.concat([dyn, stat, demo], axis=1)
    #         patient_data = dyn

    #         # Append patient's data to X_df
    #         if X_df.empty:
    #             X_df = patient_data
    #         else:
    #             X_df = pd.concat([X_df, patient_data], axis=0)
            
    #         # Append labels
    #         y = labels[labels['stay_id'] == patient_id]['label'] if self.data_icu else labels[labels['hadm_id'] == patient_id]['label']
    #         y_df = pd.concat([y_df, y])

    #     X_df = X_df.drop('index', axis=1)

    #     print("X_df shape:", X_df.shape)
    #     print("y_df shape:", y_df.shape)

    #     return X_df, y_df


    # def load_data(self):
    #     # Load labels
    #     labels = pd.read_csv(self.labels_path, header=0)
        
    #     # Get a list of patient IDs
    #     if self.data_icu:
    #         patient_ids = labels['stay_id']
    #     else:
    #         patient_ids = labels['hadm_id']
        
    #     if self.num_patients is not None:
    #         patient_ids = patient_ids[:self.num_patients]  # Limit to specified number of patients
        
    #     X_df, y_df = pd.DataFrame(), pd.DataFrame()

    #     for patient_id in patient_ids:
    #         dynamic_path = os.path.join(self.data_dir, f'{patient_id}/dynamic.csv')
    #         static_path = os.path.join(self.data_dir, f'{patient_id}/static.csv')
    #         demo_path = os.path.join(self.data_dir, f'{patient_id}/demo.csv')
            
    #         # Load dynamic, static, and demographic data
    #         dyn = pd.read_csv(dynamic_path, header=[0, 1])
    #         stat = pd.read_csv(static_path, header=[0, 1])
    #         demo = pd.read_csv(demo_path, header=0)
            
    #         # Add patient ID to all dataframes
    #         dyn['patient_id'] = patient_id
    #         stat['patient_id'] = patient_id
    #         demo['patient_id'] = patient_id

    #         # Combine dynamic, static, and demographic data for this patient
    #         # patient_data = pd.concat([dyn, stat, demo], axis=1)
    #         patient_data  = dyn

    #         # Append patient's data to X_df
    #         if X_df.empty:
    #             X_df = patient_data
    #         else:
    #             X_df = pd.concat([X_df, patient_data], axis=0)
            
    #         # Append labels
    #         y = labels[labels['stay_id'] == patient_id]['label'] if self.data_icu else labels[labels['hadm_id'] == patient_id]['label']
    #         y_df = pd.concat([y_df, y])

    #     print("X_df shape:", X_df.shape)
    #     print("y_df shape:", y_df.shape)

    #     return X_df, y_df
