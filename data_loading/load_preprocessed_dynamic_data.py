

import os
import pandas as pd

from tqdm import tqdm

class DynamicDataLoader:
    def __init__(self, data_icu, root_dir,cohort_output,  data_dir='./data/csv/', num_stays=None):
        self.data_icu = data_icu
        # self.labels_path = labels_path
        self.data_dir = data_dir
        self.num_stays = num_stays
        self.root_dir = root_dir
        self.cohort_output = cohort_output


    def load_data(self):
        # Load labels
        # labels = pd.read_csv(self.labels_path, header=0)
        # labels = pd.read_csv(self.labels_path, header=0)
        # load admission data
        data = pd.read_csv(self.root_dir + f"/data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)

        if self.data_icu:
            ids = data['stay_id']
        else:
            ids = data['hadm_id']
        
        if self.num_stays is not None:
            ids = ids[:self.num_stays]
        # if self.num_patients is not None:
        #     patient_ids = patient_ids[:self.num_patients]  # Limit to specified number of patients
        
        temporal_df = pd.DataFrame()
        static_df = pd.DataFrame()  # DataFrame for static and demo data

        skipped_count = 0

        for id in tqdm(ids, desc="Processing IDs", unit="id"):
            dynamic_path = os.path.join(self.data_dir, f'{id}/dynamic.csv')
            static_path = os.path.join(self.data_dir, f'{id}/static.csv')
            demo_path = os.path.join(self.data_dir, f'{id}/demo.csv')
            
            # Check if each file exists
            if not os.path.exists(dynamic_path) or not os.path.exists(static_path) or not os.path.exists(demo_path):
                skipped_count += 1
                print(f"Skipping id: {id} (missing file)")
                continue

            # Load dynamic, static, and demographic data
            dyn = pd.read_csv(dynamic_path, header=[0, 1])
            stat = pd.read_csv(static_path, header=[0, 1])
            demo = pd.read_csv(demo_path, header=0)
            
            # Add patient ID to all dataframes
            dyn['id'] = id
            stat['id'] = id
            demo['id'] = id

            # Add timepoint column to dynamic data
            dyn = dyn.reset_index()  # Reset index to make the timepoint accessible
            dyn['time'] = dyn.index  # Assuming each row corresponds to a different timepoint

            # Combine dynamic data for this patient
            patient_data = dyn

            # Append patient's dynamic data to X_df
            if temporal_df.empty:
                temporal_df = patient_data
            else:
                temporal_df = pd.concat([temporal_df, patient_data], axis=0)

            # Prepare static and demo data for a separate DataFrame
            # Combine static and demo data
            # static = stat
            # combined_static_demo = pd.concat([stat, demo], axis=1)
            # Keep only the patient_id column and values, dropping any time-related indices
            # combined_static_demo['id'] = patient_id
            # combined_static_demo = combined_static_demo[[col for col in combined_static_demo.columns if col != 'index']]

            # Append to static_demo_df
            if static_df.empty:
                static_df = pd.concat([stat, demo], axis=1) 
            else:
                stat = pd.concat([stat, demo], axis=1) 
                static_df = pd.concat([static_df, stat ], axis=0)

            # # Append labels
            # y = labels[labels['stay_id'] == id]['label'] if self.data_icu else labels[labels['hadm_id'] == patient_id]['label']
            # y_df = pd.concat([y_df, y])

        # Drop the 'index' column from X_df if it exists
        temporal_df       = temporal_df.drop('index', axis=1, errors='ignore')
        temporal_df['id'] = temporal_df['id'].astype(int)
        static_df['id']   = static_df['id'].astype(int)

        print("temporal_df shape:", temporal_df.shape)
        # print("y_df shape:", y_df.shape)
        print("static_df shape:", static_df.shape)

        print("Loaded data for {}/{} patients".format(len(ids) - skipped_count, len(ids)))

        return temporal_df, static_df  # Return the new DataFrame
    
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
