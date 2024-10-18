import csv
import numpy as np
import pandas as pd
import sys, os
import re
import ast
import datetime as dt
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

########################## GENERAL ##########################
def dataframe_from_csv(path, compression='gzip', header=0, index_col=0, chunksize=None):
    return pd.read_csv(path, compression=compression, header=header, index_col=index_col, chunksize=None)

def read_admissions_table(mimic4_path):
    admits = dataframe_from_csv(os.path.join(mimic4_path, 'core/admissions.csv.gz'))
    admits=admits.reset_index()
    admits = admits[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'ethnicity']]
    admits.admittime = pd.to_datetime(admits.admittime)
    admits.dischtime = pd.to_datetime(admits.dischtime)
    admits.deathtime = pd.to_datetime(admits.deathtime)
    return admits


def read_patients_table(mimic4_path):
    pats = dataframe_from_csv(os.path.join(mimic4_path, 'core/patients.csv.gz'))
    pats = pats.reset_index()
    pats = pats[['subject_id', 'gender','dod','anchor_age','anchor_year', 'anchor_year_group']]
    pats['yob']= pats['anchor_year'] - pats['anchor_age']
    #pats.dob = pd.to_datetime(pats.dob)
    pats.dod = pd.to_datetime(pats.dod)
    return pats


########################## DIAGNOSES ##########################
def read_diagnoses_icd_table(mimic4_path):
    diag = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/diagnoses_icd.csv.gz'))
    diag.reset_index(inplace=True)
    return diag


def read_d_icd_diagnoses_table(mimic4_path):
    d_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_diagnoses.csv.gz'))
    d_icd.reset_index(inplace=True)
    return d_icd[['icd_code', 'long_title']]


def read_diagnoses(mimic4_path):
    return read_diagnoses_icd_table(mimic4_path).merge(
        read_d_icd_diagnoses_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


def standardize_icd(mapping, df, root=False):
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe; adds column with converted ICD10 column"""

    def icd_9to10(icd):
        # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except:
            print("Error on code", icd)
            return np.nan

    # Create new column with original codes as default
    col_name = 'icd10_convert'
    if root: col_name = 'root_' + col_name
    df[col_name] = df['icd_code'].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    for code, group in df.loc[df.icd_version == 9].groupby(by='icd_code'):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code


########################## PROCEDURES ##########################
def read_procedures_icd_table(mimic4_path):
    proc = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/procedures_icd.csv.gz'))
    proc.reset_index(inplace=True)
    return proc


def read_d_icd_procedures_table(mimic4_path):
    p_icd = dataframe_from_csv(os.path.join(mimic4_path, 'hosp/d_icd_procedures.csv.gz'))
    p_icd.reset_index(inplace=True)
    return p_icd[['icd_code', 'long_title']]


def read_procedures(mimic4_path):
    return read_procedures_icd_table(mimic4_path).merge(
        read_d_icd_procedures_table(mimic4_path), how='inner', left_on=['icd_code'], right_on=['icd_code']
    )


########################## MAPPING ##########################
def read_icd_mapping(map_path):
    mapping = pd.read_csv(map_path, header=0, delimiter='\t')
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


########################## PREPROCESSING ##########################

#MANUALLY ADDED
def preproc_lab(module_path: str, adm_cohort_path: str) -> pd.DataFrame:
    """
    Preprocess laboratory event data for ICU patients.

    This function reads laboratory event data and admission cohort data, merges them, 
    and calculates the time elapsed from patient admission to the lab event timestamps.

    Parameters:
    -----------
    module_path : str
        Path to the CSV file containing laboratory event data (compressed as gzip).
        
    adm_cohort_path : str
        Path to the CSV file containing patient admission cohort data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the preprocessed laboratory event data with the following columns:
        
        - **subject_id**: Unique identifier for the patient.
        - **hadm_id**: Hospital admission ID, linking laboratory events to specific hospital admissions.
        - **itemid**: Identifier for the specific laboratory test, which can be cross-referenced with the items table for further details.
        - **charttime**: Timestamp indicating when the laboratory observation was recorded.
        - **storetime**: Timestamp indicating when the laboratory result was stored. This variable can be missing.
        - **valuenum**: Numeric value of the laboratory test result (if applicable). 
            -> This variable can be missing. Then maybe just the fact that the procedure was done is important. 
        - **valueuom**: Unit of measurement for the test result. 
            -> This variable can be missing. There could be no unit. 
        - **ref_range_lower**: Lower bound of the reference range for the laboratory test (if available). This variable can be missing.
        - **ref_range_upper**: Upper bound of the reference range for the laboratory test (if available). This variable can be missing.
        - **stay_id**: Identifier for the patient's stay in the hospital.
        - **intime**: Timestamp of the patient's admission to the ICU, allowing for temporal analysis of events.
        - **chart_hours_from_admit**: Time elapsed from the patient's admission to the lab event's charttime.
        - **store_hours_from_admit**: Time elapsed from the patient's admission to the lab event's storetime. This variable can be missing.

    Remarks:
    --------
    It is recommended to handle missing values appropriately based on the context of analysis.
    """
    adm = pd.read_csv(adm_cohort_path, usecols=['hadm_id', 'stay_id', 'intime'], parse_dates=['intime'])
    
    # Initialize an empty DataFrame to store the processed lab data
    df_lab = pd.DataFrame()
    
    # Read the lab events data in chunks
    chunksize = 1000000  # Adjust as necessary for your memory constraints
    for chunk in tqdm(pd.read_csv(module_path, compression='gzip', 
                                   usecols=["subject_id", 'hadm_id', 'itemid', 
                                             'charttime', 'storetime', 'valuenum',
                                             'valueuom', 'ref_range_lower', 
                                             'ref_range_upper'], 
                                   parse_dates=['charttime', 'storetime'], 
                                   chunksize=chunksize)):
        # Merge with admission data
        chunk_merged = chunk.merge(adm, on='hadm_id', how='inner')
        
        # Calculate time elapsed from admission
        chunk_merged['chart_hours_from_admit'] = chunk_merged['charttime'] - chunk_merged['intime']
        chunk_merged['store_hours_from_admit'] = chunk_merged['storetime'] - chunk_merged['intime']
        
        # Append the processed chunk to the main DataFrame
        df_lab = pd.concat([df_lab, chunk_merged], ignore_index=True)

    print("# of unique type of lab events: ", df_lab.itemid.nunique())
    print("# Admissions:  ", df_lab.stay_id.nunique())
    print("# Total rows",  df_lab.shape[0])
    
    return df_lab

def preproc_micro( module_path:str, adm_cohort_path:str) -> pd.DataFrame:
    """
    Preprocess microbiology test data for ICU patients.
    WARNING: Processing microbiology is difficult and currently not done correctly. 
    Each entry has a `test_itemid` which tells us that a certain test was performed. 
    The test is performed on a `spec_itemid` which tells us the specimen that was taken from the patient. 
    If the test found something, we also have a `org_name` which tells us which organism was found. 
    More organisms were however tested for an must be zero in that case. The list of organisms that 
    were tested by a specific test_itemid is not clear. there might be a table though. 


    Parameters:
    -----------
    module_path : str
        Path to the CSV file containing microbiology test data (compressed as gzip).
        
    adm_cohort_path : str
        Path to the CSV file containing patient admission cohort data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the preprocessed microbiology test data with the following columns:
        
        - **subject_id**: Unique identifier for the patient.
        - **hadm_id**: Hospital admission ID, linking microbiology tests to specific hospital admissions.
        - **charttime**: Timestamp indicating when the microbiology observation was recorded.
        - **spec_itemid**: Identifier for the specific specimen type associated with the test.
        - **storetime**: Timestamp indicating when the laboratory result was stored.
        - **test_itemid**: Identifier for the specific microbiology test, which can be cross-referenced with a test items table for further details.
        - **stay_id**: Identifier for the patient's stay in the hospital.
        - **intime**: Timestamp of the patient's admission to the ICU, allowing for temporal analysis of events.
        - **start_hours_from_admit**: Time elapsed from the patient's admission to the test event's charttime, represented as a timedelta.
        - **stop_hours_from_admit**: Time elapsed from the patient's admission to the test event's storetime, represented as a timedelta. This variable can be missing.

    Remarks:
    --------
    It is recommended to handle missing values appropriately based on the context of analysis.
    """

    print("Watch out: preprocessing microbiology data is tricky and needs to be looked at more carefully.")
    
    # Read admission data
    adm = pd.read_csv(adm_cohort_path, usecols=['hadm_id', 'stay_id', 'intime'], parse_dates=['intime'])
    
    # Initialize an empty DataFrame to store the processed microbiology data
    df_micro = pd.DataFrame()
    
    # Read the microbiology data in chunks
    chunksize = 1000000  # Adjust as necessary for your memory constraints
    for chunk in tqdm(pd.read_csv(module_path, compression='gzip', 
                                   usecols=['subject_id', 'hadm_id', # "chartdate", 
                                             'charttime', 'spec_itemid', # 'storedate',
                                             'storetime', 'test_itemid'], 
                                   parse_dates=['charttime', 'storetime'], 
                                   chunksize=chunksize)):
        # Merge with admission data
        chunk_merged = chunk.merge(adm, on='hadm_id', how='inner')
        
        # Calculate time elapsed from admission
        chunk_merged['start_hours_from_admit'] = chunk_merged['charttime'] - chunk_merged['intime']
        chunk_merged['stop_hours_from_admit'] = chunk_merged['storetime'] - chunk_merged['intime']
        
        # Append the processed chunk to the main DataFrame
        df_micro = pd.concat([df_micro, chunk_merged], ignore_index=True)

    print("# of unique type of micro events: ", df_micro.test_itemid.nunique())
    print("# Admissions:  ", df_micro.stay_id.nunique())
    print("# Total rows", df_micro.shape[0])

    return df_micro

def preproc_meds(module_path: str, adm_cohort_path: str) -> pd.DataFrame:
    """
    Preprocesses medication administration records for patients in a specified cohort.

    This function reads medication data and admission cohort information, merges them 
    to correlate medication administration with patient admission details, and computes 
    the elapsed time from admission for each medication event.

    Parameters:
    -----------
    module_path : str
        Path to the CSV file containing medication administration data.
    adm_cohort_path : str
        Path to the CSV file containing admission cohort data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing preprocessed medication administration records with the following columns:
        
        - `subject_id`: Identifier for the patient.
        - `stay_id`: Identifier for the patient's stay in the hospital.
        - `starttime`: Timestamp indicating when the medication administration began.
        - `endtime`: Timestamp indicating when the medication administration ended.
        - `itemid`: Identifier for the specific medication administered, which can be cross-referenced with an items table for details.
        - `amount`: Amount of medication administered.
        - `rate`: Rate of medication administration (e.g., dosage per hour).
            -> watch out: rate can be nan, then it's a one time med.
        - `orderid`: Identifier for the medication order.
        - `intime`: Timestamp of the patient's admission to the hospital.
        - `hadm_id`: Hospital admission ID, linking records to specific hospital admissions.
        - `start_hours_from_admit`: Time elapsed from the patient's admission to the start of medication administration, in hours.
        - `stop_hours_from_admit`: Time elapsed from the patient's admission to the end of medication administration, in hours.

    Notes:
    ------
    - The function drops any rows with missing values and prints the unique count of medication types,
      the number of admissions, and the total number of rows in the resulting DataFrame for diagnostics.
    - It is essential to ensure that the CSV files are in the expected format to avoid errors during processing.
    """

    adm = pd.read_csv(adm_cohort_path, usecols=['hadm_id', 'stay_id', 'intime'], parse_dates = ['intime'])
    med = pd.read_csv(module_path, compression='gzip', usecols=['subject_id', 'stay_id', 'itemid', 'starttime', 'endtime','rate','amount','orderid'], parse_dates = ['starttime', 'endtime'])
    med = med.merge(adm, left_on = 'stay_id', right_on = 'stay_id', how = 'inner')
    med['start_hours_from_admit'] = med['starttime'] - med['intime']
    med['stop_hours_from_admit'] = med['endtime'] - med['intime']
    
    #print(med.isna().sum())
    med=med.dropna()
    #med[['amount','rate']]=med[['amount','rate']].fillna(0)
    print("# of unique type of drug: ", med.itemid.nunique())
    print("# Admissions:  ", med.stay_id.nunique())
    print("# Total rows",  med.shape[0])
    
    return med
    
def preproc_proc(dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """
    Preprocesses procedure event data for patients in a specified cohort. 
    The function merges procedure data with the cohort information, ensuring that 
    only relevant observations are retained, and calculates the time from patient 
    admission to each procedure event.

    Parameters:
    -----------
    root_dir : str
        The root directory where the cohort file and relevant procedure event files are stored.
    
    dataset_path : str
        The path to the procedure event dataset file, which contains hospital procedure observations.
    
    cohort_path : str
        The path to the CSV file containing the cohort data. This file should include patient identifiers 
        and admission timestamps to filter the relevant procedure events.
    
    time_col : str
        The name of the column representing the timestamp of the procedure events (e.g., 'starttime').
    
    dtypes : dict
        A dictionary specifying the data types for columns in the procedure event dataset to optimize memory usage.
    
    usecols : list
        A list of columns to be read from the dataset, ensuring that only necessary data is loaded into memory.

    Returns:
    --------
    df_cohort : pd.DataFrame
        A DataFrame containing the preprocessed procedure events related to the specified cohort, structured as follows:
        
        - `stay_id`: Identifier for the patient's stay in the hospital.
        - `starttime`: Timestamp of when the procedure event occurred.
        - `itemid`: Identifier for the specific procedure, which can be cross-referenced with the items table for additional details.
            -> this is the important part. There is no associated value. 
        - `subject_id`: Identifier for the patient.
        - `hadm_id`: Hospital admission ID, linking the procedure to the specific hospital admission.
        - `intime`: Timestamp of the patient's admission to the hospital, allowing for temporal analysis.
        - `outtime`: Timestamp of when the patient was discharged from the hospital.
        - `event_time_from_admit`: The time elapsed from the patient's admission to the procedure event, facilitating 
          analyses related to the timing of interventions and patient responses.
    """


    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()
        #print(module.head())
        # Only consider values in our cohort
        cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['intime'])
        
        #print(module.head())
        #print(cohort.head())

        # merge module and cohort
        return module.merge(cohort[['subject_id','hadm_id','stay_id', 'intime','outtime']], how='inner', left_on='stay_id', right_on='stay_id')

    df_cohort = merge_module_cohort()
    df_cohort['event_time_from_admit'] = df_cohort[time_col] - df_cohort['intime']
    
    df_cohort=df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.dropna().nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort

def preproc_out(dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """
    Extracts and preprocesses hospital observations related to a specified cohort from the provided dataset.
    This function is designed to efficiently read and transform the data, optimizing memory usage.

    The output DataFrame includes observations for patients within the defined cohort, with timestamps of each 
    observation relative to the admission time. The data can be used to analyze clinical measurements, 
    medication administration, and other relevant hospital events.

    Parameters
    ----------
    dataset_path : str
        Path to the CSV file containing hospital observation data, which includes relevant measurements and their timestamps.
    
    cohort_path : str
        Path to the cohort file containing patient stay information, including admission and discharge times.

    time_col : str
        The name of the column in the dataset that represents the timestamp of each observation.

    dtypes : dict
        A dictionary specifying the data types for columns in the dataset to optimize memory usage during loading.

    usecols : list
        A list of columns to be read from the dataset, allowing for selective loading of relevant data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the preprocessed hospital observations for the specified cohort. The output includes:
        
        - `subject_id`: Unique identifier for the patient.
        - `hadm_id`: Hospital admission ID.
        - `stay_id`: Identifier for the patient's stay in the hospital.
        - `caregiver_id`: Identifier for the healthcare provider who recorded the observation.
        - `charttime`: Timestamp of when the observation was recorded.
        - `storetime`: Timestamp of when the observation was stored in the database.
        - `itemid`: Identifier for the specific observation or measurement.
        - `value`: The observed measurement or recorded value.
        - `valueuom`: The unit of measurement for the recorded value (e.g., mL).
        - `intime`: The timestamp of patient admission.
        - `outtime`: The timestamp of patient discharge.
        - `event_time_from_admit`: The time elapsed from admission to the event, calculated as the difference 
          between the observation time and the admission time (`intime`).

    Notes
    -----
    - Users can look up specific features and their descriptions by referencing the `itemid` in the corresponding 
      items table to understand what type of observations are being analyzed.
    - The function merges the observation data with the cohort information, ensuring that only events related to 
      patients in the specified cohort are included.
    - The resulting DataFrame is sorted by `subject_id` for consistency in analysis and interpretation.
    """


    def merge_module_cohort() -> pd.DataFrame:
        """Gets the initial module data with patients anchor year data and only the year of the charttime"""
        
        # read module w/ custom params
        module = pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col]).drop_duplicates()
        #print(module.head())
        # Only consider values in our cohort
        cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['intime'])
        
        #print(module.head())
        #print(cohort.head())

        # merge module and cohort
        return module.merge(cohort[['stay_id', 'intime','outtime']], how='inner', left_on='stay_id', right_on='stay_id')

    df_cohort = merge_module_cohort()
    df_cohort['event_time_from_admit'] = df_cohort[time_col] - df_cohort['intime']
    df_cohort=df_cohort.dropna()
    # Print unique counts and value_counts
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id

    return df_cohort

def preproc_chart(dataset_path: str, cohort_path: str, time_col: str, dtypes: dict, usecols: list) -> pd.DataFrame:
    """
    Preprocesses chart event data from the MIMIC-IV dataset for a specified patient cohort.

    Parameters:
    -----------
    dataset_path : str
        The file path to the chart events data (CSV format) from the MIMIC-IV dataset.
    
    cohort_path : str
        The file path to the cohort data (pickled) containing relevant patient stay information.
    
    time_col : str
        The name of the column representing the time of the chart event.
    
    dtypes : dict
        A dictionary specifying the data types for the columns to be read in the dataset.
    
    usecols : list
        A list of columns to be included when reading the dataset to optimize memory usage.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the preprocessed chart event data for patients in the specified cohort.
        Columns include:
        - `stay_id`: Identifier for the patient's stay in the hospital.
        - `itemid`: Identifier for specific chart observations, which can be cross-referenced with the items table for further detail.
        - `valuenum`: Numeric value of the observation (e.g., vital signs, lab results).
        - `valueuom`: Units of measurement for the `valuenum` (e.g., mmHg, bpm).
        - `event_time_from_admit`: The time elapsed from the patient's admission to the observation event, facilitating 
          analyses related to the timing of care and interventions.
    """
    
    # Only consider values in our cohort
    cohort = pd.read_csv(cohort_path, compression='gzip', parse_dates = ['intime'])
    df_cohort=pd.DataFrame()
        # read module w/ custom params
    chunksize = 10000000
    count=0
    nitem=[]
    nstay=[]
    nrows=0

    for chunk in tqdm(pd.read_csv(dataset_path, compression='gzip', usecols=usecols, dtype=dtypes, parse_dates=[time_col],chunksize=chunksize)):
        #print(chunk.head())
        count=count+1
        #chunk['valuenum']=chunk['valuenum'].fillna(0)
        chunk=chunk.dropna(subset=['valuenum'])
        chunk_merged=chunk.merge(cohort[['stay_id', 'intime']], how='inner', left_on='stay_id', right_on='stay_id')
        chunk_merged['event_time_from_admit'] = chunk_merged[time_col] - chunk_merged['intime']
        
        del chunk_merged[time_col] 
        del chunk_merged['intime']
        # chunk_merged=chunk_merged.dropna()
        chunk_merged=chunk_merged.drop_duplicates()
        if df_cohort.empty:
            df_cohort=chunk_merged
        else:
            df_cohort=df_cohort.append(chunk_merged, ignore_index=True)
        
        
#         nitem.append(chunk_merged.itemid.dropna().unique())
#         nstay=nstay.append(chunk_merged.stay_id.unique())
#         nrows=nrows+chunk_merged.shape[0]
                
        
    
    # Print unique counts and value_counts
#     print("# Unique Events:  ", len(set(nitem)))
#     print("# Admissions:  ", len(set(nstay)))
#     print("Total rows", nrows)
    print("# Unique Events:  ", df_cohort.itemid.nunique())
    print("# Admissions:  ", df_cohort.stay_id.nunique())
    print("Total rows", df_cohort.shape[0])

    # Only return module measurements within the observation range, sorted by subject_id
    return df_cohort

def preproc_icd_module( module_path:str, adm_cohort_path:str, icd_map_path=None, map_code_colname=None, only_icd10=True) -> pd.DataFrame:
    """
    Preprocesses a module dataset containing ICD (International Classification of Diseases) codes by merging it with an admission cohort and optionally mapping ICD-9 codes to ICD-10 codes using a mapping table.
    Note that we could additionally keep the order of diagnoses, but not the exact timepoint
    
    This function handles the following tasks:
    - Merges the module dataset with an admission cohort based on hospital admission IDs (`hadm_id`).
    - Optionally maps ICD-9 codes to ICD-10 codes using a provided mapping table.
    - Converts ICD-9 codes to their root ICD-10 codes and adds new columns with these conversions.
    - Filters the final output to only contain ICD-10 codes if specified.

    Parameters
    ----------
    root_dir : str
        Root directory where the datasets are stored.
    module_path : str
        Path to the CSV file containing the module data with ICD codes (ICD-9 and/or ICD-10).
    adm_cohort_path : str
        Path to the admission cohort file containing the hospital admission IDs (`hadm_id`), stay IDs (`stay_id`), and labels.
    icd_map_path : str, optional
        Path to the mapping file that maps ICD-9 codes to ICD-10 codes. If None, no mapping is applied.
    map_code_colname : str, optional
        The column name in the mapping file that contains the ICD-9 codes to be mapped.
    only_icd10 : bool, optional, default=True
        Whether to filter the final dataset to only include ICD-10 codes.

    Returns
    -------
    pd.DataFrame
        The preprocessed module dataset, with additional columns for converted ICD codes if a mapping is provided.
        It contains:
        - **hadm_id**: Unique identifier for each hospital admission.
        - **icd_code**: The original ICD code (ICD-9 or ICD-10) from the module dataset.
        - **root_icd10_convert**: The ICD-10 code obtained from the mapping of ICD-9 codes, or the original ICD-10 code if no mapping is applied.
        - **root**: The root ICD-10 code derived from `root_icd10_convert`, which consists of the first three characters of the ICD-10 code.

    Notes
    -----
    - The mapping table, if provided, should include at least two columns: one for the ICD-9 codes and one for the ICD-10 codes.
    - If `only_icd10=True`, the output DataFrame will contain a root-level ICD-10 code column.
    """
    
    def get_module_cohort(module_path:str, cohort_path:str):
        module = pd.read_csv(module_path, compression='gzip', header=0)
        adm_cohort = pd.read_csv(adm_cohort_path, compression='gzip', header=0)
        #print(module.head())
        #print(adm_cohort.head())
        
        #adm_cohort = adm_cohort.loc[(adm_cohort.timedelta_years <= 6) & (~adm_cohort.timedelta_years.isna())]
        return module.merge(adm_cohort[['hadm_id', 'stay_id', 'label']], how='inner', left_on='hadm_id', right_on='hadm_id')

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a modulenosis dataframe; adds column with converted ICD10 column"""
        
        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return mapping.loc[mapping[map_code_colname] == icd].icd10cm.iloc[0]
            except:
                #print("Error on code", icd)
                return np.nan

        # Create new column with original codes as default
        col_name = 'icd10_convert'
        if root: col_name = 'root_' + col_name
        df[col_name] = df['icd_code'].values

        # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
        for code, group in df.loc[df.icd_version == 9].groupby(by='icd_code'):
            new_code = icd_9to10(code)
            for idx in group.index.values:
                # Modify values of original df at the indexes in the groups
                df.at[idx, col_name] = new_code

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df['root'] = df[col_name].apply(lambda x: x[:3] if type(x) is str else np.nan)

    module = get_module_cohort(module_path, adm_cohort_path)
    #print(module.shape)
    #print(module['icd_code'].nunique())

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        #print(icd_map)
        standardize_icd(icd_map, module, root=True)
        print("# unique ICD-9 codes",module[module['icd_version']==9]['icd_code'].nunique())
        print("# unique ICD-10 codes",module[module['icd_version']==10]['icd_code'].nunique())
        print("# unique ICD-10 codes (After converting ICD-9 to ICD-10)",module['root_icd10_convert'].nunique())
        print("# unique ICD-10 codes (After clinical gruping ICD-10 codes)",module['root'].nunique())
        print("# Admissions:  ", module.stay_id.nunique())
        print("Total rows", module.shape[0])
    return module


def pivot_cohort(df: pd.DataFrame, prefix: str, target_col:str, values='values', use_mlb=False, ohe=True, max_features=None):
    """Pivots long_format data into a multiindex array:
                                            || feature 1 || ... || feature n ||
        || subject_id || label || timedelta ||
    """
    aggfunc = np.mean
    pivot_df = df.dropna(subset=[target_col])

    if use_mlb:
        mlb = MultiLabelBinarizer()
        output = mlb.fit_transform(pivot_df[target_col].apply(ast.literal_eval))
        output = pd.DataFrame(output, columns=mlb.classes_)
        if max_features:
            top_features = output.sum().sort_values(ascending=False).index[:max_features]
            output = output[top_features]
        pivot_df = pd.concat([pivot_df[['subject_id', 'label', 'timedelta']].reset_index(drop=True), output], axis=1)
        pivot_df = pd.pivot_table(pivot_df, index=['subject_id', 'label', 'timedelta'], values=pivot_df.columns[3:], aggfunc=np.max)
    else:
        if max_features:
            top_features = pd.Series(pivot_df[['subject_id', target_col]].drop_duplicates()[target_col].value_counts().index[:max_features], name=target_col)
            pivot_df = pivot_df.merge(top_features, how='inner', left_on=target_col, right_on=target_col)
        if ohe:
            pivot_df = pd.concat([pivot_df.reset_index(drop=True), pd.Series(np.ones(pivot_df.shape[0], dtype=int), name='values')], axis=1)
            aggfunc = np.max
        pivot_df = pivot_df.pivot_table(index=['subject_id', 'label', 'timedelta'], columns=target_col, values=values, aggfunc=aggfunc)

    pivot_df.columns = [prefix + str(i) for i in pivot_df.columns]
    return pivot_df