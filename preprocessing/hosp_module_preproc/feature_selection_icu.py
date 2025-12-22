import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


# if not os.path.exists("./data/features"):
#     os.makedirs("./data/features")
# if not os.path.exists("./data/features/chartevents"):
#     os.makedirs("./data/features/chartevents")

def feature_icu(cohort_output: str, root_dir: str, version_path: str, save_path: str, diag_flag: bool = True, 
                out_flag: bool = True, chart_flag: bool = True, proc_flag: bool = True, 
                med_flag: bool = True, lab_flag: bool = True, micro_flag: bool = True):
    """
    Extracts and preprocesses various types of ICU-related clinical data from the MIMIC-IV dataset based on the provided cohort.
    Returns the preprocessed data in a dictionary and optionally saves the data to the specified path.

    Parameters:
    -----------
    cohort_output : str
        The name of the cohort file (e.g., 'cohort_icu_readmission_30_N18_N18') that contains patient stay information. 
        This cohort is used to filter the data for ICU stays related to specific criteria, such as ICU readmission 
        within 30 days, and other relevant conditions for the analysis.
    
    root_dir : str
        The root directory where the cohort file and relevant MIMIC-IV data files are stored.
    
    version_path : str
        The path to the versioned MIMIC-IV dataset (e.g., ICU or hospital files).
    
    save_path : str
        The directory where the preprocessed files will be saved. Filenames are predefined for each type of data.
    
    diag_flag : bool, optional
        If True, extract and preprocess diagnosis data (default is True).
    
    out_flag : bool, optional
        If True, extract and preprocess output events data (default is True).
    
    chart_flag : bool, optional
        If True, extract and preprocess chart events data (default is True).
    
    proc_flag : bool, optional
        If True, extract and preprocess procedure events data (default is True).
    
    med_flag : bool, optional
        If True, extract and preprocess medication data (default is True).
    
    lab_flag : bool, optional
        If True, extract and preprocess laboratory events data (default is True).
    
    micro_flag : bool, optional
        If True, extract and preprocess microbiology events data (default is True).

    Returns:
    --------
    data_dict : dict
        A dictionary containing the preprocessed dataframes for each type of ICU-related data. 
        Keys are ['diagnosis', 'output', 'chart', 'procedures', 'medications', 'lab', 'microbiology'].

        - **diagnosis**: Contains patient-related diagnosis information with columns such as 
            `hadm_id`, `icd_code`, `root_icd10_convert`, and `root`. The `icd_code` column contains the 
            original ICD codes, while `root_icd10_convert` provides the corresponding ICD-10 codes (if applicable). 
            It is not possible to know the exact timing of the diagnosis

        - **output**: Provides observations related to patient care during their stay in the ICU, with the following columns:
            - `hadm_id`: Hospital admission ID, linking observations to specific hospital admissions.
            - `itemid`: Identifier for specific observations or measurements, which can be referenced in the items table for 
            further detail.
            - `charttime`: Timestamp indicating when the observation was recorded.
            - `intime`: Timestamp of the patient's admission to the ICU, allowing for temporal analysis of events.
            - `event_time_from_admit`: The time elapsed from the patient's admission to the observation event, facilitating 
            analyses related to the timing of interventions and patient responses.
            - `value`: The recorded measurement or observation value (e.g., medication doses, lab results).

        - **chart**: Contains detailed chart event data for patients in the cohort, structured as follows:
                    - `stay_id`: Identifier for the patient's stay in the hospital.
                    - `itemid`: Identifier for specific chart observations, which can be cross-referenced with the items table for further detail.
                    - `event_time_from_admit`: The time elapsed from the patient's admission to the observation event, facilitating 
                    analyses related to the timing of care and interventions.
                    - `valuenum`: Numeric value of the observation (e.g., vital signs, lab results).

        - **procedures**: Contains data on procedures performed during ICU stays, structured as follows:
            - `hadm_id`: Hospital admission ID, linking the procedure to the specific hospital admission.
            - `itemid`: Identifier for the specific procedure, which can be cross-referenced with the items table for additional details.
                -> this is the important procedure. there is no associated value. 
            - `starttime`: Timestamp of when the procedure event occurred.
            - `intime`: Timestamp of the patient's admission to the hospital.
            - `event_time_from_admit`: The time elapsed from the patient's admission to the procedure event, facilitating 
              analyses related to the timing of interventions and patient responses.

        - **medications**: Contains information on medication administration during the ICU stay, structured as follows:
            - hadm_id: Hospital admission ID, linking medications to specific hospital admissions.
            - itemid: Identifier for the specific medication, which can be referenced in the items table for further details.
                -> which medication was given
            - starttime: Timestamp of when the medication administration started.
                -> from when was medication given
            - endtime: Timestamp of when the medication administration ended.
                -> until when was medication given
            - start_hours_from_admit: Time elapsed from the patient's admission to the start of the medication administration, facilitating analysis of medication timing.
            - stop_hours_from_admit: Time elapsed from the patient's admission to the end of the medication administration, facilitating analysis of medication duration.
            - rate: Rate of medication administration (if applicable).
                -> how often was medication given. Can be None, then it's a one-time med. 
            - amount: Amount of medication administered.
                -> how much information was given
            - orderid: Identifier for the medication order, which can be used to track the order details.

        - **lab**: Contains laboratory test result data for ICU patients, structured as follows:
            - `subject_id`: Unique identifier for the patient.
            - `hadm_id`: Hospital admission ID, linking laboratory events to specific hospital admissions.
            - `stay_id`: Identifier for the patient's stay in the hospital.
            - `itemid`: Identifier for the specific laboratory test, which can be cross-referenced with the items table for further details.
            - `charttime`: Timestamp indicating when the laboratory observation was recorded.
            - `storetime`: Timestamp indicating when the laboratory result was stored. This variable can be missing.
            - `valuenum`: Numeric value of the laboratory test result (if applicable). 
                -> This variable can be missing. In that case maybe it has no value.
            - `valueuom`: Unit of measurement for the test result. This variable can be missing.
            - `ref_range_lower`: Lower bound of the reference range for the laboratory test (if available). This variable can be missing.
            - `ref_range_upper`: Upper bound of the reference range for the laboratory test (if available). This variable can be missing.
            - `chart_hours_from_admit`: Time elapsed from the patient's admission to the lab event's charttime.
            - `store_hours_from_admit`: Time elapsed from the patient's admission to the lab event's storetime. This variable can be missing.
     
    """
    
    # Filepaths for each type of data to be stored
    filenames = {
        'diagnosis': "preproc_diag_icu.csv.gz",
        'output': "preproc_out_icu.csv.gz",
        'chart': "preproc_chart_icu.csv.gz",
        'procedures': "preproc_proc_icu.csv.gz",
        'medications': "preproc_med_icu.csv.gz",
        'lab': "preproc_labevents_icu.csv.gz",
        'microbiology': "preproc_microbiologyevents_icu.csv.gz"
    }
    
    data_dict = {}
    
    # Diagnosis data preprocessing
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag_cols = ['subject_id', 'hadm_id', 'stay_id', 'icd_code', 'root_icd10_convert', 'root']
        diag = preproc_icd_module(  module_path = version_path+"/hosp/diagnoses_icd.csv.gz", 
                                    adm_cohort_path = root_dir + '/data/cohort/'+ cohort_output+'.csv.gz', 
                                    icd_map_path='./utils/mappings/ICD9_to_ICD10_mapping.txt', 
                                    map_code_colname='diagnosis_code', 
                                    only_icd10=True)
        diag_filtered = diag[diag_cols]
        print(f"Columns kept for diagnosis: {diag_cols}")
        diag_filtered.to_csv(f"{save_path}/{filenames['diagnosis']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        data_dict['diag'] = diag_filtered
    
    # Output events preprocessing
    if out_flag:
        print("[EXTRACTING OUTPUT EVENTS DATA]")
        out_cols = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit', "value"]
        out = preproc_out(dataset_path = version_path+"/icu/outputevents.csv.gz", 
                          cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz', 
                          time_col = 'charttime', dtypes=None, usecols=None)
        out_filtered = out[out_cols]
        print(f"Columns kept for output events: {out_cols}")
        out_filtered.to_csv(f"{save_path}/{filenames['output']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
        data_dict['out'] = out_filtered
    
    # Chart events preprocessing
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart_cols = ['stay_id', 'itemid', 'event_time_from_admit', 'valuenum']
        chart = preproc_chart( dataset_path = version_path+"/icu/chartevents.csv.gz", 
                               cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz',
                               time_col = 'charttime', dtypes=None, 
                               usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart_filtered = chart[chart_cols]
        print(f"Columns kept for chart events: {chart_cols}")
        chart_filtered.to_csv(f"{save_path}/{filenames['chart']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
        data_dict['chart'] = chart_filtered
    
    # Procedures data preprocessing
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc_cols = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']
        proc = preproc_proc(
            dataset_path = version_path+"/icu/procedureevents.csv.gz", 
            cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz', 
            time_col = 'starttime',dtypes=None, usecols=['stay_id','starttime','itemid'])
        proc_filtered = proc[proc_cols]
        print(f"Columns kept for procedures: {proc_cols}")
        proc_filtered.to_csv(f"{save_path}/{filenames['procedures']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        data_dict['proc'] = proc_filtered
    
    # Medication data preprocessing
    if med_flag:
        print("[EXTRACTING MEDICATION DATA]")
        med_cols = ['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'endtime', 'start_hours_from_admit', 'stop_hours_from_admit', 'rate', 'amount', 'orderid']
        med = preproc_meds( module_path = version_path+"/icu/inputevents.csv.gz", 
                            adm_cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
        med_filtered = med[med_cols]
        print(f"Columns kept for medication: {med_cols}")
        med_filtered.to_csv(f"{save_path}/{filenames['medications']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATION DATA]")
        data_dict['meds'] = med_filtered

    # Lab events data preprocessing
    if lab_flag:
        print("[EXTRACTING LAB EVENTS DATA]")
        lab_cols = ["subject_id", 'hadm_id', "stay_id", 'itemid', 'charttime', 'storetime', 
                    'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', "chart_hours_from_admit", 'store_hours_from_admit']
        lab = preproc_lab( module_path = version_path+"/hosp/labevents.csv.gz", 
                          adm_cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
        lab_filtered = lab[lab_cols]
        print(f"Columns kept for lab events: {lab_cols}")
        lab_filtered.to_csv(f"{save_path}/{filenames['lab']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED LAB EVENTS DATA]")
        data_dict['lab'] = lab_filtered
    
    # Microbiology events data preprocessing
    if micro_flag:
        print("[EXTRACTING MICROBIOLOGY EVENTS DATA]")
        micro_cols = ['subject_id', 'hadm_id', "stay_id", "chartdate", 'charttime', 'spec_itemid', 
                      'storedate', 'storetime', 'test_itemid', "start_hours_from_admit", 'stop_hours_from_admit']
        micro = preproc_micro(  module_path = version_path+"/hosp/microbiologyevents.csv.gz", 
                                adm_cohort_path = root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
        micro_filtered = micro[micro_cols]
        print(f"Columns kept for microbiology events: {micro_cols}")
        micro_filtered.to_csv(f"{save_path}/{filenames['microbiology']}", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MICROBIOLOGY EVENTS DATA]")
        data_dict['micro'] = micro_filtered
    
    return data_dict

# OLD: can be deleted
# #def feature_icu(cohort_output, root_dir: str, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True):
# def feature_icu(cohort_output, root_dir: str, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True,lab_flag=True,micro_flag=True):
#     if diag_flag:
#         print("[EXTRACTING DIAGNOSIS DATA]")
#         diag = preproc_icd_module(root_dir, version_path+"/hosp/diagnoses_icd.csv.gz", root_dir + '/data/cohort/'+ cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
#         # diag = preproc_icd_module(version_path+"/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
#         diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
#     if out_flag:  
#         print("[EXTRACTING OUPTPUT EVENTS DATA]")
#         out = preproc_out(root_dir, version_path+"/icu/outputevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
#         out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit', "value"]].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
#     if chart_flag:
#         print("[EXTRACTING CHART EVENTS DATA]")
#         chart=preproc_chart(root_dir, version_path+"/icu/chartevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
#         chart = drop_wrong_uom(chart, 0.95)
#         chart[['stay_id', 'itemid','event_time_from_admit','valuenum']].to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    
#     if proc_flag:
#         print("[EXTRACTING PROCEDURES DATA]")
#         proc = preproc_proc(root_dir, version_path+"/icu/procedureevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz', 'starttime', dtypes=None, usecols=['stay_id','starttime','itemid'])
#         proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
#     if med_flag:
#         print("[EXTRACTING MEDICATIONS DATA]")
#         med = preproc_meds(root_dir, version_path+"/icu/inputevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
#         med[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

#     if lab_flag:
#         print("[EXTRACTING LAB EVENTS DATA]")
#         lab = preproc_lab(root_dir, version_path+"/hosp/labevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
#         lab[["subject_id", 'hadm_id', "stay_id", 'itemid', 'charttime', 'storetime', 'valuenum',
#        'valueuom', 'ref_range_lower', 'ref_range_upper', "start_hours_from_admit", 'stop_hours_from_admit']].to_csv('./data/features/preproc_labevents_icu.csv.gz', compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED LAB EVENTS DATA]")

#     if micro_flag:
#         print("[EXTRACTING MICROBIOLOGY EVENTS DATA]")
#         micro = preproc_micro(root_dir, version_path+"/hosp/microbiologyevents.csv.gz", root_dir + '/data/cohort/'+cohort_output+'.csv.gz')
#         micro[['subject_id', 'hadm_id', "stay_id", "chartdate", 'charttime', 'spec_itemid', 'storedate', 'storetime', 'test_itemid', "start_hours_from_admit", 'stop_hours_from_admit']].to_csv('./data/features/preproc_microbiologyevents_icu.csv.gz', compression='gzip', index=False)
#         print("[SUCCESSFULLY SAVED MICROBIOLOGY EVENTS DATA]")

def preprocess_features_icu(cohort_output: str, save_path: str, diag_flag: bool, group_diag: str, 
                            chart_flag: bool, clean_chart: bool, 
                            impute_outlier_chart: bool, thresh: float, 
                            left_thresh: float) -> None:
    """
    Preprocesses diagnosis and chart event data for ICU patient cohorts.

    This function performs preprocessing of diagnosis data and chart events based on specified 
    flags and criteria. It can retain, convert, or group ICD codes for diagnosis data and 
    handle outlier imputation for chart events.

    Parameters:
    -----------
    cohort_output : str
        Path to the output file or directory for the processed cohort data.
        
    diag_flag : bool
        A flag indicating whether to process diagnosis data. If True, diagnosis data will be loaded 
        and processed.
        
    group_diag : str
        Specifies the approach for handling ICD codes. Options include:
        - 'Keep both ICD-9 and ICD-10 codes'
        - 'Convert ICD-9 to ICD-10 codes'
        - 'Convert ICD-9 to ICD-10 and group ICD-10 codes'
        
    chart_flag : bool
        A flag indicating whether to process chart event data. If True, chart data will be loaded 
        and processed.
        
    clean_chart : bool
        A flag indicating whether to perform cleaning operations on chart data. If True, 
        outlier imputation will be performed on the chart values.
        
    impute_outlier_chart : bool
        A flag indicating whether to impute outliers in the chart data during processing.
        
    thresh : float
        Threshold value used for outlier detection in chart data.
        
    left_thresh : float
        The lower bound threshold for outlier detection in chart data.

    Returns:
    --------
    None
        The function saves processed diagnosis and chart data to specified files and does not return any value.
    
    Remarks:
    --------
    - The function prints messages to indicate the progress of data processing and saving steps.
    - The processed diagnosis data is saved as a gzip-compressed CSV file at 
      "./data/features/preproc_diag_icu.csv.gz".
    - The processed chart data is saved as a gzip-compressed CSV file at 
      "./data/features/preproc_chart_icu.csv.gz".
    """
    
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv(f"{save_path}preproc_diag_icu.csv.gz", compression='gzip', header=0)
        if group_diag == 'Keep both ICD-9 and ICD-10 codes':
            diag['new_icd_code'] = diag['icd_code']
        if group_diag == 'Convert ICD-9 to ICD-10 codes':
            diag['new_icd_code'] = diag['root_icd10_convert']
        if group_diag == 'Convert ICD-9 to ICD-10 and group ICD-10 codes':
            diag['new_icd_code'] = diag['root']

        diag = diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
        print("Total number of rows", diag.shape[0])
        diag.to_csv(f"{save_path}preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")

    if chart_flag:
        if clean_chart:
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv(f"{save_path}preproc_chart_icu.csv.gz", compression='gzip', header=0)
            chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh, left_thresh, impute_outlier_chart)

            print("Total number of rows", chart.shape[0])
            chart.to_csv(f"{save_path}preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    return


    # if diag_flag:
    #     print("[PROCESSING DIAGNOSIS DATA]")
    #     diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', header=0)
    #     if group_diag == 'Keep both ICD-9 and ICD-10 codes':
    #         diag['new_icd_code'] = diag['icd_code']
    #     if group_diag == 'Convert ICD-9 to ICD-10 codes':
    #         diag['new_icd_code'] = diag['root_icd10_convert']
    #     if group_diag == 'Convert ICD-9 to ICD-10 and group ICD-10 codes':
    #         diag['new_icd_code'] = diag['root']

    #     diag = diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
    #     print("Total number of rows", diag.shape[0])
    #     diag.to_csv(f"{save_path}/preproc_diag_icu.csv.gz", compression='gzip', index=False)
    #     print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        
    # if chart_flag:
    #     if clean_chart:   
    #         print("[PROCESSING CHART EVENTS DATA]")
    #         chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0)
    #         chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh, left_thresh, impute_outlier_chart)
            
    #         print("Total number of rows", chart.shape[0])
    #         chart.to_csv(f"{save_path}/preproc_chart_icu.csv.gz", compression='gzip', index=False)
    #         print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

    # return 

            
        
        
#def generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag, lab_flag):
def generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag, lab_flag, micro_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        freq=diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv('./data/summary/diag_features.csv',index=False)

    
    #MANUALLY ADDED
    if lab_flag:
        lab = pd.read_csv("./data/features/preproc_labevents_icu.csv.gz", compression='gzip',header=0)
        freq=lab.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        #missing=lab[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=lab.groupby('itemid').size().reset_index(name="total_count")
        #summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,total,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/lab_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/lab_features.csv',index=False)

    if micro_flag:
        micro = pd.read_csv("./data/features/preproc_microbiologyevents_icu.csv.gz", compression='gzip',header=0)
        freq=micro.groupby(['stay_id','test_itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['test_itemid'])['mean_frequency'].mean().reset_index()
        
        #missing=micro[micro['amount']==0].groupby('test_itemid').size().reset_index(name="missing_count")
        total=micro.groupby('test_itemid').size().reset_index(name="total_count")
        #summary=pd.merge(missing,total,on='test_itemid',how='right')
        summary=pd.merge(freq,freq,on='test_itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/micro_summary.csv',index=False)
        summary['test_itemid'].to_csv('./data/summary/micro_features.csv',index=False)


    if med_flag:
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=med.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/med_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/med_features.csv',index=False)

    
    
    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/proc_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/proc_features.csv',index=False)

        
    if out_flag:
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
        freq=out.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=out.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/out_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/out_features.csv',index=False)
        
    if chart_flag:
        chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

        missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=chart.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        #summary=summary.fillna(0)

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/chart_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/chart_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")
    
def features_selection_icu(cohort_output, diag_flag,proc_flag,med_flag,out_flag,chart_flag,group_diag,group_med,group_proc,group_out,group_chart):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/diag_features.csv",header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/med_features.csv",header=0)
            med=med[med['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/proc_features.csv",header=0)
            proc=proc[proc['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/out_features.csv",header=0)
            out=out[out['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            out.to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            features=pd.read_csv("./data/summary/chart_features.csv",header=0)
            chart=chart[chart['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")