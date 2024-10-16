import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import pickle
import datetime
import os
import sys
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')
if not os.path.exists("./data/dict"):
    os.makedirs("./data/dict")
if not os.path.exists("./data/csv"):
    os.makedirs("./data/csv")
    
class Generator():
    def __init__(self,root_dir, cohort_output,if_mort,if_admn,if_los,feat_cond,feat_proc,feat_out,feat_chart,feat_med,feat_lab,feat_micro,impute,include_time=24,bucket=1,predW=6):
        self.feat_cond,self.feat_proc,self.feat_out,self.feat_chart,self.feat_med,self.feat_lab,self.feat_micro = feat_cond,feat_proc,feat_out,feat_chart,feat_med,feat_lab,feat_micro
        self.cohort_output=cohort_output
        self.impute=impute
        self.data = self.generate_adm(root_dir = root_dir)
        print("[ READ COHORT ]")
        
        self.generate_feat()
        print("[ READ ALL FEATURES ]")
        
        if if_mort:
            self.mortality_length(include_time,predW)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_admn:
            self.readmission_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        elif if_los:
            self.los_length(include_time)
            print("[ PROCESSED TIME SERIES TO EQUAL LENGTH  ]")
        
        self.smooth_meds(bucket)
        print("[ SUCCESSFULLY SAVED DATA DICTIONARIES ]")
    
    def generate_feat(self):
        if(self.feat_cond):
            print("[ ======READING DIAGNOSIS ]")
            self.generate_cond()
        if(self.feat_proc):
            print("[ ======READING PROCEDURES ]")
            self.generate_proc()
        if(self.feat_out):
            print("[ ======READING OUT EVENTS ]")
            self.generate_out()
        if(self.feat_chart):
            print("[ ======READING CHART EVENTS ]")
            self.generate_chart()
        if(self.feat_med):
            print("[ ======READING MEDICATIONS ]")
            self.generate_meds()

        #MANUAL INPUT
        if(self.feat_lab):
            print("[ ======READING LAB EVENTS ]")
            self.generate_lab()
        if(self.feat_micro):
            print("[ ======READING MICROBIOLOGY EVENTS ]")
            self.generate_micro()

    def generate_adm(self, root_dir : str):
        data=pd.read_csv(root_dir + f"/data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        # data=pd.read_csv(root_dir + f"./data/cohort/{self.cohort_output}.csv.gz", compression='gzip', header=0, index_col=None)
        data['intime'] = pd.to_datetime(data['intime'])
        data['outtime'] = pd.to_datetime(data['outtime'])
        data['los']=pd.to_timedelta(data['outtime']-data['intime'],unit='h')
        data['los']=data['los'].astype(str)
        data[['days', 'dummy','hours']] = data['los'].str.split(' ', -1, expand=True)
        data[['hours','min','sec']] = data['hours'].str.split(':', -1, expand=True)
        data['los']=pd.to_numeric(data['days'])*24+pd.to_numeric(data['hours'])
        data=data.drop(columns=['days', 'dummy','hours','min','sec'])
        data=data[data['los']>0]
        data['Age']=data['Age'].astype(int)
        #print(data.head())
        #print(data.shape)
        return data
    
    def generate_cond(self):
        cond=pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', header=0, index_col=None)
        cond=cond[cond['stay_id'].isin(self.data['stay_id'])]
        cond_per_adm = cond.groupby('stay_id').size().max()
        self.cond, self.cond_per_adm = cond, cond_per_adm
    
    def generate_proc(self):
        proc=pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', header=0, index_col=None)
        proc=proc[proc['stay_id'].isin(self.data['stay_id'])]
        proc[['start_days', 'dummy','start_hours']] = proc['event_time_from_admit'].str.split(' ', -1, expand=True)
        proc[['start_hours','min','sec']] = proc['start_hours'].str.split(':', -1, expand=True)
        proc['start_time']=pd.to_numeric(proc['start_days'])*24+pd.to_numeric(proc['start_hours'])
        proc=proc.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        proc=proc[proc['start_time']>=0]
        
        ###Remove where event time is after discharge time
        proc=pd.merge(proc,self.data[['stay_id','los']],on='stay_id',how='left')
        proc['sanity']=proc['los']-proc['start_time']
        proc=proc[proc['sanity']>0]
        del proc['sanity']
        
        self.proc=proc
        
    def generate_out(self):
        out=pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', header=0, index_col=None)
        out=out[out['stay_id'].isin(self.data['stay_id'])]
        out[['start_days', 'dummy','start_hours']] = out['event_time_from_admit'].str.split(' ', -1, expand=True)
        out[['start_hours','min','sec']] = out['start_hours'].str.split(':', -1, expand=True)
        out['start_time']=pd.to_numeric(out['start_days'])*24+pd.to_numeric(out['start_hours'])
        out=out.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        out=out[out['start_time']>=0]
        
        ###Remove where event time is after discharge time
        out=pd.merge(out,self.data[['stay_id','los']],on='stay_id',how='left')
        out['sanity']=out['los']-out['start_time']
        out=out[out['sanity']>0]
        del out['sanity']
        
        self.out=out
        
        
    def generate_chart(self):
        chunksize = 5000000
        final=pd.DataFrame()
        for chart in tqdm(pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0, index_col=None,chunksize=chunksize)):
            chart=chart[chart['stay_id'].isin(self.data['stay_id'])]
            chart[['start_days', 'dummy','start_hours']] = chart['event_time_from_admit'].str.split(' ', -1, expand=True)
            chart[['start_hours','min','sec']] = chart['start_hours'].str.split(':', -1, expand=True)
            chart['start_time']=pd.to_numeric(chart['start_days'])*24+pd.to_numeric(chart['start_hours'])
            chart=chart.drop(columns=['start_days', 'dummy','start_hours','min','sec','event_time_from_admit'])
            chart=chart[chart['start_time']>=0]

            ###Remove where event time is after discharge time
            chart=pd.merge(chart,self.data[['stay_id','los']],on='stay_id',how='left')
            chart['sanity']=chart['los']-chart['start_time']
            chart=chart[chart['sanity']>0]
            del chart['sanity']
            del chart['los']
            
            if final.empty:
                final=chart
            else:
                final=final.append(chart, ignore_index=True)
        
        self.chart=final
        
        
        
    def generate_meds(self):
        meds=pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['stay_id'].isin(self.data['stay_id'])]
        meds=pd.merge(meds,self.data[['stay_id','los']],on='stay_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        
        meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.meds=meds

    #MANUAL INPUT
    def generate_lab(self):
        lab=pd.read_csv("./data/features/preproc_labevents_icu.csv.gz", compression='gzip', header=0, index_col=None)
        lab[['start_days', 'dummy','start_hours']] = lab['start_hours_from_admit'].str.split(' ', -1, expand=True)
        lab[['start_hours','min','sec']] = lab['start_hours'].str.split(':', -1, expand=True)
        lab['start_time']=pd.to_numeric(lab['start_days'])*24+pd.to_numeric(lab['start_hours'])
        lab[['start_days', 'dummy','start_hours']] = lab['stop_hours_from_admit'].str.split(' ', -1, expand=True)
        lab[['start_hours','min','sec']] = lab['start_hours'].str.split(':', -1, expand=True)
        lab['stop_time']=pd.to_numeric(lab['start_days'])*24+pd.to_numeric(lab['start_hours'])
        lab=lab.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        lab['sanity']=lab['stop_time']-lab['start_time']
        lab=lab[lab['sanity']>0]
        del lab['sanity']
        #####Select hadm_id as in main file
        lab=lab[lab['stay_id'].isin(self.data['stay_id'])]
        lab=pd.merge(lab,self.data[['stay_id','los']],on='stay_id',how='left')

        #####Remove where start time is after end of visit
        lab['sanity']=lab['los']-lab['start_time']
        lab=lab[lab['sanity']>0]
        del lab['sanity']
        ####Any stop_time after end of visit is set at end of visit
        lab.loc[lab['stop_time'] > lab['los'],'stop_time']=lab.loc[lab['stop_time'] > lab['los'],'los']
        del lab['los']
        
        #meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        #meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.lab=lab

    def generate_micro(self):
        meds=pd.read_csv("./data/features/preproc_microbiologyevents_icu.csv.gz", compression='gzip', header=0, index_col=None)
        meds[['start_days', 'dummy','start_hours']] = meds['start_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['start_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds[['start_days', 'dummy','start_hours']] = meds['stop_hours_from_admit'].str.split(' ', -1, expand=True)
        meds[['start_hours','min','sec']] = meds['start_hours'].str.split(':', -1, expand=True)
        meds['stop_time']=pd.to_numeric(meds['start_days'])*24+pd.to_numeric(meds['start_hours'])
        meds=meds.drop(columns=['start_days', 'dummy','start_hours','min','sec'])
        #####Sanity check
        meds['sanity']=meds['stop_time']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        #####Select hadm_id as in main file
        meds=meds[meds['stay_id'].isin(self.data['stay_id'])]
        meds=pd.merge(meds,self.data[['stay_id','los']],on='stay_id',how='left')

        #####Remove where start time is after end of visit
        meds['sanity']=meds['los']-meds['start_time']
        meds=meds[meds['sanity']>0]
        del meds['sanity']
        ####Any stop_time after end of visit is set at end of visit
        meds.loc[meds['stop_time'] > meds['los'],'stop_time']=meds.loc[meds['stop_time'] > meds['los'],'los']
        del meds['los']
        
        #meds['rate']=meds['rate'].apply(pd.to_numeric, errors='coerce')
        #meds['amount']=meds['amount'].apply(pd.to_numeric, errors='coerce')
        
        self.micro= meds
    
    def mortality_length(self,include_time,predW):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time+predW)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
        
        #self.los=include_time
    
    def los_length(self,include_time):
        print("include_time",include_time)
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=self.meds[self.meds['start_time']<=include_time]
            self.meds.loc[self.meds.stop_time >include_time, 'stop_time']=include_time
                    
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=self.proc[self.proc['start_time']<=include_time]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=self.out[self.out['start_time']<=include_time]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=self.chart[self.chart['start_time']<=include_time]
            
    def readmission_length(self,include_time):
        self.los=include_time
        self.data=self.data[(self.data['los']>=include_time)]
        self.hids=self.data['stay_id'].unique()
        
        if(self.feat_cond):
            self.cond=self.cond[self.cond['stay_id'].isin(self.data['stay_id'])]
        self.data['select_time']=self.data['los']-include_time
        self.data['los']=include_time

        ####Make equal length input time series and remove data for pred window if needed
        
        ###MEDS
        if(self.feat_med):
            self.meds=self.meds[self.meds['stay_id'].isin(self.data['stay_id'])]
            self.meds=pd.merge(self.meds,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.meds['stop_time']=self.meds['stop_time']-self.meds['select_time']
            self.meds['start_time']=self.meds['start_time']-self.meds['select_time']
            self.meds=self.meds[self.meds['stop_time']>=0]
            self.meds.loc[self.meds.start_time <0, 'start_time']=0
        
        ###PROCS
        if(self.feat_proc):
            self.proc=self.proc[self.proc['stay_id'].isin(self.data['stay_id'])]
            self.proc=pd.merge(self.proc,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.proc['start_time']=self.proc['start_time']-self.proc['select_time']
            self.proc=self.proc[self.proc['start_time']>=0]
            
        ###OUT
        if(self.feat_out):
            self.out=self.out[self.out['stay_id'].isin(self.data['stay_id'])]
            self.out=pd.merge(self.out,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.out['start_time']=self.out['start_time']-self.out['select_time']
            self.out=self.out[self.out['start_time']>=0]
            
       ###CHART
        if(self.feat_chart):
            self.chart=self.chart[self.chart['stay_id'].isin(self.data['stay_id'])]
            self.chart=pd.merge(self.chart,self.data[['stay_id','select_time']],on='stay_id',how='left')
            self.chart['start_time']=self.chart['start_time']-self.chart['select_time']
            self.chart=self.chart[self.chart['start_time']>=0]
                 
    def smooth_meds(self,bucket):
        """
        Smooth and aggregate time-series data for medications, procedures, output events, 
        and chart events into time intervals based on the specified `bucket` size (time interval).

        This function processes each type of medical data (medications, procedures, output events, 
        and chart events) by grouping and aggregating them into fixed-length time intervals (`bucket`), 
        smoothing over these intervals, and then generating time-aligned features for each patient 
        stay (`stay_id`). The function works in four steps:
        
        1. Sorts the relevant medical data based on the start time of the event.
        2. Iterates over the entire length of stay (`los`), and for each time bucket, 
        groups the events by stay ID and aggregates various fields such as event duration 
        or measurement values (depending on the type of event).
        3. Aggregates the data per bucket and appends it to the final dataframe.
        4. Calculates per-patient summary statistics such as the number of events 
        (e.g., procedures, medications) per admission.

        Parameters
        ----------
        bucket : int
            The time interval (in hours) for smoothing the data. All events occurring 
            within this interval are grouped and summarized.

        Attributes
        ----------
        final_meds : DataFrame
            Aggregated medication data, grouped and smoothed by `bucket` intervals.
        final_proc : DataFrame
            Aggregated procedure data, grouped and smoothed by `bucket` intervals.
        final_out : DataFrame
            Aggregated output event data, grouped and smoothed by `bucket` intervals.
        final_chart : DataFrame
            Aggregated chart event data, grouped and smoothed by `bucket` intervals.

        Example
        -------
        Given a bucket size of 1 hour, medications, procedures, output events, 
        and chart events are grouped into 1-hour intervals, smoothed, and the relevant 
        statistics are calculated for each patient.
        
        Notes
        -----
        - This function handles large medical datasets and performs time-based aggregation.
        - The bucket size controls how data is summarized over time.
        - After processing, the function generates dictionaries of processed features 
        for further use in modeling or analysis.
        
        """
        final_meds=pd.DataFrame()
        final_proc=pd.DataFrame()
        final_out=pd.DataFrame()
        final_chart=pd.DataFrame()

        #manual input
        final_lab=pd.DataFrame()
        final_micro=pd.DataFrame()
        
        if(self.feat_med):
            self.meds=self.meds.sort_values(by=['start_time'])
        if(self.feat_proc):
            self.proc=self.proc.sort_values(by=['start_time'])
        if(self.feat_out):
            self.out=self.out.sort_values(by=['start_time'])
        if(self.feat_chart):
            self.chart=self.chart.sort_values(by=['start_time'])
        if(self.feat_lab):
            self.lab=self.lab.sort_values(by=['start_time'])
        if(self.feat_micro):
            self.micro=self.micro.sort_values(by=['start_time'])
        
        t=0
        for i in tqdm(range(0,self.los,bucket)): 
            ###MEDS
             if(self.feat_med):
                sub_meds=self.meds[(self.meds['start_time']>=i) & (self.meds['start_time']<i+bucket)].groupby(['stay_id','itemid','orderid']).agg({'stop_time':'max','subject_id':'max','rate':np.nanmean,'amount':np.nanmean})
                sub_meds=sub_meds.reset_index()
                sub_meds['start_time']=t
                sub_meds['stop_time']=sub_meds['stop_time']/bucket
                if final_meds.empty:
                    final_meds=sub_meds
                else:
                    final_meds=final_meds.append(sub_meds)

            #manual input
            ##LAB
             if(self.feat_lab):
                sub_lab=self.lab[(self.lab['start_time'] >= i) & (self.lab['start_time'] < i + bucket)].groupby(['stay_id', 'itemid']).agg({'stop_time': 'max','subject_id':'max','valuenum': np.nanmean,'ref_range_lower': np.nanmean,'ref_range_upper': np.nanmean,'valueuom': 'max'})
                sub_lab=sub_lab.reset_index()
                sub_lab['start_time']=t
                sub_lab['stop_time']=sub_lab['stop_time']/bucket
                if final_lab.empty:
                    final_lab=sub_lab
                else:
                    final_lab=final_lab.append(sub_lab)

            ##MICRO
             if(self.feat_micro):
                sub_micro=self.micro[(self.micro['start_time']>=i) & (self.micro['start_time']<i+bucket)].groupby(['stay_id','test_itemid']).agg({'stop_time':'max','subject_id':'max','spec_itemid':"max"})
                sub_micro=sub_micro.reset_index()
                sub_micro['start_time']=t
                sub_micro['stop_time']=sub_micro['stop_time']/bucket
                if final_micro.empty:
                    final_micro=sub_micro
                else:
                    final_micro=final_micro.append(sub_micro)
            
            ###PROC
             if(self.feat_proc):
                sub_proc=self.proc[(self.proc['start_time']>=i) & (self.proc['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'subject_id':'max'})
                sub_proc=sub_proc.reset_index()
                sub_proc['start_time']=t
                if final_proc.empty:
                    final_proc=sub_proc
                else:    
                    final_proc=final_proc.append(sub_proc)
                    
              ###OUT
             if(self.feat_out):
                sub_out=self.out[(self.out['start_time']>=i) & (self.out['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'subject_id':'max'})
                sub_out=sub_out.reset_index()
                sub_out['start_time']=t
                if final_out.empty:
                    final_out=sub_out
                else:    
                    final_out=final_out.append(sub_out)
                    
                    
              ###CHART
             if(self.feat_chart):
                sub_chart=self.chart[(self.chart['start_time']>=i) & (self.chart['start_time']<i+bucket)].groupby(['stay_id','itemid']).agg({'valuenum':np.nanmean})
                sub_chart=sub_chart.reset_index()
                sub_chart['start_time']=t
                if final_chart.empty:
                    final_chart=sub_chart
                else:    
                    final_chart=final_chart.append(sub_chart)
            
             t=t+1
        print("bucket",bucket)
        los=int(self.los/bucket)
        
        
        ###MEDS
        if(self.feat_med):
            f2_meds=final_meds.groupby(['stay_id','itemid','orderid']).size()
            self.med_per_adm=f2_meds.groupby('stay_id').sum().reset_index()[0].max()                 
            self.medlength_per_adm=final_meds.groupby('stay_id').size().max()

        #manual input
        ##LAB
        if(self.feat_lab):
            f2_lab=final_lab.groupby(['stay_id','itemid']).size()
            self.lab_per_adm=f2_lab.groupby('stay_id').sum().reset_index()[0].max()                 
            self.lablength_per_adm=final_lab.groupby('stay_id').size().max()

        ##MICRO
        if(self.feat_micro):
            f2_micro=final_micro.groupby(['stay_id',"test_itemid"]).size()
            self.micro_per_adm=f2_micro.groupby('stay_id').sum().reset_index()[0].max()                 
            self.microlength_per_adm=final_micro.groupby('stay_id').size().max()
        
        ###PROC
        if(self.feat_proc):
            f2_proc=final_proc.groupby(['stay_id','itemid']).size()
            self.proc_per_adm=f2_proc.groupby('stay_id').sum().reset_index()[0].max()       
            self.proclength_per_adm=final_proc.groupby('stay_id').size().max()
            
        ###OUT
        if(self.feat_out):
            f2_out=final_out.groupby(['stay_id','itemid']).size()
            self.out_per_adm=f2_out.groupby('stay_id').sum().reset_index()[0].max() 
            self.outlength_per_adm=final_out.groupby('stay_id').size().max()
            
            
        ###chart
        if(self.feat_chart):
            f2_chart=final_chart.groupby(['stay_id','itemid']).size()
            self.chart_per_adm=f2_chart.groupby('stay_id').sum().reset_index()[0].max()             
            self.chartlength_per_adm=final_chart.groupby('stay_id').size().max()
        
        print("[ PROCESSED TIME SERIES TO EQUAL TIME INTERVAL ]")
        ###CREATE DICT
#         if(self.feat_chart):
#             self.create_chartDict(final_chart,los)
#         else:
        self.create_Dict(final_meds,final_proc,final_out,final_chart, final_lab, final_micro,los)
        
    
    def create_chartDict(self,chart,los):
        dataDic={}
        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Chart':{},'label':int(grp['label'])}
        for hid in tqdm(self.hids):
            ###CHART
            if(self.feat_chart):
                df2=chart[chart['stay_id']==hid]
                val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                df2['val']=1
                df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                #print(df2.shape)
                add_indices = pd.Index(range(los)).difference(df2.index)
                add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                df2=pd.concat([df2, add_df])
                df2=df2.sort_index()
                df2=df2.fillna(0)
                
                val=pd.concat([val, add_df])
                val=val.sort_index()
                if self.impute=='Mean':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.mean())
                elif self.impute=='Median':
                    val=val.ffill()
                    val=val.bfill()
                    val=val.fillna(val.median())
                val=val.fillna(0)
                
                
                df2[df2>0]=1
                df2[df2<0]=0
                #print(df2.head())
                dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")
            
            
                
        ######SAVE DICTIONARIES##############
        with open("./data/dict/metaDic", 'rb') as fp:
            metaDic=pickle.load(fp)
        
        with open("./data/dict/dataChartDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

      
        with open("./data/dict/chartVocab", 'wb') as fp:
            pickle.dump(list(chart['itemid'].unique()), fp)
        self.chart_vocab = chart['itemid'].nunique()
        metaDic['Chart']=self.chart_per_adm
        
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
    def create_Dict(self,meds,proc,out,chart, lab, micro,los):
        dataDic={}
        print(los)
        labels_csv=pd.DataFrame(columns=['stay_id','label'])
        labels_csv['stay_id']=pd.Series(self.hids)
        labels_csv['label']=0
#         print("# Unique gender",self.data.gender.nunique())
#         print("# Unique ethnicity",self.data.ethnicity.nunique())
#         print("# Unique insurance",self.data.insurance.nunique())

        for hid in self.hids:
            grp=self.data[self.data['stay_id']==hid]
            dataDic[hid]={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{},'Lab': {},   # Add Lab data
            'Micro': {},'ethnicity':grp['ethnicity'].iloc[0],'age':int(grp['Age']),'gender':grp['gender'].iloc[0],'label':int(grp['label'])}
            labels_csv.loc[labels_csv['stay_id']==hid,'label']=int(grp['label'])
            

            #print(static_csv.head())
        for hid in tqdm(self.hids):
            grp=self.data[self.data['stay_id']==hid]
            demo_csv=grp[['Age','gender','ethnicity','insurance']]
            if not os.path.exists("./data/csv/"+str(hid)):
                os.makedirs("./data/csv/"+str(hid))
            demo_csv.to_csv('./data/csv/'+str(hid)+'/demo.csv',index=False)
            
            dyn_csv=pd.DataFrame()
            ###MEDS
            if(self.feat_med):
                feat=meds['itemid'].unique()
                df2=meds[meds['stay_id']==hid]
                if df2.shape[0]==0:
                    amount=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    amount=amount.fillna(0)
                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                else:
                    rate=df2.pivot_table(index='start_time',columns='itemid',values='rate')
                    #print(rate)
                    amount=df2.pivot_table(index='start_time',columns='itemid',values='amount')
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='stop_time')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.ffill()
                    df2=df2.fillna(0)

                    rate=pd.concat([rate, add_df])
                    rate=rate.sort_index()
                    rate=rate.ffill()
                    rate=rate.fillna(-1)

                    amount=pd.concat([amount, add_df])
                    amount=amount.sort_index()
                    amount=amount.ffill()
                    amount=amount.fillna(-1)
                    #print(df2.head())
                    df2.iloc[:,0:]=df2.iloc[:,0:].sub(df2.index,0)
                    df2[df2>0]=1
                    df2[df2<0]=0
                    rate.iloc[:,0:]=df2.iloc[:,0:]*rate.iloc[:,0:]
                    amount.iloc[:,0:]=df2.iloc[:,0:]*amount.iloc[:,0:]
                    #print(df2.head())
                    dataDic[hid]['Med']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['rate']=rate.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Med']['amount']=amount.iloc[:,0:].to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(amount.columns)))
    #                 print(feat)
    #                 print(amount.columns)
    #                 print(amount.head())
                    amount=pd.concat([amount,feat_df],axis=1)

                    amount=amount[feat]
                    amount=amount.fillna(0)
    #                 print(amount.columns)
                    amount.columns=pd.MultiIndex.from_product([["MEDS"], amount.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=amount
                else:
                    dyn_csv=pd.concat([dyn_csv,amount],axis=1)


            ###LAB
            # LAB Processing
            if(self.feat_lab):
                feat = lab['itemid'].unique()  # unique lab test types
                df2 = lab[lab['stay_id'] == hid]  # extract lab data for the current patient
                if df2.shape[0] == 0:
                    # If no lab data exists for this patient
                    valuenum = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    valuenum = valuenum.fillna(0)
                    valuenum.columns = pd.MultiIndex.from_product([["LABS"], valuenum.columns])
                else:
                    # Process valuenum (continuous lab test values)
                    valuenum = df2.pivot_table(index='start_time', columns='itemid', values='valuenum')
                    df2 = df2.pivot_table(index='start_time', columns='itemid', values='stop_time')

                    # Handle missing time points
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.ffill().fillna(0)

                    # Fill missing values in valuenum
                    valuenum = pd.concat([valuenum, add_df])
                    valuenum = valuenum.sort_index()
                    valuenum = valuenum.ffill().fillna(-1)

                    # Binary mask logic: positive/negative time adjustments
                    df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                    df2[df2 > 0] = 1
                    df2[df2 < 0] = 0

                    # Apply the mask to valuenum
                    valuenum.iloc[:, 0:] = df2.iloc[:, 0:] * valuenum.iloc[:, 0:]

                    # Store the processed lab data
                    dataDic[hid]['Lab']['valuenum'] = valuenum.iloc[:, 0:].to_dict(orient="list")

                    # Handle missing lab test features
                    feat_df = pd.DataFrame(columns=list(set(feat) - set(valuenum.columns)))
                    valuenum = pd.concat([valuenum, feat_df], axis=1)
                    valuenum = valuenum[feat].fillna(0)

                    # Assign multi-level column headers for lab data
                    valuenum.columns = pd.MultiIndex.from_product([["LABS"], valuenum.columns])

                # Append lab data to the dynamic CSV
                if dyn_csv.empty:
                    dyn_csv = valuenum
                else:
                    dyn_csv = pd.concat([dyn_csv, valuenum], axis=1)

            # MICRO Processing
            if(self.feat_micro):
                feat = micro['spec_itemid'].unique()  # unique specimen IDs
                df2 = micro[micro['stay_id'] == hid]  # extract micro data for the current patient
                if df2.shape[0] == 0:
                    # If no microbiology data exists for this patient
                    spec_itemid = pd.DataFrame(np.zeros([los, len(feat)]), columns=feat)
                    spec_itemid = spec_itemid.fillna(0)
                    spec_itemid.columns = pd.MultiIndex.from_product([["MICRO"], spec_itemid.columns])
                else:
                    # Process spec_itemid (discrete specimen IDs)
                    #import pdb
                    #pdb.set_trace()
                    spec_itemid = df2.pivot_table(index='start_time', columns='test_itemid', values='spec_itemid', aggfunc='first')
                    df2 = df2.pivot_table(index='start_time', columns='spec_itemid', values='stop_time')

                    # Handle missing time points
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2 = pd.concat([df2, add_df])
                    df2 = df2.sort_index()
                    df2 = df2.ffill().fillna(0)

                    # Fill missing values in spec_itemid
                    spec_itemid = pd.concat([spec_itemid, add_df])
                    spec_itemid = spec_itemid.sort_index()
                    spec_itemid = spec_itemid.ffill().fillna(-1)

                    # Binary mask logic: positive/negative time adjustments
                    df2.iloc[:, 0:] = df2.iloc[:, 0:].sub(df2.index, 0)
                    df2[df2 > 0] = 1
                    df2[df2 < 0] = 0

                    # Apply the mask to spec_itemid
                    spec_itemid.iloc[:, 0:] = df2.iloc[:, 0:] * spec_itemid.iloc[:, 0:]

                    # Store the processed microbiology data
                    dataDic[hid]['Micro']['spec_itemid'] = spec_itemid.iloc[:, 0:].to_dict(orient="list")

                    # Handle missing specimen IDs
                    feat_df = pd.DataFrame(columns=list(set(feat) - set(spec_itemid.columns)))
                    spec_itemid = pd.concat([spec_itemid, feat_df], axis=1)
                    spec_itemid = spec_itemid[feat].fillna(0)

                    # Assign multi-level column headers for microbiology data
                    spec_itemid.columns = pd.MultiIndex.from_product([["MICRO"], spec_itemid.columns])

                # Append micro data to the dynamic CSV
                if dyn_csv.empty:
                    dyn_csv = spec_itemid
                else:
                    dyn_csv = pd.concat([dyn_csv, spec_itemid], axis=1)
                
                
                
            
            
            ###PROCS
            if(self.feat_proc):
                feat=proc['itemid'].unique()
                df2=proc[proc['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                else:
                    df2['val']=1
                    #print(df2)
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    #print(df2.head())
                    dataDic[hid]['Proc']=df2.to_dict(orient="list")


                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["PROC"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
                
                
                   
            ###OUT
            if(self.feat_out):
                feat=out['itemid'].unique()
                df2=out[out['stay_id']==hid]
                if df2.shape[0]==0:
                    df2=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                else:
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)
                    df2[df2>0]=1
                    #print(df2.head())
                    dataDic[hid]['Out']=df2.to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(df2.columns)))
                    df2=pd.concat([df2,feat_df],axis=1)

                    df2=df2[feat]
                    df2=df2.fillna(0)
                    df2.columns=pd.MultiIndex.from_product([["OUT"], df2.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=df2
                else:
                    dyn_csv=pd.concat([dyn_csv,df2],axis=1)
                
                
                
            ###CHART
            if(self.feat_chart):
                feat=chart['itemid'].unique()
                df2=chart[chart['stay_id']==hid]
                if df2.shape[0]==0:
                    val=pd.DataFrame(np.zeros([los,len(feat)]),columns=feat)
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
                else:
                    val=df2.pivot_table(index='start_time',columns='itemid',values='valuenum')
                    df2['val']=1
                    df2=df2.pivot_table(index='start_time',columns='itemid',values='val')
                    #print(df2.shape)
                    add_indices = pd.Index(range(los)).difference(df2.index)
                    add_df = pd.DataFrame(index=add_indices, columns=df2.columns).fillna(np.nan)
                    df2=pd.concat([df2, add_df])
                    df2=df2.sort_index()
                    df2=df2.fillna(0)

                    val=pd.concat([val, add_df])
                    val=val.sort_index()
                    if self.impute=='Mean':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.mean())
                    elif self.impute=='Median':
                        val=val.ffill()
                        val=val.bfill()
                        val=val.fillna(val.median())
                    val=val.fillna(0)


                    df2[df2>0]=1
                    df2[df2<0]=0
                    #print(df2.head())
                    dataDic[hid]['Chart']['signal']=df2.iloc[:,0:].to_dict(orient="list")
                    dataDic[hid]['Chart']['val']=val.iloc[:,0:].to_dict(orient="list")

                    feat_df=pd.DataFrame(columns=list(set(feat)-set(val.columns)))
                    val=pd.concat([val,feat_df],axis=1)

                    val=val[feat]
                    val=val.fillna(0)
                    val.columns=pd.MultiIndex.from_product([["CHART"], val.columns])
                
                if(dyn_csv.empty):
                    dyn_csv=val
                else:
                    dyn_csv=pd.concat([dyn_csv,val],axis=1)
            
            #Save temporal data to csv
            dyn_csv.to_csv('./data/csv/'+str(hid)+'/dynamic.csv',index=False)
            
            ##########COND#########
            if(self.feat_cond):
                feat=self.cond['new_icd_code'].unique()
                grp=self.cond[self.cond['stay_id']==hid]
                if(grp.shape[0]==0):
                    dataDic[hid]['Cond']={'fids':list(['<PAD>'])}
                    feat_df=pd.DataFrame(np.zeros([1,len(feat)]),columns=feat)
                    grp=feat_df.fillna(0)
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
                else:
                    dataDic[hid]['Cond']={'fids':list(grp['new_icd_code'])}
                    grp['val']=1
                    grp=grp.drop_duplicates()
                    grp=grp.pivot(index='stay_id',columns='new_icd_code',values='val').reset_index(drop=True)
                    feat_df=pd.DataFrame(columns=list(set(feat)-set(grp.columns)))
                    grp=pd.concat([grp,feat_df],axis=1)
                    grp=grp.fillna(0)
                    grp=grp[feat]
                    grp.columns=pd.MultiIndex.from_product([["COND"], grp.columns])
            grp.to_csv('./data/csv/'+str(hid)+'/static.csv',index=False)   
            labels_csv.to_csv('./data/csv/labels.csv',index=False)    
            
                
        ######SAVE DICTIONARIES##############
        metaDic={'Cond':{},'Proc':{},'Med':{},'Out':{},'Chart':{}, 'Lab': {}, 'Micro': {},'LOS':{}}
        metaDic['LOS']=los
        with open("./data/dict/dataDic", 'wb') as fp:
            pickle.dump(dataDic, fp)

        with open("./data/dict/hadmDic", 'wb') as fp:
            pickle.dump(self.hids, fp)
        
        with open("./data/dict/ethVocab", 'wb') as fp:
            pickle.dump(list(self.data['ethnicity'].unique()), fp)
            self.eth_vocab = self.data['ethnicity'].nunique()
            
        with open("./data/dict/ageVocab", 'wb') as fp:
            pickle.dump(list(self.data['Age'].unique()), fp)
            self.age_vocab = self.data['Age'].nunique()
            
        with open("./data/dict/insVocab", 'wb') as fp:
            pickle.dump(list(self.data['insurance'].unique()), fp)
            self.ins_vocab = self.data['insurance'].nunique()
            
        if(self.feat_med):
            with open("./data/dict/medVocab", 'wb') as fp:
                pickle.dump(list(meds['itemid'].unique()), fp)
            self.med_vocab = meds['itemid'].nunique()
            metaDic['Med']=self.med_per_adm

        if(self.feat_lab):
            with open("./data/dict/labVocab", 'wb') as fp:
                pickle.dump(list(meds['itemid'].unique()), fp)
            self.lab_vocab = meds['itemid'].nunique()
            metaDic['Lab']=self.lab_per_adm

        if(self.feat_micro):
            with open("./data/dict/microVocab", 'wb') as fp:
                pickle.dump(list(micro['itemid'].unique()), fp)
            self.micro_vocab = meds['test_itemid'].nunique()
            metaDic['Micro']=self.micro_per_adm
            
        if(self.feat_out):
            with open("./data/dict/outVocab", 'wb') as fp:
                pickle.dump(list(out['itemid'].unique()), fp)
            self.out_vocab = out['itemid'].nunique()
            metaDic['Out']=self.out_per_adm
            
        if(self.feat_chart):
            with open("./data/dict/chartVocab", 'wb') as fp:
                pickle.dump(list(chart['itemid'].unique()), fp)
            self.chart_vocab = chart['itemid'].nunique()
            metaDic['Chart']=self.chart_per_adm
        
        if(self.feat_cond):
            with open("./data/dict/condVocab", 'wb') as fp:
                pickle.dump(list(self.cond['new_icd_code'].unique()), fp)
            self.cond_vocab = self.cond['new_icd_code'].nunique()
            metaDic['Cond']=self.cond_per_adm
        
        if(self.feat_proc):    
            with open("./data/dict/procVocab", 'wb') as fp:
                pickle.dump(list(proc['itemid'].unique()), fp)
            self.proc_vocab = proc['itemid'].nunique()
            metaDic['Proc']=self.proc_per_adm
            
        with open("./data/dict/metaDic", 'wb') as fp:
            pickle.dump(metaDic, fp)
            
            
      


