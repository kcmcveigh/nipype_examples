import os
import glob
import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch
from collections import namedtuple


def CreateTextFileFromOneCondition(behave_data,
                                   condition_int,
                                   condition_str,
                                   run_path):
    text_file = np.zeros((4,3))
    condition_data = behave_data.loc[behave_data.condition==str(condition_int)]
    
    text_file[:,0]= condition_data.onsets.values
    text_file[:,1]=5
    text_file[:,2]= condition_data.rating.values
    
    save_path = os.path.join(run_path,condition_str)

    text_file = np.nan_to_num(text_file)
    np.savetxt(save_path,text_file)

def CreateRegressorTextFiles(subject_behavioural, 
                             output_dir,
                             subject_num,
                             column_dict = {'condition':1,'onsets':9,'offset':10,'rating':12},
                             n_trials=12,
                             start_index=0):
    #get how many runs total
    n_runs = round((subject_behavioural.shape[0]+1)/13)
    columns_of_interest = list(column_dict.values())
    column_names =list(column_dict.keys())
    
    #make directory for all runs
    events_dir = os.path.join(output_dir,f'subject_{subject_num}_events')
    os.mkdir(events_dir)

    for i in range(n_runs):
            #get run data for one run
            run_num = i + 1
            end_index = start_index + n_trials
            run_data = subject_behavioural.iloc[start_index:end_index,columns_of_interest]
            run_data.columns = column_names
            #make directory for run
            run_dir = os.path.join(events_dir,f'run_0{run_num}')
            os.mkdir(run_dir)
            
            #make files in each run
            CreateTextFileFromOneCondition(run_data,1,'heights', run_dir)
            CreateTextFileFromOneCondition(run_data,2,'spiders', run_dir)
            CreateTextFileFromOneCondition(run_data,3,'pain', run_dir)
            start_index += n_trials + 1

def CreateArrayEVRegressors(regressor_files, n_trials, parametric=False):
    n_regressors = len(regressor_files)
    print('ev array regressors')
    onsets = np.zeros((n_trials,n_regressors))
    durations = np.zeros((n_trials,n_regressors))
    amplitudes = np.zeros((n_trials,n_regressors))
    names = []
    for i in range(n_regressors):
        ev_df = pd.read_csv(regressor_files[i], sep=' ',header=None)
        onsets[:,i] = ev_df.iloc[:,0]
        durations[:,i] = ev_df.iloc[:,1]
        amplitudes[:,i] =  ev_df.iloc[:,2] - np.mean(ev_df.iloc[:,2]) if parametric else np.ones_like(ev_df.iloc[:,1])
        names.append(regressor_files[i].split('/')[-1]) #this naming scheme is a little dangerous as it relies
        #on the files being named correctly for the contrasts
        #if the name does not exactly match the contrast name the contrast won't work

    regressor_info = namedtuple('EV_Regressors', 'onsets durations amplitudes names')
    return regressor_info(onsets,durations,amplitudes,names)

def CreateEVRegressors(regressor_files, n_trials, parametric=False):
    non_parametric = CreateArrayEVRegressors(regressor_files, n_trials, False)

    if parametric:
        parametric = CreateArrayEVRegressors(regressor_files, n_trials, True)
        parametric_names = [name+'_parametric' for name in parametric.names]

        regressors_info = namedtuple('EV_Regressors', 'onsets durations amplitudes names')
        regressors_info.amplitudes = np.append(non_parametric.amplitudes, parametric.amplitudes, axis=1)
        regressors_info.onsets = np.append(non_parametric.onsets, parametric.onsets, axis=1)
        regressors_info.durations = np.append(non_parametric.durations, parametric.durations, axis=1)
        regressors_info.names = np.append(non_parametric.names, parametric_names)

    else:
        regressors_info = non_parametric


    return regressors_info



def CreateSubjectInfoRunBunch(regressor_files,
                             confounds_file,
                             parametric = False,
                             n_trials=4
                            ):
    
    #get confounds 
    confound_df = pd.read_csv(confounds_file, sep='\t')
    nuisance_columns = [column for column in confound_df.columns if 'AROMA' in column]
    nuisance_regressors = confound_df[nuisance_columns].values

    ev_regressors = CreateEVRegressors(regressor_files, n_trials, parametric=parametric)

    subject_info_bunch = Bunch(conditions=ev_regressors.names,
                              onsets=ev_regressors.onsets.T,
                              amplitudes=ev_regressors.amplitudes.T.tolist(),
                              durations=ev_regressors.durations.T,
                              regressors=nuisance_regressors.T,
                              regressor_names = nuisance_columns
                              )
    return subject_info_bunch

def CreateSubjectInfoAllRuns(subject_num,
                             regressor_base_paths,
                             confounds_base_paths,
                             parametric=False):

    regressors_run_paths = os.path.join(regressor_base_paths,f'subject_{subject_num}_events/run_*')
    regressors_run_folders = glob.glob(regressors_run_paths)
    confounds_path = os.path.join(confounds_base_paths,f'sub-{subject_num}/func/*confounds.tsv')
    confound_files = glob.glob(confounds_path)
    subject_info_list = []
    
    if (len(confound_files)==len(regressors_run_folders)):
        for i in range(len(regressors_run_folders)):
                regressor_paths = os.path.join(regressors_run_folders[i],'*')
                regressor_files = glob.glob(regressor_paths)
                confound_file = confound_files[i]
                run_bunch = CreateSubjectInfoRunBunch(regressor_files,confound_file, parametric = parametric)
                subject_info_list.append(run_bunch)
    else:
        print("ERROROR NOT THE SAME LENGTH")
        
        print(confounds_path)
        print(confound_files)
        
        print(regressors_run_paths)
        print(regressors_run_folders)
    
    return subject_info_list

