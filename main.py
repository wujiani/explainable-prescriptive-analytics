
import argparse
import json
import os
import shutil
import warnings

# REFACTOR
from os.path import join

import numpy as np
import pandas as pd

# Converter
import pm4py

from IO import read, folders, create_folders
from load_dataset import prepare_dataset

# Create backup folder
if not os.path.exists('explanations'):
    os.mkdir('explanations')
if not os.path.exists('experiments'):
    os.mkdir('experiments')


def convert_to_csv(filename=str):
    if '.csv' in filename:
        return None
    if '.xes.gz' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-7] + '.csv'), index=None)
        print('Conversion ok')
        return None
    if '.xes' in filename:
        pm4py.convert_to_dataframe(pm4py.read_xes(filename)).to_csv(path_or_buf=(filename[:-4] + '.csv'), index=None)
        print('Conversion ok')
    else:
        raise TypeError('Check the path or the log type, admitted formats : csv, xes, xes.gz')


def modify_filename(filename):
    if '.csv' in filename: return filename
    if '.xes.gz' in filename: return filename[:-7] + '.csv'
    if '.xes' in filename:
        return filename[:-4] + '.csv'
    else:
        None

def convert_predictor(predictor):
    if predictor=='catboost':
        return 0
    elif predictor=='linear':
        return 1
    elif predictor=='neural_network':
        return 2
    else:
        raise ValueError('Please insert a valid type of predictor, the allowed are linear, catboost, neural_network')


def read_data(filename, start_time_col, date_format="%Y-%m-%d %H:%M:%S"):
    if '.csv' in filename:
        try:
            df = pd.read_csv(filename, header=0, low_memory=False)
            # if df.columns[0]=='Unnamed: 0':
            #     df = df.iloc[:,1:]
        except UnicodeDecodeError:
            df = pd.read_csv(filename, header=0, encoding="cp1252", low_memory=False)
    elif '.parquet' in filename:
        df = pd.read_parquet(filename, engine='pyarrow')
    # if a datetime cast it to seconds
    if not np.issubdtype(df[start_time_col], np.number):
        df[start_time_col] = pd.to_datetime(df[start_time_col], format=date_format)
        df[start_time_col] = df[start_time_col].astype(np.int64) / int(1e9)
    return df


parser = argparse.ArgumentParser(
    description='Main script for Catboost training')

#TODO : Nota che ho aggiunto il default a --case_id_name, --activity_name, --start_date_name per passarli al debugger

parser.add_argument('--filename_completed', default='data/completed.csv')
parser.add_argument('--filename_running', default=None)
parser.add_argument('--case_id_name', default='REQUEST_ID', type=str)
parser.add_argument('--activity_name', default='ACTIVITY', type=str)
parser.add_argument('--start_date_name', default='START_DATE', type=str)
parser.add_argument('--end_date_name', type=str, default='END_DATE')
parser.add_argument('--resource_name', type=str, default='CE_UO')
parser.add_argument('--role_name', type=str, default='ROLE')
parser.add_argument('--predict_activities', type=str, nargs='*', default=None)
parser.add_argument('--retained_activities', type=str, nargs='*', default=None)
parser.add_argument('--lost_activities', type=str, nargs='*', default=None)

parser.add_argument('--date_format', default="%Y-%m-%d %H:%M:%S")
parser.add_argument('--pred_column', default='lead_time')
parser.add_argument('--experiment_name', default='time_BAC')
parser.add_argument('--pred_attributes', type=str, nargs='*', default=None)
parser.add_argument('--costs', default=None)
parser.add_argument('--working_time', default=None)
parser.add_argument('--mode', default="train")

parser.add_argument('--override', default=True)  # if True retrains model and overrides previous one
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--outlier_thrs', default=0, type=int)
# parser.add_argument('--predictor', default='neural_network', type=str)

args = parser.parse_args()

# mandatory parameters
filename = args.filename_completed
filename_running = args.filename_running
case_id_name = args.case_id_name
activity_name = args.activity_name
start_date_name = args.start_date_name
date_format = args.date_format
pred_column = args.pred_column  # remaining_time if you want to predict that
experiment_name = args.experiment_name
# experiment_name = create_experiment_name(filename, pred_column)


if args.costs is not None:
    costs = json.loads(args.costs)
    working_time = json.loads(args.working_time)
else:
    costs = None
    working_time = None

# optional parameters
end_date_name = args.end_date_name
resource_column_name = args.resource_name
role_column_name = args.role_name
predict_activities = args.predict_activities
retained_activities = args.retained_activities
lost_activities = args.lost_activities
pred_attributes = args.pred_attributes
override = args.override
num_epochs = args.num_epochs
outlier_thrs = args.outlier_thrs
# predictor= args.predictor

convert_to_csv(filename)
filename = modify_filename(filename)
df = read_data(filename, args.start_date_name, args.date_format)
print(df.shape)
if filename_running is not None:
    df_running = read_data(filename_running, args.start_date_name, args.date_format)
    print(df_running.shape)

create_folders(folders, safe=override)

use_remaining_for_num_targets = None
custom_attribute_column_name = None
if pred_column == "total_cost":
    pred_column = "case_cost"
    use_remaining_for_num_targets = False
elif pred_column == "remaining_cost":
    pred_column = "case_cost"
    use_remaining_for_num_targets = True
elif pred_column == "remaining_time":
    use_remaining_for_num_targets = True
elif pred_column == "lead_time":
    pred_column = "remaining_time"
    use_remaining_for_num_targets = False
elif pred_column != "independent_activity" and pred_column != "churn_activity":
    # we can also predict a custom attribute
    if pred_column in df.columns:
        custom_attribute_column_name = pred_column
        pred_column = "custom_attribute"
    else:
        raise NotImplementedError

# predictor = convert_predictor(predictor)

np.random.seed(1618)  # 6415

prepare_dataset(df=df, case_id_name=case_id_name, activity_column_name=activity_name, start_date_name=start_date_name,
                date_format=date_format,
                end_date_name=end_date_name, pred_column=pred_column, mode="train", experiment_name=experiment_name,
                override=override,#predictor=predictor,
                pred_attributes=pred_attributes, costs=costs,
                working_times=working_time, resource_column_name=resource_column_name,
                role_column_name=role_column_name,
                use_remaining_for_num_targets=use_remaining_for_num_targets,
                predict_activities=predict_activities, lost_activities=lost_activities,
                retained_activities=retained_activities, custom_attribute_column_name=custom_attribute_column_name)

# copy results as a backup
fromDirectory = join(os.getcwd(), 'experiment_files')
toDirectory = join(os.getcwd(), 'experiments', experiment_name)

if os.path.exists(toDirectory):
    answer = None
    while answer not in {'y', 'n'}:
        print('An experiment with this name already exists, do you want to replace the folder storing the data ? [y/n]')
        answer = input()
    if answer == 'y':
        shutil.rmtree(toDirectory)
        shutil.copytree(fromDirectory, toDirectory)
    else:
        print('Backup folder not created')
else:
    shutil.copytree(fromDirectory, toDirectory)
    print('Data and results saved')

print('Analyze variables...')
# quantitative_vars, qualitative_trace_vars, qualitative_vars = utils.variable_type_analysis(X_train, case_id_name,
#                                                                                            activity_name)
warnings.filterwarnings("ignore")
print('Variable analysis done')

if __name__ == '__main__':
    print('run')