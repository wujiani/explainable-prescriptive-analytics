import pandas as pd
import os

import tqdm

os.getcwd()

df = pd.read_csv('data/completed.csv')
df_run = pd.DataFrame(columns=df.columns)
for idx in tqdm.tqdm(df['REQUEST_ID'].unique()):
    trace = df[df['REQUEST_ID'] == idx]
    trace = trace.iloc[:int(len(trace)*.80)]
    df_run = pd.concat([df_run, trace])
df_run.reset_index(drop=True, inplace=True)
print('Finito<>vez')
df_run.to_csv('data/bac_running.csv')

# import threading
# import time
#
#
# def useless_function(seconds):
#     print(f'Waiting for {seconds} second(s)', end="\n")
#     time.sleep(seconds)
#     print(f'Done Waiting {seconds}  second(s)')
#
#
# start = time.perf_counter()
# t = threading.Thread(target=useless_function, args=[3])
# t.start()
# print(f'Active Threads: {threading.active_count()}')
# t.join()
# end = time.perf_counter()
# print(f'Finished in {end - start} second(s)')


def cut_last_act(log, case_id_name, activity_name):
    log_ret = pd.DataFrame(columns=log.columns)
    last_activity = log_ret
    for idx in df['SR_Number'].unique():

