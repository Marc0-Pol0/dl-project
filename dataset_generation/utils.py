# Imports
import pandas as pd
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import warnings


from theme_attention.rnd import llmTools as llm 
from theme_attention.rnd import basicTools as bt
from theme_attention.pipeline import dbTool as dbt 


from theme_attention.rnd import PMTools as pm
from theme_attention.pipeline import themeClean as tc
from theme_attention.pipeline import dbTool as dbt 
from theme_attention.rnd import EODTools as edt
from theme_attention.rnd import PMTools as pm
from theme_attention.rnd import factor as ft
from theme_attention.rnd import weighting as wt
from theme_attention.rnd import selection as se
from theme_attention.rnd  import basicTools as bt
from theme_attention.rnd import marketsTool as tools

import warnings

from base.sql import eqitron as eq
from base.sql import factset as fs
from base.storage import azblob as azt
from base import tkmanager as tm

from eod import EodHistoricalData


# EOD Historical Data client
EOD_API = tm.get_EOD_Key()
client = EodHistoricalData(EOD_API)





warnings.filterwarnings("ignore")




def get_fundamentals(factset_id, startDate, endDate, config): 
    startDate = startDate 
    dataConfig = config['data']

    dataDict = {}
    fundamentalConfig = dataConfig['fundamental']

    for item, thisDataC in fundamentalConfig.items():
        field = thisDataC['field']
        freqD = thisDataC['frequencyDependency']
        dataDict[item] = fs.getFundamentals([factset_id], field, freqD, startDate, endDate)

    return dataDict





def get_ratios(factset_id, startDate, endDate, config): 
    buffer = 16
    startDate = startDate - relativedelta(months=buffer) # subtracting some months from the starting date in needed to add buffer to start date because of how data is processed in the function
    dataConfig = config['data']

    # 1) Load base fundamentals
    dataDict = {}
    fundamentalConfig = dataConfig['fundamental']

    for item, thisDataC in fundamentalConfig.items():
        field = thisDataC['field']
        freqD = thisDataC['frequencyDependency']
        dataDict[item] = fs.getFundamentals([factset_id], field, freqD, startDate, endDate)

        # Renaming
        if freqD == 1 and 'ltm' in dataDict[item].columns:
            dataDict[item] = dataDict[item].rename(columns={'ltm': 'qf'})

    # 2) Build computed fundamentals 
    for synthetic_name, spec in dataConfig.get('computed_fundamental', {}).items():
        op = spec['operation'].lower()
        a, b = spec['operands']

        if a not in dataDict or b not in dataDict:
            raise KeyError(f"Missing operand(s) for {synthetic_name}: {a}, {b}")

        A, B = dataDict[a], dataDict[b]

        merge_cols = ['fsym_id', 'date']
        value_cols = ['af', 'saf', 'qf']

        merged = pd.merge(
            A[merge_cols + value_cols],
            B[merge_cols + value_cols],
            on=merge_cols,
            suffixes=('_A', '_B')
        )

        C = merged[merge_cols].copy()
        for col in value_cols:
            if op == 'minus':
                C[col] = merged[f'{col}_A'] - merged[f'{col}_B']
            elif op == 'add':
                C[col] = merged[f'{col}_A'] + merged[f'{col}_B']
            elif op == 'multiply':
                C[col] = merged[f'{col}_A'] * merged[f'{col}_B']
            elif op == 'divide':
                denom = merged[f'{col}_B']
                try:
                    denom_safe = denom.replace(0, np.nan)
                except AttributeError:
                    denom_safe = np.where(denom == 0, np.nan, denom)
                C[col] = merged[f'{col}_A'] / denom_safe
            else:
                raise ValueError(f"Unsupported operation: {op}")

        dataDict[synthetic_name] = C  # expose like any other field

    # 3) Construct ratios directly into a dataframe
    final_df = pd.DataFrame()
    ratioConfig = dataConfig['ratio']

    for item, thisDataC in ratioConfig.items():
        ratio = ft.fundamental_ratio([factset_id], dataDict, {item: thisDataC})

        # Remove rows where 'fsym_id' is NaN in the DataFrame
        ratio[item] = ratio[item].loc[~ratio[item]['fsym_id'].isna()]

        ratioMat = ft.fundamental_ratio_process([factset_id], ratio, {item: thisDataC}, startDate, endDate) 
        
        # ratioMat is {item: df}, get the DataFrame
        df = ratioMat[item]

        # take the only column (fsym_id) and rename it to the ratio name
        series = df.iloc[:, 0].rename(item)

        # Normalize percentages to normal numbers if 'format' exists and is 'percentage'
        if 'format' in thisDataC and thisDataC['format'] == 'percentage':
            series = series / 100  # Directly normalize the single column

        # join into final_df by index (date)
        final_df = series.to_frame() if final_df.empty else final_df.join(series, how="outer")

    # make sure index is datetime and sorted
    final_df.index = pd.to_datetime(final_df.index, errors="coerce")
    final_df = final_df.sort_index()

    # Replace non-finite values (inf, -inf) with NaN
    final_df = final_df.applymap(lambda x: x if np.isfinite(x) else np.nan)

    # Remove the first 'buffer' rows
    final_df = final_df.iloc[buffer+1:]


    return final_df
    




# download company description from EOD and save to local path
# description is saved as a parquet file with gzip compression
# to load the file, use pd.read_parquet(file_path)
def update_company_description(Code:list,local_path:str,file_name_root:str):

    stockDescription = dict()
    for id in Code:
        print(id)
        thisDesp = client.get_fundamental_equity(id,filter_='General::Description') 
        stockDescription[id] = thisDesp
    df = pd.DataFrame.from_dict(stockDescription,orient='index',columns=['Description'])
    file_name_root = file_name_root + '.parquet.gzip'
    df.to_parquet(os.path.join(local_path,file_name_root),compression='gzip')

    return 'Descript update successfully'


