from lakehouse import LakehouseIceberg
from lakehouse.dataset_info import DatasetInfo
from lakehouse.assets.dataset import Dataset
import pandas as pd

lh = LakehouseIceberg()

# check if table for backend exists

# if yes, append csv data to table

# if no, create table with appropriate columns for specific backend

# also append data to consolidated table for all backends

# data = {
#    'col_1': [5, 6, 7, 8], 
#    'col_2': ['e', 'f', 'g', 'h'],
# }

df = pd.DataFrame.from_dict(data)

dataset = Dataset(lh=lh, dataset_name='data', namespace='fmwork')

dataset.append_dataframe(df=df)