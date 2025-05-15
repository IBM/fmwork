from lakehouse import LakehouseIceberg, Table
from lakehouse.core import TableDetails
import pandas as pd
import sys


def table_exists(lh,lh_table):
    try:    
        table = Table(lh=lh, namespace='fm_work', table_name=lh_table)
        return True
    except Exception as e:
        return False

def process_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Failed to load CSV file: {e}")
        return None
    
def append_to_table(lh, lh_table, df):
    
    if table_exists(lh,lh_table):
        table = Table(lh=lh, namespace='fm_work', table_name=lh_table)
        existing_df = table.to_pandas()
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        unique_df = combined_df.drop_duplicates(subset='etim', keep=False)
        if unique_df.empty:
            print(lh_table, "table exists... no new data entries to append")
        else:
            print(lh_table, "table exists... appending new data to existing table")
            unique_df = unique_df.fillna('')
            table.append_dataframe(unique_df)
    else:
        print("This table does not exist in the fm_work namespace in lakehouse... creating", lh_table, "table")
        table_details = TableDetails(
            namespace = 'fm_work',
            name = lh_table,
            identifier_fields=['etim'],
            mandatory_fields = ['work','user','host',
                                'btim','etim','hw',
                                'hwc','back','mm',
                                'prec','dp','ii',
                                'oo','bb','tp',
                                'med','ttft','gen',
                                'itl','thp'],
            is_public = True,
        )

        Table.from_dataframe(
            lh = lh,
            df = df,
            table_details = table_details,
        )
    
def main(lh_table, csv_file):

    df = process_csv(csv_file)
    if df is not None:
        print(f"CSV file at {csv_file} loaded successfully.")

    lh = LakehouseIceberg()
    backend_table = lh_table.split('_')[0]
    hw_table = lh_table.split('_')[1]
    append_to_table(lh, lh_table, df)
    append_to_table(lh, backend_table, df)
    append_to_table(lh, hw_table, df)

    print("Done. All CSV data inserted into the following Lakehouse tables:",lh_table,backend_table,hw_table)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python lakehouse_append.py <table_name> <csv_file_path>")
        sys.exit(1)
    
    lh_table = sys.argv[1]

    valid_tablenames = [
        "vllm_cuda",
        "vllm_gaudi",
        "vllm_rocm",
        "vllm_spyre",
        "trtllm_cuda",
        "transformers_cuda"
    ]

    if lh_table not in valid_tablenames:
        print(f"Error: table name must be one of {valid_tablenames}")
        sys.exit(1)

    csv_file = sys.argv[2]
    main(lh_table, csv_file)
