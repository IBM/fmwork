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

def main(lh_table, csv_file):

    df = process_csv(csv_file)
    if df is not None:
        print(f"CSV file at {csv_file} loaded successfully.")

    lh = LakehouseIceberg()

    if table_exists(lh,lh_table):
        print("Table exists... appending to existing table")
        table = Table(lh=lh, namespace='fm_work', table_name=lh_table)
        table.append_dataframe(df=df)
    else:
        print("This table does not exist in the fm_work namespace in lakehouse... creating table")
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
        table = Table(lh=lh, namespace='fm_work', table_name=lh_table)
    
    print(table.to_pandas())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python lakehouse_append.py <table_name> <csv_file_path>")
        sys.exit(1)
    
    lh_table = sys.argv[1]
    csv_file = sys.argv[2]
    main(lh_table, csv_file)

