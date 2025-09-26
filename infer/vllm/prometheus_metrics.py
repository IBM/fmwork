"""Script to get cpu and memory metrics from given container running in kube cluster

Setup:
pip3 install prometheus-api-client

Example command:

python3 prometheus_metrics.py  --pod-name <POD-NAME> --namespace <NAMESPACE> --container-name <CONTAINER-NAME> --start-time 1756333696 --end-time $(date +%s)

"""

# Assisted by watsonx Code Assistant

import argparse
import os
import datetime
from enum import Enum
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo


from prometheus_api_client import PrometheusConnect



#### CONFIGURATIONS
THANOS_API_TOKEN = os.getenv("THANOS_API_TOKEN", None)
THANOS_API_URL = os.getenv("THANOS_API_URL", "http://localhost:19090")


### CONSTANTS
STEP = "5m" # Representing query resolution step width


class QueryFunction(Enum):
    AVG = "avg_over_time"
    MAX = "max_over_time"


def get_pod_metrics_at_timestamp(prometheus_url, start_time, end_time, query=None, metric_name=None) -> Dict[str, Any]:
    """
    Collects a specific Prometheus metric for a given pod at a specified timestamp.

    Args:
        prometheus_url (str): The URL of your Prometheus server (e.g., "http://localhost:9090").
        labels (str): Kubernetes label configuration
        start_time (datetime): The timestamp at which to query the metric.
        end_time (datetime): The timestamp at which to query the metric.
        metric_name (str): The name of the Prometheus metric to query (e.g., "container_cpu_user_seconds_total").
        query_fn (str): The vector function to be used for querying (default is "avg_over_time").

    Returns:
        Dict[str, Any]: A dictionary containing the metric data, or None if an error occurs.

    Raises:
        ValueError: If metric_name is not specified.
    """

    if query is None:
        raise ValueError("Metric name must be specified")

    try:
        headers = {"Authorization": f"Bearer {THANOS_API_TOKEN}"}
        prom = PrometheusConnect(url=prometheus_url, headers=headers)

        # Execute the query
        result = prom.custom_query_range(query=query, start_time=start_time, end_time=end_time, step=STEP)


        if result:
            return result  # Return the result as-is
        else:
            raise ValueError(f"No data found for metric '{metric_name}' between {start_time} and {end_time}.")

    except Exception as e:
        raise ValueError(f"Error connecting to Prometheus or querying metrics: {e}")



def process_results(query_fn, results):
    """
    Process results based on the specified query function.

    Args:
        query_fn (QueryFunction): The type of query function to apply.
        results (list): The list of numerical results to process.

    Returns:
        float: The result of the query function applied to the results.

    Raises:
        ValueError: If the query function is not recognized.
    """
    if query_fn == QueryFunction.AVG:
        return sum(results) / len(results)
    elif query_fn == QueryFunction.MAX:
        return max(results)
    # To be used for python 3.10+
    # match query_fn:
    #     case QueryFunction.AVG:
    #         return sum(results) / len(results)
    #     case QueryFunction.MAX:
    #         return max(results)


def get_cpu_metrics(prometheus_url: str, pod_label_config: str, start_time, end_time, query_fn=QueryFunction.AVG) -> Optional[dict]:
    """Fetches CPU metrics for a specified pod in a given namespace from Prometheus and prints the results.

    Args:
        prometheus_url (str): The URL of the Prometheus server.
        pod_label_config (str): Kubernetes label configuration
        start_time (int): The start timestamp for the query in Unix epoch time.
        end_time (int): The end timestamp for the query in Unix epoch time.
        query_fn (str, optional): The Prometheus query function. Default is "avg_over_time".

    Returns:
        Optinal(dict)
    """
    METRIC_NAME = "container_cpu_usage_seconds_total"


    query = f'rate({METRIC_NAME}{pod_label_config}[{STEP}])'

    # Fetch the metric data for the given pod, namespace, and time range
    metric_data = get_pod_metrics_at_timestamp(prometheus_url, start_time, end_time, metric_name=METRIC_NAME, query=query)

    metric_collection = None
    if metric_data:

        cores_used_list = []
        cores_percent_list = []
        for result in metric_data:
            metric_labels = result['metric']
            time_series = result['values']

            print(f"\tCPU Metrics:")

            for timestamp, value in time_series:
                # Convert the timestamp to a human-readable format
                # readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

                # Convert the value to a float for calculations
                cpu_cores_used = float(value)
                cpu_percentage = cpu_cores_used * 100
                cores_used_list.append(cpu_cores_used)
                cores_percent_list.append(cpu_percentage)

            result_cores_used = process_results(query_fn, cores_used_list)
            result_cores_percentage = process_results(query_fn, cores_percent_list)
            # NOTE: This will return the last value returned by the metric_data
            metric_collection = {
                "cores": round(result_cores_used, 3),
                "percentage": round(result_cores_percentage, 3)
            }
            print(f"\t\tCores: {result_cores_used:.3f}, Percentage: {result_cores_percentage:.3f}%")

        return metric_collection


def get_memory_metrics(prometheus_url: str, pod_label_config, start_time, end_time, query_fn=QueryFunction.AVG) -> Optional[dict]:
    """
    Fetches time-series data for a given metric from Prometheus for a specific pod and namespace within a time range.

    Args:
        prometheus_url (str): The URL of the Prometheus server.
        pod_label_config (str): The name of the pod to retrieve metrics for.
        start_time (int): The start timestamp for the query in Unix epoch time.
        end_time (int): The end timestamp for the query in Unix epoch time.
        metric_name (str): The name of the Prometheus metric to query.
        query_fn (str): The Prometheus query function.

    Returns:
        Optional(dict): dictionary containing the memory metrics
    """
    # container_memory_working_set_bytes is the amount of memory the container is actively. A container will get killed if it uses
    # memory exceeding this memory limit.
    METRIC_NAME = "container_memory_working_set_bytes"

    metric_collection = None

    query = f'{query_fn.value}({METRIC_NAME}{pod_label_config}[{STEP}])'
    # Fetch the metric data for the given pod, namespace, and time range
    metric_data = get_pod_metrics_at_timestamp(prometheus_url, start_time, end_time, metric_name=METRIC_NAME, query=query)

    if metric_data:

        mem_bytes_list = []
        mem_mb_list = []
        mem_gb_list = []
        for result in metric_data:
            metric_labels = result['metric']
            time_series = result['values']
            print(f"\tMemory Metrics:")
            for timestamp, value in time_series:
                readable_time = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                memory_bytes = float(value)
                memory_mb = memory_bytes / 1024 / 1024
                memory_gb = memory_mb / 1024

                mem_bytes_list.append(memory_bytes)
                mem_mb_list.append(memory_mb)
                mem_gb_list.append(memory_gb)

            result_mem_bytes = process_results(query_fn, mem_bytes_list)
            result_mem_mb = process_results(query_fn, mem_mb_list)
            result_mem_gb = process_results(query_fn, mem_gb_list)

            # Store in dict
            # NOTE: This will return the last value returned by the metric_data
            # But the query is currently only supposed to have 1 value
            metric_collection = {
                "mb": round(result_mem_mb, 3),
                "gb": round(result_mem_gb, 3)
            }

            print(f"\t\tBytes: {result_mem_bytes:.3f}, MB: {result_mem_mb:.3f}, GB: {result_mem_gb:.3f}")

        return metric_collection





def get_arguments(argparse):

    parser = argparse.ArgumentParser(description="Fetch resource metrics for a Kubernetes pod.")

    parser.add_argument("--pod-name", type=str, required=True, help="Name of the pod")
    parser.add_argument("--namespace", type=str, required=True, help="Namespace of the pod")
    parser.add_argument("--container-name", type=str, required=True, help="Name of the container inside the pod")
    parser.add_argument("--start-time", type=float, required=True, help="Start timestamp in Unix epoch")
    parser.add_argument("--end-time", type=float, required=True, help="End timestamp in Unix epoch")
    parser.add_argument("--api-token", type=str, default=None, required=False, help="Api token to access thanos apis")
    parser.add_argument("--timezone", type=str, default="UTC", required=False, help="Timezone to use to query prometheus")

    return parser.parse_args()

if __name__ == "__main__":

    args = get_arguments(argparse)

    if not args.api_token and not THANOS_API_TOKEN:
        raise ValueError("API_TOKEN needs to be provided either via env variable or argument")

    if args.api_token:
        API_TOKEN = args.api_token

    # Configuration
    pod_name = args.pod_name
    namespace = args.namespace
    container_name = args.container_name

    timezone = ZoneInfo(args.timezone)

    print("Timezone used: ", timezone)
    start_time = datetime.datetime.fromtimestamp(args.start_time, tz=timezone)
    end_time = datetime.datetime.fromtimestamp(args.end_time, tz=timezone)

    pod_label_config = f'{{namespace="{namespace}", pod="{pod_name}", container="{container_name}"}}'

    query_fns = [QueryFunction.AVG, QueryFunction.MAX]

    for query_fn in query_fns:

        print(f"\nQuery Fn: {query_fn}")

        get_cpu_metrics(THANOS_API_URL, pod_label_config, start_time, end_time, query_fn=query_fn)

        get_memory_metrics(THANOS_API_URL, pod_label_config, start_time, end_time, query_fn=query_fn)

        print("======================================")