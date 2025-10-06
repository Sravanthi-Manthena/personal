import itertools
import json
import pandas as pd
from pymongo import MongoClient
import boto3
from io import BytesIO


def get_all_periods(*dfs):
    """Collect all unique (year, quarter) and (year, month) from given dfs"""
    years, quarters, months = set(), set(), set()
    for df in dfs:
        if df is not None and not df.empty:
            years.update(df["Year"].unique())
            quarters.update(zip(df["Year"], df["Quarter"]))
            months.update(zip(df["Year"], df["Month"]))
    return sorted(years), sorted(quarters), sorted(months)


def get_values_from_df(df, combo, measure, dimensions_list):
    """
    Extract values for given dimension combo.
    Returns dict with only the quarters and months that exist in the dataset.
    Missing quarters/months are skipped.
    """

    # Ensure all dimension columns are strings (dynamically)
    if dimensions_list is None:
        dimensions_list = combo.keys()  # fallback

    for col in dimensions_list:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Filter rows matching combo dynamically
    subset = df.copy()
    for dim_col, dim_val in combo.items():
        if dim_col in subset.columns:
            subset = subset[subset[dim_col] == str(dim_val)]

    if subset.empty:
        return {}  # Return empty if no rows match

    # Convert Month to datetime safely
    subset['Month_dt'] = pd.to_datetime(subset['Month'], format='%b-%Y', errors='coerce')

    # Aggregate totals
    quarter_groups = subset.groupby(["Year", "Quarter"])[measure].sum().to_dict()
    month_groups = subset.groupby(subset['Month_dt'])[measure].sum().to_dict()

    # Define quarters to months mapping
    months_map = {
        "Q1": ["Jan", "Feb", "Mar"],
        "Q2": ["Apr", "May", "Jun"],
        "Q3": ["Jul", "Aug", "Sep"],
        "Q4": ["Oct", "Nov", "Dec"]
    }

    values = {}
    years = sorted(subset["Year"].unique())

    for year in years:
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            quarter_key = (year, f"{year}{q}")
            if quarter_key in quarter_groups:
                # Include quarter only if it exists
                values[f"{q}({year})"] = float(quarter_groups[quarter_key])

                # Include months of this quarter only if they exist
                for month_name in months_map[q]:
                    month_dt_candidates = subset[
                        (subset['Month'].str.startswith(month_name)) &
                        (subset['Year'] == year)
                    ]
                    if not month_dt_candidates.empty:
                        month_val = month_dt_candidates[measure].sum()
                        values[f"{month_name}({year})"] = float(month_val)

    return values


def build_exploded_json(hierarchies, dimensions_list, actual_df=None, forecast_df=None, budget_df=None):
    # Collect all periods across available dfs
    all_years, all_quarters, all_months = get_all_periods(actual_df, forecast_df, budget_df)

    dim_values = {}
    for dim in dimensions_list:
        dim_dict = hierarchies.get(dim, {})
        dim_values[dim] = []
        for pid in dim_dict.keys():
            dim_values[dim].append({
                "name":dim,
                "value": pid,
                "hasChildren": bool(dim_dict[pid]),
                "drillDownId": pid
            })

    all_combinations = list(itertools.product(*[dim_values[dim] for dim in dimensions_list]))
    data = []

    for combo in all_combinations:
        dim_entry = {}
        combo_dict = {}
        for dim, val in zip(dimensions_list, combo):
            dim_entry[dim] = val  # <-- assign dict directly, no list
            combo_dict[dim] = val["value"]

        # First dimension value will be the record id
        # record_id = combo_dict[dimensions_list[0]]
        record_id = "-".join([combo_dict[dim] for dim in dimensions_list])

        record = {
            "id": record_id,
            "hierarchy_level": 0,
            "dimensions": dim_entry,  # now this is the structure you want
            "recommendation_config": {
                "hierarchy_rows": dimensions_list
            } 
        }

        # Add values only if df exists and not empty
        if actual_df is not None and not actual_df.empty:
            record["actual_values"] = get_values_from_df(actual_df, combo_dict, "amount", dimensions_list)
        if budget_df is not None and not budget_df.empty:
            record["budget_values"] = get_values_from_df(budget_df, combo_dict,"budget_amount", dimensions_list)
        if forecast_df is not None and not forecast_df.empty:
            record["forecast_values"] = get_values_from_df(forecast_df, combo_dict, "amount", dimensions_list)


        data.append(record)
    response = {"success": True, "data": data}
    response = json.dumps(response, indent=4)

    return response

def drilldown_exploded_json(hierarchies, dimensions_list, dimension_clicked, parent_id, parent_combo,
                            actual_df=None, forecast_df=None, budget_df=None):

    dim_index = dimensions_list.index(dimension_clicked)

    # ✅ Children should come from the clicked dimension directly
    children = hierarchies.get(dimension_clicked, {}).get(parent_id, {})

    data = []
    for child_id in children.keys():
        combo_dict = parent_combo.copy()
        combo_dict[dimension_clicked] = child_id  # update clicked dimension with child

        # ✅ Carry forward values from previous dimensions
        composite_id = "-".join([combo_dict[dim] for dim in dimensions_list])

        record = {
            "id": composite_id,
            "hierarchy_level": dim_index,
            "dimensions": {
                dim: {
                    "name": dim,
                    "value": combo_dict[dim],
                    "hasChildren": bool(hierarchies.get(dim, {}).get(combo_dict[dim], {})),
                    "drillDownId": combo_dict[dim]
                }
                for dim in dimensions_list
            },
            "recommendation_config": {
                "hierarchy_rows": dimensions_list
            }
        }

        # attach values
        if actual_df is not None and not actual_df.empty:
            record["actual_values"] = get_values_from_df(actual_df, combo_dict, "amount", dimensions_list)
        if budget_df is not None and not budget_df.empty:
            record["budget_values"] = get_values_from_df(budget_df, combo_dict,"budget_amount", dimensions_list)
        if forecast_df is not None and not forecast_df.empty:
            record["forecast_values"] = get_values_from_df(forecast_df, combo_dict, "amount", dimensions_list)

        data.append(record)

    # ✅ Dump the whole response once, not just `data`
    response = {"success": True, "data": data}
    response = json.dumps(response, indent=4)
    return response



def fetch_hierarchies(db, project_id, planning_scenario_id, dimensions_list):
    """
    Fetch hierarchies for given project_id + planning_scenario_id
    and return them in the order defined by dimensions_list.
    """
    collection = db["hierarchies"]

    hierarchies = {}

    for dim in dimensions_list:
        # Query to find matching file_name inside planning_scenarios
        result = collection.find_one(
            {
                "project_id": project_id,
                f"planning_scenarios.{planning_scenario_id}.file_name": dim
            },
            {
                f"planning_scenarios.{planning_scenario_id}.$": 1
            }
        )

        if result and planning_scenario_id in result["planning_scenarios"]:
            for item in result["planning_scenarios"][planning_scenario_id]:
                if item["file_name"] == dim:
                    hierarchies[dim] = item.get("hierarchy_json", {})
                    break
        else:
            hierarchies[dim] = {}

    return hierarchies


def get_recommended_dimensions(db, project_id, planning_scenario_id, filename=None):
    rec_collection = db["recommendeddimensions"]

    query = {
        "project_id": project_id,
        "planning_scenario_id": planning_scenario_id
    }
    if filename:
        query["filename"] = filename

    doc = rec_collection.find_one(query)

    if not doc:
        return {"dimensions": []}
    dimensions = doc.get("rows", [])

    return dimensions

def get_data_from_s3(project_id: str,planning_scenario_id: str,file_name: str,data_type: str,budget_type: str = None,term: str = None,s3_client=None,bucket_name: str = "dev-ai-analytics-private") -> pd.DataFrame:
    """
    Fetch data (transaction, budget, or forecasting) from S3 parquet file.

    Args:
        project_id (str): Project identifier
        planning_scenario_id (str): Planning scenario identifier
        file_name (str): Base name of parquet file
        data_type (str): Type of data to fetch ("transaction", "budget", "forecasting")
        budget_type (str, optional): Required if data_type="budget"
        term (str, optional): Required if data_type="budget"
        s3_client: Optional boto3 S3 client
        bucket_name (str): S3 bucket name (default: dev-ai-analytics-private)

    Returns:
        pd.DataFrame: Requested data or empty DataFrame on failure
    """
    try:
        # --- Step 1: Build S3 key based on data_type ---
        if data_type == "actual":
            s3_key = f"fpa/template_data/{project_id}/{planning_scenario_id}/{file_name}_template.parquet"

        elif data_type == "budget":
            if not budget_type or not term:
                raise ValueError("budget_type and term are required when data_type='budget'")
            s3_key = f"fpa/budget_template/{project_id}/{planning_scenario_id}/Budget_data_{budget_type}_{term}_{file_name}.parquet"

        elif data_type == "forecast":
            s3_key = f"fpa/forecast_template/{project_id}/{planning_scenario_id}/{file_name}_template.parquet"

        else:
            raise ValueError("data_type must be one of ['actual', 'budget', 'forecast']")

        if s3_client is None:
            s3_client = boto3.client("s3")

        # --- Step 2: Read parquet from S3 ---
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_parquet(BytesIO(response["Body"].read()))

        return df

    except Exception as e:
        print(f"Error fetching {data_type} data: {e}")
        return pd.DataFrame()


def hierarchy_to_df_col(db, project_id, planning_scenario_id):
    """Fetch mapping of hierarchy names to dataframe columns from mapping_results"""
    collection = db["mapping_results"]
    result = collection.find_one(
        {"project_id": project_id, f"planning_scenarios.{planning_scenario_id}": {"$exists": True}},
        {f"planning_scenarios.{planning_scenario_id}": 1}
    )

    if not result:
        raise ValueError("No mapping found for given project_id and planning_scenario_id")

    ps_obj = result["planning_scenarios"][planning_scenario_id]

    # The hierarchy mapping is under "result"
    mapping = ps_obj.get("result", {})

    print("DEBUG - hierarchy_to_df_col mapping:", mapping)
    return mapping



def get_hierarchy_keys_for_dimensions(dimensions, mapping):
    """
    Given a list of dataframe columns (dimensions) and a mapping dict,
    return the list of keys whose values match the dimensions.
    """
    keys = []
    for dim in dimensions:
        for key, value in mapping.items():
            if value == dim:
                keys.append(key)
    return keys

def remap_hierarchy_keys(hierarchies, mapping):
    """
    Rename hierarchy keys using mapping dict so that
    Mongo keys (like 'companycode') become DataFrame keys (like 'company_code_id').
    
    Args:
        hierarchies (dict): Original hierarchies dict from Mongo
        mapping (dict): Mapping from Mongo keys -> DataFrame column names

    Returns:
        dict: Hierarchies with remapped keys
    """
    new_hierarchies = {}
    for mongo_key, hierarchy_data in hierarchies.items():
        # Find corresponding DataFrame column name
        df_col = mapping.get(mongo_key)
        if df_col:
            new_hierarchies[df_col] = hierarchy_data
        else:
            # Keep original if no mapping found
            new_hierarchies[mongo_key] = hierarchy_data
    return new_hierarchies


def run_fpa_pipeline(mongo_uri: str,db_name: str,project_id: str,planning_scenario_id: str,filename: str,data_types: list,
    budget_type: str = None,
    term: str = None,
    dimensions_clicked: str = None,
    parent_id: str = None,
    parent_combo: dict = None,
    bucket_name: str = "dev-ai-analytics-private"):
    """
    Run the full FPA pipeline:
      1. Connect to MongoDB and fetch dimensions + hierarchies
      2. Fetch data from S3 based on provided data_types
      3. Build exploded JSON
      4. If drilldown params provided → run drilldown_exploded_json

    Returns:
        dict with keys:
            "dimensions", "hierarchies", "exploded_json", "drilldown_json" (if applicable)
    """

    # --- Step 1: Connect Mongo ---
    client = MongoClient(mongo_uri)
    db = client[db_name]


    # --- Step 2: Get recommended dimensions + hierarchies ---
    dimensions = get_recommended_dimensions(db, project_id, planning_scenario_id, filename)
    # print("Dimensions: ", dimensions)
    mapping = hierarchy_to_df_col(db, project_id, planning_scenario_id)
    # print("Mapping: ", mapping)
    result = get_hierarchy_keys_for_dimensions(dimensions, mapping)
    # print("hierarchy map: ", result)
    raw_hierarchies = fetch_hierarchies(db, project_id, planning_scenario_id, result)
    # print("raw Hierarchies: ", raw_hierarchies)
    # 4. Remap to DataFrame column names
    hierarchies = remap_hierarchy_keys(raw_hierarchies, mapping)
    # print("Remapped hierarchies:", hierarchies)
   

    # --- Step 3: Fetch data from S3 ---
    df_actual, df_budget, df_forecast = None, None, None

    if "actual" in data_types or "transaction" in data_types:
        df_actual = get_data_from_s3(project_id, planning_scenario_id, filename, data_type="actual", bucket_name=bucket_name)

    if "budget" in data_types:
        df_budget = get_data_from_s3(
            project_id, planning_scenario_id, filename,
            data_type="budget", budget_type=budget_type, term=term, bucket_name=bucket_name
        )

    if "forecast" in data_types:
        df_forecast = get_data_from_s3(
            project_id, planning_scenario_id, filename,
            data_type="forecast", bucket_name=bucket_name
        )

    # --- Step 4 & 5: Build JSON depending on data_types ---
    kwargs = {}
    if "actual" in data_types or "transaction" in data_types:
        kwargs["actual_df"] = df_actual
    if "budget" in data_types:
        kwargs["budget_df"] = df_budget
    if "forecast" in data_types:
        kwargs["forecast_df"] = df_forecast

    exploded_json = build_exploded_json(hierarchies, dimensions, **kwargs)

    drilldown_json = None
    if dimensions_clicked and parent_id and parent_combo:
        drilldown_json = drilldown_exploded_json(
            hierarchies, dimensions, dimensions_clicked, parent_id, parent_combo, **kwargs
        )

    if dimensions_clicked and parent_id and parent_combo:
        return drilldown_json
        
    else:
        return  exploded_json

