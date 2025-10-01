import pandas as pd
from pymongo import MongoClient
import json
import logging
import boto3
from io import BytesIO
import random
import pickle


# ---------- Setup Logger ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# ---------------- Rollup Function ----------------
# def build_rollup_fact(transaction_data, dimensions, date_col, measure_col):
#     logger.info("▶ Building rollup fact table")
    
#     # Validate date and measure column exist
#     if date_col not in transaction_data.columns:
#         raise ValueError(f"❌ Date column '{date_col}' not found in transaction data")
#     if measure_col not in transaction_data.columns:
#         raise ValueError(f"❌ Measure column '{measure_col}' not found in transaction data")

#     # Ensure posting_date is datetime
#     transaction_data[date_col] = pd.to_datetime(transaction_data[date_col], errors="coerce")
#     transaction_data["Year"] = transaction_data[date_col].dt.year
#     transaction_data["Quarter"] = transaction_data[date_col].dt.to_period("Q").astype(str)
#     transaction_data["Month"] = transaction_data[date_col].dt.strftime("%b-%Y")

#     # Build parent-child maps for all dimensions
#     parent_maps = {}
#     for dim in dimensions:
#         if not dim["id_col"]:
#             raise ValueError(f"❌ Dimension {dim['dim_name']} is missing id_col")

#         if dim["id_col"] not in dim["dim_df"].columns:
#             raise ValueError(f"❌ Column {dim['id_col']} not found in {dim['dim_name']}")

#         # If parent_col missing, use id_col as self-parent (flat hierarchy)
#         parent_col = dim["parent_col"] or dim["id_col"]

#         if parent_col not in dim["dim_df"].columns:
#             # If parent_col is supposed to be id_col, it’s safe. Otherwise, error out.
#             if parent_col != dim["id_col"]:
#                 raise ValueError(f"❌ Parent column {parent_col} not found in {dim['dim_name']}")

#         parent_maps[dim["dim_name"]] = dict(
#             zip(dim["dim_df"][dim["id_col"]], dim["dim_df"][parent_col])
#         )
#     logger.info("✅ Parent-child maps built for %d dimensions", len(dimensions))
#     logger.info("parent_child maps %d", parent_maps)

#     def get_ancestors(node, parent_map):
#         ancestors = [node]
#         while node in parent_map and parent_map[node] != node:
#             node = parent_map[node]
#             ancestors.append(node)
#         return ancestors

#     expanded_rows = []

#     for _, row in transaction_data.iterrows():
#         dim_ancestors_list = []
#         for dim in dimensions:
#             if dim["trans_id_col"] not in row:
#                 raise ValueError(f"❌ Transaction column {dim['trans_id_col']} not found in data")
#             node = row[dim["trans_id_col"]]
#             ancestors = get_ancestors(node, parent_maps[dim["dim_name"]])
#             dim_ancestors_list.append((dim["dim_name"], ancestors))

#         def cartesian_product(idx, current_combination):
#             if idx == len(dim_ancestors_list):
#                 expanded_rows.append({
#                     **{dim["trans_id_col"]: v for dim, v in zip(dimensions, current_combination)},
#                     "Year": row["Year"],
#                     "Quarter": row["Quarter"],
#                     "Month": row["Month"],
#                     measure_col: row[measure_col]
#                 })

#                 return
#             dim_name, ancestors = dim_ancestors_list[idx]
#             for ancestor in ancestors:
#                 cartesian_product(idx + 1, current_combination + [ancestor])

#         cartesian_product(0, [])

#     expanded_df = pd.DataFrame(expanded_rows)
#     logger.info("✅ Expanded transactions into %d rows", len(expanded_df))

#     group_cols = [dim["trans_id_col"] for dim in dimensions] + ["Year", "Quarter", "Month"]
#     fact_table = expanded_df.groupby(group_cols, as_index=False)[measure_col].sum()

#     logger.info("✅ Rollup fact table created with %d rows", len(fact_table))
#     return fact_table


def build_rollup_fact(transaction_data, dimensions, date_col, measure_col):
    logger.info("▶ Building rollup fact table")

    # --- Validate ---
    if date_col not in transaction_data.columns:
        raise ValueError(f"❌ Date column '{date_col}' not found in transaction data")

    # if single column, convert to list for uniform handling
    if isinstance(measure_col, str):
        measure_cols = [measure_col]
    else:
        measure_cols = measure_col
    # logger.info("measure_cols: %s", measure_cols)

    missing_measures = [m for m in measure_cols if m not in transaction_data.columns]
    if missing_measures:
        raise ValueError(f"❌ Measure columns {missing_measures} not found in transaction data")
        # --- Convert variance_pct to numeric if it exists ---
    
    if "variance_pct" in transaction_data.columns:
        # Remove % symbol and convert to float
        transaction_data["variance_pct"] = (
            transaction_data["variance_pct"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)   # in case numbers like "98,5%"
            .astype(float)
        )

    # --- Date features ---
    transaction_data[date_col] = pd.to_datetime(transaction_data[date_col], errors="coerce")
    transaction_data["Year"] = transaction_data[date_col].dt.year
    transaction_data["Quarter"] = transaction_data[date_col].dt.to_period("Q").astype(str)
    transaction_data["Month"] = transaction_data[date_col].dt.strftime("%b-%Y")

    # --- Build parent-child maps ---
    parent_maps = {}
    for dim in dimensions:
        if not dim["id_col"]:
            raise ValueError(f"❌ Dimension {dim['dim_name']} is missing id_col")

        if dim["id_col"] not in dim["dim_df"].columns:
            raise ValueError(f"❌ Column {dim['id_col']} not found in {dim['dim_name']}")

        parent_col = dim["parent_col"] or dim["id_col"]
        if parent_col not in dim["dim_df"].columns and parent_col != dim["id_col"]:
            raise ValueError(f"❌ Parent column {parent_col} not found in {dim['dim_name']}")

        parent_maps[dim["dim_name"]] = dict(
            zip(dim["dim_df"][dim["id_col"]], dim["dim_df"][parent_col])
        )
    logger.info("✅ Parent-child maps built for %d dimensions", len(dimensions))

    def get_ancestors(node, parent_map):
        ancestors = [node]
        while node in parent_map and parent_map[node] != node:
            node = parent_map[node]
            ancestors.append(node)
        return ancestors

    expanded_rows = []

    for _, row in transaction_data.iterrows():
        dim_ancestors_list = []
        for dim in dimensions:
            if dim["trans_id_col"] not in row:
                raise ValueError(f"❌ Transaction column {dim['trans_id_col']} not found in data")
            node = row[dim["trans_id_col"]]
            ancestors = get_ancestors(node, parent_maps[dim["dim_name"]])
            dim_ancestors_list.append((dim["dim_name"], ancestors))

        def cartesian_product(idx, current_combination):
            if idx == len(dim_ancestors_list):
                # Base row
                base_row = {dim["trans_id_col"]: v for dim, v in zip(dimensions, current_combination)}
                base_row.update({
                    "Year": row["Year"],
                    "Quarter": row["Quarter"],
                    "Month": row["Month"],
                })
                # Add all measures
                for m in measure_cols:
                    base_row[m] = row[m]
                expanded_rows.append(base_row)
                return

            _, ancestors = dim_ancestors_list[idx]
            for ancestor in ancestors:
                cartesian_product(idx + 1, current_combination + [ancestor])

        cartesian_product(0, [])

    expanded_df = pd.DataFrame(expanded_rows)
    logger.info("✅ Expanded transactions into %d rows", len(expanded_df))

    # --- Group by dimensions + date fields ---
        # --- Group by dimensions + date fields ---
    group_cols = [dim["trans_id_col"] for dim in dimensions] + ["Year", "Quarter", "Month"]

    # define aggregation rules
    agg_dict = {}
    # logger.info("measure_cols: %s", measure_cols)
    for m in measure_cols:
        if m.endswith("_pct"):     # any percentage column
            agg_dict[m] = "mean"
        else:
            agg_dict[m] = "sum"

    fact_table = expanded_df.groupby(group_cols, as_index=False).agg(agg_dict)


    logger.info("✅ Rollup fact table created with %d rows and measures %s", len(fact_table), measure_cols)
    return fact_table



def fetch_data_from_s3(data_type: str, project_id: str, planning_scenario_id: str,budget_type:str, term:str,
                       new_filename_base: str, mapping_collection, user_id: str,
                       s3_client=None, bucket_name="dev-ai-analytics-private") -> pd.DataFrame:
    """
    Fetch transaction/budget data from S3 parquet or csv file and keep only required columns.
    """
    try:
        if s3_client is None:
            s3_client = boto3.client("s3")

        # --- Step 1: Build S3 key based on data_type ---
        if data_type == "forecast":
            s3_key = f"fpa/forecast_data/{project_id}/{planning_scenario_id}/{new_filename_base}_forecast.parquet"
        elif data_type == "budget":
            # budget files are in CSV (as per example provided)
            s3_key = f"fpa/Budget_data/{project_id}/{planning_scenario_id}/Budget_data_{budget_type}_{term}_{new_filename_base}.parquet"
        elif data_type == "variance":
            s3_key = f"fpa/variance_data/{project_id}/{planning_scenario_id}/{new_filename_base}_variance.parquet"
        else:
            raise ValueError(f"❌ Unsupported data_type: {data_type}")

        logger.info("▶ Fetching from S3: s3://%s/%s", bucket_name, s3_key)

        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        file_data = response["Body"].read()

        # --- Step 2: Load DataFrame depending on format ---
               
        if s3_key.endswith(".parquet"):
            df = pd.read_parquet(BytesIO(file_data))
        elif s3_key.endswith(".csv"):
            # try utf-8-sig first, fallback to latin1
            try:
                df = pd.read_csv(BytesIO(file_data), encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(BytesIO(file_data), encoding="latin1")
        else:
            raise ValueError(f"❌ Unsupported file format for key: {s3_key}")

        logger.info("✅ Loaded data with %d rows, %d columns", df.shape[0], df.shape[1])

        # --- Step 3: Get required columns from Mongo ---
        mapping_doc = mapping_collection.find_one(
            {"user_id": user_id, "project_id": project_id},
            {"_id": 0, f"planning_scenarios.{planning_scenario_id}.result": 1}
        )

        required_cols = []
        if mapping_doc:
            result_dict = (
                mapping_doc.get("planning_scenarios", {})
                           .get(planning_scenario_id, {})
                           .get("result", {})
            )
            if result_dict:
                required_cols = [col for col in result_dict.values() if col]

         # --- Step 4: Filter DataFrame ---
        if required_cols:
            # always include date column depending on data_type
            if data_type == "forecast":
                date_col = "month"
            elif data_type == "variance":
                date_col = "month"
            else:
                date_col = result_dict.get("date_dimension")

            # always include budget_amount if budget type
            extra_cols = []
            if data_type == "budget":
                extra_cols.append("budget_amount")

            # always include specific variance fields if variance type
            if data_type == "variance":
                extra_cols.extend([
                    "period",
                    "actual_amount",
                    "forecast_amount",
                    "budget_amount",
                    "variance",
                    "variance_pct",
                    "actual_year",
                    "budget_year"
                ])

            cols_to_keep = list(set(required_cols + [date_col] + extra_cols))
            available_cols = [col for col in cols_to_keep if col in df.columns]

            missing_cols = [col for col in cols_to_keep if col not in df.columns]
            if missing_cols:
                logger.warning("⚠️ Missing columns in dataset: %s", missing_cols)

            if available_cols:
                df = df[available_cols]
                logger.info("✅ Kept columns: %s", available_cols)
            else:
                logger.warning("⚠️ No matching columns found, keeping full dataset")
        else:
            logger.warning("⚠️ No required columns mapping found, keeping full dataset")

        return df

    except Exception as e:
        logger.error("❌ Error fetching transaction/budget data: %s", str(e), exc_info=True)
        return pd.DataFrame()    

def get_term_of_plan(collection, project_id, planning_scenario_id, file_name):
    result = collection.find_one(
        {
            "project_id": project_id,
            "planning_scenario_id": planning_scenario_id,
            "filename": file_name
        },
        {
            "term_of_plan": 1,  # project only the field you want
            "_id": 0
        }
    )
    return result.get("term_of_plan") if result else None
    

def prepare_dimension_list(db, user_id, project_id, planning_scenario_id):
    logger.info("▶ Preparing dimension list")
    files_collection = db["fileuploaddata"]

    file_docs = list(files_collection.find({
        "user_id": user_id,
        "project_id": project_id,
        "planning_scenario_id": planning_scenario_id
    }))

    if not file_docs:
        raise FileNotFoundError("❌ Dimension files not found")

    # Normalize helper: lowercase + remove underscores
    def normalize(name: str) -> str:
        return name.replace("_", "").lower() if name else ""

    # Group uploaded dimension files
    grouped_files = {}
    for doc in file_docs:
        filename = doc.get("filename")
        if filename in ["date_dimension", "version_dimension"]:
            continue

        data_obj = doc.get("data")
        if isinstance(data_obj, dict):
            row_df = pd.DataFrame([data_obj])
        elif isinstance(data_obj, list):
            row_df = pd.DataFrame(data_obj)
        else:
            row_df = pd.DataFrame()

        grouped_files[filename] = pd.concat(
            [grouped_files.get(filename, pd.DataFrame()), row_df], ignore_index=True
        )

    if not grouped_files:
        raise FileNotFoundError("❌ No valid dimension files found after filtering")

    # Fetch mapping
    mapping_collection = db["mapping_results"]
    doc = mapping_collection.find_one({
        "user_id": user_id,
        "project_id": project_id,
        f"planning_scenarios.{planning_scenario_id}": {"$exists": True}
    })
    mapping = {}
    if doc:
        scenario_mapping = doc["planning_scenarios"].get(planning_scenario_id, {})
        mapping = scenario_mapping.get("result", {})

    if not mapping:
        raise ValueError("❌ Mapping is required but missing")

    col_names_collection = db["table_metadata"]
    dimensions = []

    for fname, df in grouped_files.items():
        transaction_col_name = None
        for k, v in mapping.items():
            if not v:  # skip nulls
                continue

            # normalize both sides
            key_base = k.replace("_dimension", "")
            if normalize(fname) == normalize(key_base):
                transaction_col_name = v
                break

        if not transaction_col_name:
            logger.warning("⚠️ Skipping dimension %s because no transaction mapping found", fname)
            continue

        # Fetch metadata for this dimension
        col_doc = col_names_collection.find_one({
            "user_id": user_id,
            "project_id": project_id,
            "scenarios.planning_scenario_id": planning_scenario_id
        }, {"_id": 0, "scenarios": 1})

        id_col = None
        hierarchy_col = None
        if col_doc:
            for scenario in col_doc.get("scenarios", []):
                if scenario.get("planning_scenario_id") == planning_scenario_id:
                    for table in scenario.get("tables", []):
                        if table.get("table_name") == fname:
                            id_col = (table.get("unique_id_columns") or [None])[0]
                            hierarchy_col = (table.get("hierarchy_columns") or [None])[0]
                            break

        if not id_col:
            logger.error("❌ Skipping dimension %s because id_col is missing", fname)
            continue

        # hierarchy_col can be None (flat hierarchy)
        dimensions.append({
            "dim_name": fname,
            "dim_df": df,
            "id_col": id_col,
            "parent_col": hierarchy_col,  # can be None
            "trans_id_col": transaction_col_name
        })

    logger.info("✅ Prepared %d dimensions", len(dimensions))
    return dimensions


def get_recommended_dimensions(db, project_id, planning_scenario_id, filename=None):
    logger.info("▶ Fetching recommended dimensions & measures for scenario=%s", planning_scenario_id)
    rec_collection = db["recommendeddimensions"]

    query = {
        "project_id": project_id,
        "planning_scenario_id": planning_scenario_id
    }
    if filename:
        query["filename"] = filename

    doc = rec_collection.find_one(query)

    if not doc:
        logger.warning("⚠️ No recommended data found")
        return {"dimensions": [], "measures": []}

    return {
        "dimensions": doc.get("rows", []),
        "measures": doc.get("columns", {})
    }


def insert_template_to_s3(df, data_type: str,budget_type:str, term:str, project_id, planning_scenario_id, file_name,
                          bucket_name="dev-ai-analytics-private"):
    logger.info("▶ Storing template to S3 for project_id=%s, planning_scenario_id=%s", project_id, planning_scenario_id)

    if df.empty:
        logger.warning("⚠️ DataFrame is empty. Nothing to store.")
        return None

    # ✅ Convert *_id columns to string
    for col in df.columns:
        if col.endswith("_id"):
            df[col] = df[col].astype(str)

    # --- Path depends on data_type ---
    if data_type == "forecast":
        s3_key = f"fpa/forecast_template/{project_id}/{planning_scenario_id}/{file_name}_template.parquet"
    elif data_type == "budget":
        s3_key = f"fpa/budget_template/{project_id}/{planning_scenario_id}/Budget_data_{budget_type}_{term}_{file_name}.parquet"
    elif data_type == "variance":
        s3_key = f"fpa/variance_template/{project_id}/{planning_scenario_id}/{file_name}_variance_template.parquet"
    else:
        raise ValueError(f"❌ Unsupported data_type: {data_type}")
    
    s3_path = f"s3://{bucket_name}/{s3_key}"

    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)

        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
        logger.info("✅ Stored %d rows to S3 at %s (replaced if existed)", len(df), s3_path)
        return s3_path

    except Exception as e:
        logger.error("❌ Failed to store DataFrame to S3: %s", e, exc_info=True)
        return None




def store_pickle_to_s3(model, user_id, project_id, planning_scenario_id, file_name, bucket_name="dev-ai-analytics-private"):
    """
    Store the best model pickle file to S3.
    """
    logger.info("▶ Storing pickle file to S3 for project_id=%s, planning_scenario_id=%s", project_id, planning_scenario_id)

    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        pickle.dump(model, buffer)
        buffer.seek(0)

        s3_key = f"fpa/forecast_model/{project_id}/{planning_scenario_id}/{file_name}_best_model.pkl"
        s3_path = f"s3://{bucket_name}/{s3_key}"

        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
        logger.info("✅ Stored pickle to S3 at %s (replaced if existed)", s3_path)
        return s3_path

    except Exception as e:
        logger.error("❌ Failed to store pickle to S3: %s", e, exc_info=True)
        return None


# ---------------- Lambda Handler ----------------
def lambda_handler(event, context):
    try:
        logger.info("🚀 Lambda triggered with event: %s", event)
        if "body" in event:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)   # convert JSON string to dict
        else:
            body = event   # Step Functions passes direct JSON

        # --- Step 2: Extract parameters ---
        mongo_uri = body.get("mongo_uri")
        db_name = body.get("db_name", "devfpadb")
        user_id = int(body.get("user_id"))
        project_id = body.get("project_id")
        planning_scenario_id = body.get("planning_scenario_id")
        data_type = body.get("data_type")
        budget_type = body.get("budget_type")
        transaction_filename = body.get("transaction_filename")
        analysis_type = body.get( "analysis_type")
        

        client = MongoClient(mongo_uri)
        db = client[db_name]
        mapping_collection = db["mapping_results"]
        transaction_collection =db["transactionaldata"]
        term = get_term_of_plan(transaction_collection, project_id, planning_scenario_id, transaction_filename)

        # Fetch transaction data
        transaction_df = fetch_data_from_s3(data_type = data_type, project_id = project_id, planning_scenario_id = planning_scenario_id,budget_type = budget_type, term = term,new_filename_base = transaction_filename, mapping_collection = mapping_collection, user_id = user_id) 
        logger.info("transaction_df : %s", transaction_df.head(20))
        # transaction_df.to_csv("forecasted.csv")


        if transaction_df is None or transaction_df.empty:
            raise ValueError(" No transaction data found")

        # Fetch all dimensions
        all_dimensions = prepare_dimension_list(db, user_id, project_id, planning_scenario_id)
        if not all_dimensions:
            raise ValueError(" No dimension definitions found")


        # Filter dimensions based on recommended list
        recommended_dims = get_recommended_dimensions(db, project_id, planning_scenario_id)
        print("Recommended_dims: ", recommended_dims)
        dimensions = [dim for dim in all_dimensions if dim["trans_id_col"] in recommended_dims.get("dimensions", [])]
        print("Dimensions: ", dimensions)
        if not dimensions:
            raise ValueError(" No dimensions selected after filtering by recommended list")

        # Fetch mappings
        print("recommendation_dims:", recommended_dims)
        logger.info("recommendation_dims: %s", recommended_dims)
        if data_type == "forecast":
            date_col = "month"
            measure_col = recommended_dims["measures"].get("measure")
        elif data_type == "variance":
            date_col = "period"
            # always include actual + variance fields
            measure_col = ["actual_amount", "variance", "variance_pct"]
            # conditionally add forecast or budget
            if analysis_type == "actual_vs_forecast":
                measure_col.append("forecast_amount")
            elif analysis_type == "actual_vs_budget":
                measure_col.append("budget_amount")
        else:  # budget
            date_col = recommended_dims["measures"].get("date_dimension")
            measure_col = "budget_amount"

        logger.info("Using date_col='%s', measure_col='%s'", date_col, measure_col)
        if not date_col or not measure_col:
            raise ValueError(" Missing date or measure mapping in DB")
        logger.info("transaction_df columns: %s", transaction_df.columns.tolist())

        # Build rollup fact table
        rollup_fact_df = build_rollup_fact(transaction_data=transaction_df, dimensions=dimensions, date_col=date_col, measure_col=measure_col)
        logger.info("rollup_fact :  %s", rollup_fact_df.head(20))
        rollup_fact_df.to_csv("rollup_table.csv")

             
        s3_path_returned = insert_template_to_s3(df = rollup_fact_df, budget_type = budget_type, term = term, data_type = data_type, project_id = project_id, planning_scenario_id= planning_scenario_id, file_name = transaction_filename)
        logger.info(" Rollup fact table built successfully")
        

        return {
            "statusCode": 200,
            "body": {
                "message": " Rollup fact table built successfully",
                "s3_path" : s3_path_returned
            }
        }

    except Exception as e:
        logger.error(" Lambda failed: %s", str(e), exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(e)}
        }


event = {
"mongo_uri": "mongodb://devfpauser:Ok74c3YE7GLN@13.202.247.111:27017/devfpadb",
"db_name": "devfpadb",
"user_id": 18,
"project_id": "01K5E06REG5HA0Y7DK74Q15MF2",
"planning_scenario_id": "01K1TAZPVGNR5KK6BBRHZFYWQ7",
"transaction_filename": "trans",
"data_type" : "budget",
"budget_type" : "AI",
"analysis_type" : "actual_vs_forecast"
}

response = lambda_handler(event, None)
print(response)

