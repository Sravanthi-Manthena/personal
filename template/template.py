import pandas as pd
from pymongo import MongoClient
import json
import logging
import boto3
from io import BytesIO
import random


# ---------- Setup Logger ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# ---------------- Rollup Function ----------------
def build_rollup_fact(transaction_data, dimensions, date_col, measure_col):
    logger.info("▶ Building rollup fact table")

    # Validate date and measure column exist
    if date_col not in transaction_data.columns:
        raise ValueError(f"❌ Date column '{date_col}' not found in transaction data")
    if measure_col not in transaction_data.columns:
        raise ValueError(f"❌ Measure column '{measure_col}' not found in transaction data")

    # Ensure posting_date is datetime
    transaction_data[date_col] = pd.to_datetime(transaction_data[date_col], errors="coerce")
    transaction_data["Year"] = transaction_data[date_col].dt.year
    transaction_data["Quarter"] = transaction_data[date_col].dt.to_period("Q").astype(str)
    transaction_data["Month"] = transaction_data[date_col].dt.strftime("%b-%Y")

    # Build parent-child maps for all dimensions
    parent_maps = {}
    for dim in dimensions:
        if not dim["id_col"]:
            raise ValueError(f"❌ Dimension {dim['dim_name']} is missing id_col")

        if dim["id_col"] not in dim["dim_df"].columns:
            raise ValueError(f"❌ Column {dim['id_col']} not found in {dim['dim_name']}")

        # If parent_col missing, use id_col as self-parent (flat hierarchy)
        parent_col = dim["parent_col"] or dim["id_col"]

        if parent_col not in dim["dim_df"].columns:
            # If parent_col is supposed to be id_col, it’s safe. Otherwise, error out.
            if parent_col != dim["id_col"]:
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
                expanded_rows.append({
                    **{dim["trans_id_col"]: v for dim, v in zip(dimensions, current_combination)},
                    "Year": row["Year"],
                    "Quarter": row["Quarter"],
                    "Month": row["Month"],
                    measure_col: row[measure_col]
                })

                return
            dim_name, ancestors = dim_ancestors_list[idx]
            for ancestor in ancestors:
                cartesian_product(idx + 1, current_combination + [ancestor])

        cartesian_product(0, [])

    expanded_df = pd.DataFrame(expanded_rows)
    logger.info("✅ Expanded transactions into %d rows", len(expanded_df))

    expanded_df[measure_col] = pd.to_numeric(expanded_df[measure_col], errors="coerce").fillna(0)

    group_cols = [dim["trans_id_col"] for dim in dimensions] + ["Year", "Quarter", "Month"]
    fact_table = expanded_df.groupby(group_cols, as_index=False)[measure_col].sum()

    logger.info("✅ Rollup fact table created with %d rows", len(fact_table))
    return fact_table



def fetch_transaction_data_from_s3(project_id: str, planning_scenario_id: str, new_filename_base: str,mapping_collection,user_id: str, s3_client=None, bucket_name="dev-ai-analytics-private") -> pd.DataFrame:
    """
    Fetch transaction data from S3 parquet file and keep only required columns.
    
    Args:
        project_id (str): Project identifier
        planning_scenario_id (str): Planning scenario identifier
        new_filename_base (str): Base name of parquet file
        mapping_collection: Mongo collection with mapping info
        user_id (str): Current user ID
        s3_client: Optional boto3 S3 client
        bucket_name (str): S3 bucket name (default: dev-ai-analytics-private)
    
    Returns:
        pd.DataFrame: Filtered transaction data
    """
    try:
        # --- Step 1: Build S3 key ---
        s3_key = f"fpa/transactional_data/{project_id}/{planning_scenario_id}/{new_filename_base}.parquet"
        logger.info("▶ Fetching from S3: s3://%s/%s", bucket_name, s3_key)

        if s3_client is None:
            s3_client = boto3.client("s3")

        # --- Step 2: Read parquet from S3 ---
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        df = pd.read_parquet(BytesIO(response["Body"].read()))
        logger.info("✅ Loaded parquet with %d rows, %d columns", df.shape[0], df.shape[1])

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
            df = df[[col for col in required_cols if col in df.columns]]
            logger.info("✅ Kept required columns: %s", required_cols)
        else:
            logger.warning("⚠️ No required columns mapping found, keeping full dataset")

        return df

    except Exception as e:
        logger.error("❌ Error fetching transaction data: %s", str(e), exc_info=True)
        return pd.DataFrame()




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


def insert_template_to_s3(
    df,
    user_id,
    project_id,
    planning_scenario_id,
    file_name,
    template_file_path=None,
    bucket_name="dev-ai-analytics-private",
    dimension_cols=None
):
    logger.info("▶ Storing template to S3 for project_id=%s, planning_scenario_id=%s", project_id, planning_scenario_id)

    if df.empty:
        logger.warning("⚠️ DataFrame is empty. Nothing to store.")
        return None

    # If a template parquet is provided, enforce schema
    if template_file_path:
        try:
            template_df = pd.read_parquet(template_file_path)
            df = pd.concat([template_df, df], ignore_index=True)[template_df.columns]
            logger.info("✅ Applied template from %s", template_file_path)
        except Exception as e:
            logger.error("❌ Failed to read template parquet file: %s", e)
            return None

    # ✅ Convert all dimension columns to string to avoid pyarrow dtype conflicts
    if dimension_cols:
        for col in dimension_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

    # ✅ Also force all *_id columns to string (safety net)
    for col in df.columns:
        if col.endswith("_id"):
            df[col] = df[col].astype(str)

    # Add metadata
    # df["user_id"] = str(user_id)
    # df["project_id"] = str(project_id)
    # df["planning_scenario_id"] = str(planning_scenario_id)
    # df["file_name"] = str(file_name)

    # Generate deterministic S3 path
    s3_key = f"fpa/template_data/{project_id}/{planning_scenario_id}/{file_name}_template.parquet"
    s3_path = f"s3://{bucket_name}/{s3_key}"

    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")  # ✅ safe now
        buffer.seek(0)

        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
        logger.info("✅ Stored %d rows to S3 at %s (replaced if existed)", len(df), s3_path)
        return s3_path

    except Exception as e:
        logger.error("❌ Failed to store DataFrame to S3: %s", e, exc_info=True)
        return None



# ---------------- Lambda Handler ----------------
def lambda_handler(event, context):
    try:
        logger.info("🚀 Lambda triggered with event: %s", event)

        mongo_uri = event.get("mongo_uri")
        db_name = event.get("db_name", "devfpadb")
        user_id = int(event["user_id"])
        project_id = event["project_id"]
        planning_scenario_id = event["planning_scenario_id"]
        transaction_filename = event["transaction_filename"]
        

        client = MongoClient(mongo_uri)
        db = client[db_name]
        mapping_collection = db["mapping_results"]

        # Fetch transaction data
        trans_collection = db["transactionaldata"]
        transaction_df = fetch_transaction_data_from_s3(project_id, planning_scenario_id, transaction_filename ,mapping_collection,user_id)
        logger.info("transaction_df : %s", transaction_df.head(2))
        

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
        date_col = recommended_dims["measures"].get("date_dimension")
        measure_col = recommended_dims["measures"].get("measure")


        if not date_col or not measure_col:
            raise ValueError(" Missing date or measure mapping in DB")

        # Build rollup fact table
        rollup_fact_df = build_rollup_fact(transaction_data=transaction_df, dimensions=dimensions, date_col=date_col, measure_col=measure_col)
        logger.info("rollup_fact :  %s", rollup_fact_df.head(2))
        rollup_fact_df.to_csv("rollup_fact.csv")

             
        s3_path_returned = insert_template_to_s3(rollup_fact_df, user_id, project_id, planning_scenario_id, transaction_filename)
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
