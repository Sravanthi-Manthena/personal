import os
import json
import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient, ReturnDocument
from openai import OpenAI
from datetime import datetime
import boto3
from io import BytesIO

# ---------- Setup Logger ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def get_hierarchy_json(collection, project_id, planning_scenario_id, file_name):
    
    collection = collection  # replace with your collection name

    # Query to match project and scenario
    result = collection.find_one(
        {
            "project_id": project_id,
            f"planning_scenarios.{planning_scenario_id}.file_name": file_name
        },
        {
            f"planning_scenarios.{planning_scenario_id}.$": 1  # project only matching array element
        }
    )

    if result and planning_scenario_id in result["planning_scenarios"]:
        for item in result["planning_scenarios"][planning_scenario_id]:
            if item["file_name"] == file_name:
                return item.get("hierarchy_json", {})
    return None



def get_leaf_nodes(tree, verbose=False):
    leaf_nodes = set()

    def dfs(node, subtree):
        if not subtree:  # no children → leaf
            leaf_nodes.add(node)
        else:
            for child, child_tree in subtree.items():
                dfs(child, child_tree)

    for root, subtree in tree.items():
        dfs(root, subtree)

    logger.info("Extracted %d leaf nodes", len(leaf_nodes))
    if verbose:
        logger.info("Leaf nodes: %s", leaf_nodes)
    return leaf_nodes

# 1️⃣ Function to fetch transaction data
def get_transaction_data(project_id: str, planning_scenario_id: str, new_filename_base: str, s3_client=None, bucket_name="dev-ai-analytics-private") -> pd.DataFrame:
    """
    Fetch transaction data from S3 parquet file.

    Args:
        project_id (str): Project identifier
        planning_scenario_id (str): Planning scenario identifier
        new_filename_base (str): Base name of parquet file
        s3_client: Optional boto3 S3 client
        bucket_name (str): S3 bucket name (default: dev-ai-analytics-private)

    Returns:
        pd.DataFrame: Transaction data
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

        return df

    except Exception as e:
        logger.error("❌ Error fetching transaction data: %s", str(e), exc_info=True)
        return pd.DataFrame()


# ---------- Validation 1 ----------
def validate_transaction_not_empty(transaction_df):
    logger.info("Running first validation")

    if transaction_df is None or transaction_df.empty:
        logger.error("❌ Uploaded transaction table is empty.")
        return {
            "status": "error",
            "message": "Uploaded transaction table is empty."
        }

    # Check for null values
    if transaction_df.isnull().values.any():
        null_counts = transaction_df.isnull().sum()
        logger.warning("⚠️ Transaction table contains null values:\n%s", null_counts)
        return {
            "status": "warning",
            "message": "Transaction table contains null values.",
            "null_summary": null_counts.to_dict()
        }

    logger.info("✅ Transaction table has data and no null values.")
    return {
        "status": "success",
        "message": "Transaction table has data and no null values."
    }




# ---------- Validation 2 ----------
def validate_account_dimension(db, user_id, project_id, planning_scenario_id):
    logger.info("Running second validation (Account Dimension Check)")
    
    col_names_collection = db["table_metadata"]
     # Fetch scenario metadata
    col_doc = col_names_collection.find_one({
        "user_id": user_id,
        "project_id": project_id,
        "scenarios.planning_scenario_id": planning_scenario_id
    }, {"_id": 0, "scenarios": 1})

    if not col_doc:
        msg = "No column metadata found for this scenario."
        logger.error("❌ %s", msg)
        return {"status": "error", "message": msg}

    account_found = False
    account_table = None

    # Traverse scenarios → tables
    for scenario in col_doc.get("scenarios", []):
        if scenario.get("planning_scenario_id") == planning_scenario_id:
            for table in scenario.get("tables", []):
                if table.get("Type", "").lower() == "account":
                    account_found = True
                    account_table = table.get("table_name")
                    break

    if account_found:
        msg = f"Account Dimension exists: {account_table}"
        logger.info("✅ %s", msg)
        return {"status": "success", "message": msg}
    else:
        msg = "Account Dimension is missing (Type: Account not found)."
        logger.error("❌ %s", msg)
        return {"status": "error", "message": msg}


# ---------- Validation 3 ----------
def validate_transaction_ids(df_transactions, dimensions_list, mapping_dict, dimension_tables):
    logger.info("Running third validation (Transaction IDs vs Dimension IDs)")
    results = {}

    for dim_name in dimensions_list:
        tx_col = mapping_dict.get(dim_name)

        # If mapping not found
        if not tx_col:
            results[dim_name] = {
                "status": "error",
                "message": "No mapping found."
            }
            continue

        # If column missing in transactions
        if tx_col not in df_transactions.columns:
            results[dim_name] = {
                "status": "error",
                "message": f"Column '{tx_col}' missing in transactions."
            }
            continue

        # If dimension table not provided
        if tx_col not in dimension_tables:
            results[dim_name] = {
                "status": "error",
                "message": "Dimension table not provided."
            }
            continue

        # Validate IDs
        dim_df, id_col, _ = dimension_tables[tx_col]
        tx_ids = set(df_transactions[tx_col].dropna().unique())
        dim_ids = set(dim_df[id_col].dropna().unique())
        invalid_ids = tx_ids - dim_ids
        invalid_ids_serializable = [
            int(i) if isinstance(i, (np.integer,)) else str(i) 
            for i in invalid_ids
        ]

        if invalid_ids_serializable:
            results[dim_name] = {
                "status": "error",
                "message": f"{len(invalid_ids_serializable)} invalid ID(s) found.",
                "invalid_ids": invalid_ids_serializable
            }
        else:
            results[dim_name] = {
                "status": "success",
                "message": "All IDs are valid."
            }

    return results

# ---------- Validation 4 ----------
def validate_leaf_nodes(transaction_df, dimensions, mapping, collection, project_id, planning_scenario_id):
    logger.info("Running fourth validation (Leaf Node Check)")
    results = {}
    logger.info("Dimensions: %s", dimensions)

    for dim in dimensions:
        tx_col_name = mapping.get(dim)

        # Basic checks
        if not tx_col_name or tx_col_name not in transaction_df.columns:
            results[dim] = {
                "status": "error",
                "message": f"Column or mapping not found for `{dim}`."
            }
            continue

        # Use dimension name as filename for hierarchy fetch
        file_name = dim  

        # Fetch hierarchy JSON from Mongo
        tree = get_hierarchy_json(collection, project_id, planning_scenario_id, file_name)

        if not tree:
            results[dim] = {
                "status": "error",
                "message": f"No hierarchy found in DB for `{dim}` (file: {file_name})."
            }
            continue

        # Extract leaf nodes
        leaf_nodes = get_leaf_nodes(tree)

        # Transaction IDs
        tx_ids = set(transaction_df[tx_col_name].dropna().unique())
        non_leaf_ids = tx_ids - leaf_nodes
        non_leaf_ids_serializable = [
            int(i) if isinstance(i, (np.integer,)) else str(i)
            for i in non_leaf_ids
        ]

        # Validation result
        if non_leaf_ids_serializable:
            results[dim] = {
                "status": "error",
                "message": f"{len(non_leaf_ids_serializable)} invalid ID(s) found.",
                "invalid_ids": non_leaf_ids_serializable
            }
        else:
            results[dim] = {
                "status": "success",
                "message": "All IDs are valid leaf nodes."
            }

    return results


# ---------- Validation 5 ----------
def identify_date_col(df, groq_key):
    logger.info("Running fifth validation: Identify date column via Groq LLM")

    sample_data = df.head(3).to_dict(orient="records")
    sample_json = json.dumps(sample_data, indent=2)

    prompt = f'''
# Date Column Identifier Prompt
You are a date mapping expert. Your task is to analyze a JSON table and identify which column contains dates.

## Input:
- Table: {sample_json}

## Task:
Analyze the JSON data and return only the name of the column that contains dates.

## Output:
Return only the column name as a string. If multiple date columns exist, return the primary/most relevant one. If no date column is found, return "None".

## Examples:

**Input:**
```json
{{
  "columns": ["id", "created_date", "amount"],
  "data": [[1, "2023-01-15", 100], [2, "2023-02-20", 250]]
}}
```

**Output:**
```
{{
  "date_column": "created_date"
}}
```

**Input:**
```json
{{
  "columns": ["name", "age", "city"],
  "data": [["John", 25, "NYC"], ["Jane", 30, "LA"]]
}}
```

**Output:**
```
{{
  "date_column": null
}}
```

'''
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
    res = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": "Classify tables."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    date = json.loads(res.choices[0].message.content)
    logger.info("Groq detected date column: %s", date)
    return date

def validate_date_range(df, date_col, start_month_year, end_month_year):
    logger.info("Validating date range for column `%s`", date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Define start and end dates
    start_date = pd.to_datetime(start_month_year + "-01")
    end_date = pd.to_datetime(end_month_year + "-01") + pd.offsets.MonthEnd(0)
    
    # Find invalid dates
    invalid_dates = df.loc[~df[date_col].between(start_date, end_date), date_col]
    
    if invalid_dates.empty:
        message = f"All dates are within range [{start_date.date()} → {end_date.date()}]."
        logger.info(message)
        # print(message)
        return {"status": "success", "message": message}
    
    else:
        invalid_list = sorted(invalid_dates.dropna().unique())
        min_invalid = min(invalid_list).date()
        max_invalid = max(invalid_list).date()
        message = (
            f"Found {len(invalid_list)} invalid dates outside range "
            f"[{min_invalid} → {max_invalid}]"
        )
        logger.info(message)
        # print(message)
        return {"status": "error", "message": message}
    

def compute_overall_status(result_dict):
    """
    Recursively check all nested validation results.
    Return 'success' only if every status == 'success'.
    """
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "status":
                    if v != "success":
                        return False
                else:
                    if not walk(v):
                        return False
        elif isinstance(obj, list):
            for item in obj:
                if not walk(item):
                    return False
        return True

    return "success" if walk(result_dict) else "failed"



def store_in_mongo(output_collection, user_id, project_id, planning_scenario_id, validation_results, status):
    """
    Store validation results in MongoDB for a given user/project/planning_scenario.
    """
    try:
        update_result = output_collection.find_one_and_update(
            {"user_id": user_id, "project_id": project_id},
            {
                "$set": {
                    f"planning_scenarios.{planning_scenario_id}": {
                        "validation_status": status,
                        "validations": validation_results,   # ✅ full dict stored here
                        "time_stamp": datetime.utcnow().isoformat()
                    }
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        object_id = update_result["_id"]
        logger.info("Stored mapping result in MongoDB. ObjectId: %s", object_id)
        return object_id
    except Exception as e:
        logger.error("Error storing validation results in MongoDB: %s", str(e))
        raise



# ---------- Lambda Handler ----------
def lambda_handler(event, context):
    try:
        logger.info("Lambda triggered with event: %s", event)
        mongo_uri = event["mongo_uri"]
        groq_key = event["groq_key"]
        user_id = event["user_id"]
        project_id = event["project_id"]
        planning_scenario_id = event["planning_scenario_id"]
        transaction_filename = event["transaction_filename"]
        start_month_year = event["start_month_year"]
        end_month_year = event["end_month_year"]

        client = MongoClient(mongo_uri)
        db = client.get_database()
        output_collection = db["validation_results"]
        hierarchy_collection = db["hierarchies"]

        # 1️⃣ Transactions
        df = get_transaction_data(project_id=project_id, planning_scenario_id=planning_scenario_id, new_filename_base=transaction_filename)
        if df is None:
            return {"statusCode": 404, "body": json.dumps({"error": "No transactional data found"})}
        logger.info("Loaded %d transactions", len(df))


        # 2️⃣ Dimensions
        files_collection = db["fileuploaddata"]
        filename_list = files_collection.distinct("filename", {
            "user_id": user_id,
            "project_id": project_id,
            "planning_scenario_id": planning_scenario_id
        })
        for drop_name in ["date_dimension", "version_dimension"]:
            if drop_name in filename_list:
                filename_list.remove(drop_name)
        logger.info("Dimension files: %s", filename_list)
        if not filename_list:
            logger.error("No dimension files found in the given path")
            raise FileNotFoundError("Dimension files are required but not found")



        # 3️⃣ Mapping
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

        logger.info("Mapping found: %s", mapping)
        if not mapping:
            logger.error("Mapping not found for dimension processing")
            raise ValueError("Mapping is required but missing")


        # 4️⃣ Dimension tables
        col_names_collection = db["table_metadata"]
        df_dict = {}
        for filename in filename_list:
            transaction_col_name = None
            for k, v in mapping.items():
                if v and filename.startswith(k.replace("_dimension", "")):
                    transaction_col_name = mapping[k]
                    break
            if not transaction_col_name:
                continue
            dim_data = list(files_collection.find({
                "user_id": user_id,
                "project_id": project_id,
                "planning_scenario_id": planning_scenario_id,
                "filename": filename
            }, {"_id": 0, "data": 1}))
            dim_df = pd.DataFrame([doc["data"] for doc in dim_data])

            col_doc = col_names_collection.find_one({
                "user_id": user_id,
                "project_id": project_id,
                "scenarios.planning_scenario_id": planning_scenario_id
            }, {"_id": 0, "scenarios": 1})

            id_col, hierarchy_col = None, None
            if col_doc:
                for scenario in col_doc.get("scenarios", []):
                    if scenario.get("planning_scenario_id") == planning_scenario_id:
                        for table in scenario.get("tables", []):
                            if table.get("table_name") == filename:
                                id_col = (table.get("unique_id_columns") or [None])[0]
                                hierarchy_col = (table.get("hierarchy_columns") or [None])[0]
                                break
            df_dict[transaction_col_name] = (dim_df, id_col, hierarchy_col)


            # Run all validations
        first_result = validate_transaction_not_empty(df)

        # ⛔ If first validation fails, stop and store only that
        if first_result["status"] == "error":
            result_dict = {
                "empty_transaction_table": first_result
            }
            # Store only first validation result in Mongo
            store_in_mongo(output_collection, user_id, project_id, planning_scenario_id, result_dict)
            return {
                "statusCode": 400,
                "body": json.dumps(result_dict)
            }
        

        # ✅ Continue with remaining validations only if first passes
        second_result = validate_account_dimension(db, user_id, project_id, planning_scenario_id)
        third_result = validate_transaction_ids(df, filename_list, mapping, df_dict)
        fourth_result = validate_leaf_nodes(df, filename_list, mapping, hierarchy_collection, project_id, planning_scenario_id)
        date_info = identify_date_col(df, groq_key)
        date_col = date_info.get("date_column") if isinstance(date_info, dict) else date_info
        fifth_result = validate_date_range(df, date_col, start_month_year, end_month_year)

        # 📝 Combine results
        result_dict = {
            "empty_transaction_table": first_result,
            "account_dimension_missing": second_result,
            "id_does_not_exist": third_result,
            "must_be_a_leaf_node": fourth_result,
            "date_out_of_range": fifth_result
        }
        
        # print("result_dict: ", result_dict)
        overall_status = compute_overall_status(result_dict)

        # Store in Mongo
        # Store result in Mongo

        object_id = store_in_mongo(output_collection, user_id, project_id, planning_scenario_id, result_dict, overall_status)

        return {
            "statusCode": 200,
            "body": json.dumps({"validation_dict" :result_dict, "object_id": str(object_id)})}
        

    except FileNotFoundError as e:
        logger.error("Dimension files missing: %s", str(e))
        return {
            "statusCode": 404,
            "body": json.dumps({"error": str(e)})
        }

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        return {
            "statusCode": 400,
            "body": json.dumps({"error": str(e)})
        }

    except Exception as e:
        logger.error("Unexpected error: %s", str(e))
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal Server Error", "details": str(e)})
        }

