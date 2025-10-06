import pandas as pd
from openai import OpenAI
import json
import logging
import sys
from pymongo import MongoClient, ReturnDocument
from datetime import datetime
import boto3
from io import BytesIO

# Get Lambda's root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add console handler only if running locally (not in AWS Lambda)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def map_table_columns_to_dimensions(table:pd.DataFrame, dimensions, groq_key):
    logger.info("Starting mapping of table columns to dimensions...")
    sample_data = table.copy()
    sample_data = sample_data.head(3)
    print(sample_data)
    sample_json = json.dumps(sample_data.to_dict(), indent=2)
    # print("sample_json: ", sample_json)
    logger.debug("Sample transactional data for mapping:\n%s", sample_json)

    prompt = f'''
You are a data mapping expert. Your task is to analyze a table and map its column names to a given list of dimension names.

**Input:**
- Table: {sample_json}
- Dimension Names: {dimensions}

**Task:**
Carefully analyze the table column names and match them to the most appropriate dimension names from the provided list.

**Output Format:**
Return your analysis as a JSON object where each dimension name is mapped to its corresponding table column name. If no suitable match is found for a dimension, map it to null.
```json
{{
  "dimension_name_1": "mapped_column_name_1",
  "dimension_name_2": "mapped_column_name_2", 
  "dimension_name_3": null,
  "dimension_name_n": "mapped_column_name_n"
}}
'''
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_key)
        res = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "Classify tables."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        mapping_result = json.loads(res.choices[0].message.content)
        logger.info("Groq successfully mapped columns to dimensions.")
        logger.debug("Mapping result: %s", mapping_result)
        return mapping_result
    except Exception as e:
        logger.error("Groq mapping failed: %s", str(e))
        raise

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
    

# 2️⃣ Function to fetch filenames
def get_filename_names(files_collection, user_id, project_id, planning_scenario_id):
    try:
        file_query = {
            "user_id": user_id,
            "project_id": project_id,
            "planning_scenario_id": planning_scenario_id
        }
        filename_list = files_collection.distinct("filename", file_query)
        logger.info("Fetched %d filenames for scenario.", len(filename_list))
        return filename_list
    except Exception as e:
        logger.error("Error fetching filenames: %s", str(e))
        raise


# 3️⃣ Function to store results in MongoDB
def store_in_mongo(output_collection, user_id, project_id, planning_scenario_id, mapping_result, columns):
    try:
        result_doc = output_collection.find_one_and_update(
            {"user_id": user_id, "project_id": project_id},
            {
                "$set": {
                    f"planning_scenarios.{planning_scenario_id}": {
                        "result": mapping_result,
                        "transaction_columns": list(columns),
                        "time_stamp": datetime.utcnow().isoformat()
                    }
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        object_id = result_doc["_id"]
        logger.info("Stored mapping result in MongoDB. ObjectId: %s", object_id)
        return object_id
    except Exception as e:
        logger.error("Error storing result in MongoDB: %s", str(e))
        raise

# Lambda handler
def lambda_handler(event, context):
    logger.info("Lambda triggered with event: %s", event)

    try:
        # Parse inputs
        groq_key = event["groq_key"]
        user_id = event["user_id"]
        project_id = event["project_id"]
        planning_scenario_id = event["planning_scenario_id"]
        measures = event["measures"]
        mongo_uri = event["mongo_uri"]
        db_name = event["db_name"]
        transaction_filename = event["transaction_filename"]
        files_collection_name = "fileuploaddata"
        trans_collection_name = "transactionaldata"
        output_collection_name = "mapping_results"

        # Connect to MongoDB
        mongo_client = MongoClient(mongo_uri)
        db = mongo_client[db_name]
        files_collection = db[files_collection_name]
        trans_collection = db[trans_collection_name]
        output_collection = db[output_collection_name]

        # Fetch transactional data
        df = get_transaction_data(project_id=project_id, planning_scenario_id=planning_scenario_id, new_filename_base=transaction_filename)

        if df is None:
            return {"statusCode": 404, "body": json.dumps({"error": "No transactional data found"})}
        columns = df.columns.tolist()
        print("after fetching from s3: ", df.head(3))

        # Get filenames
        filename_list = get_filename_names(files_collection, user_id, project_id, planning_scenario_id)
        print("filenames_list : ", filename_list)

        # Combine measures + filenames
        combined_list = measures + filename_list
        logger.debug("Combined dimension list: %s", combined_list)
        print("combined _ list : ", combined_list)

        # Map table columns to dimensions
        mapping_result = map_table_columns_to_dimensions(df, combined_list, groq_key)


        # Store result in Mongo
        object_id = store_in_mongo(output_collection, user_id, project_id, planning_scenario_id, mapping_result, columns)

        # Success response
        return {"statusCode": 200, "body": json.dumps({"mapping": mapping_result, "object_id": str(object_id)})}

    except KeyError as e:
        return {"statusCode": 400, "body": json.dumps({"error": f"Missing required field: {str(e)}"})}
    except Exception as e:
        logger.exception("Unexpected error occurred")
        return {"statusCode": 500, "body": json.dumps({"error": f"Unexpected error: {str(e)}"})}





