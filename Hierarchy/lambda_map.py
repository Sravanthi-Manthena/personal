# Hierarchy code
import pandas as pd
import json
import logging
from pymongo import MongoClient, UpdateOne
import datetime
from collections import defaultdict


# ---------- Setup Logger ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Change to DEBUG for verbose logging
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s")

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def build_internal_hierarchy(df, id_col, parent_col=None):
    logger.debug("Building hierarchy for id_col=%s, parent_col=%s", id_col, parent_col)
    lookup = {}
    child_to_parent = {}

    for _, row in df.iterrows():
        node_id = str(row[id_col]) if pd.notna(row[id_col]) else "ROOT"

        if parent_col and parent_col in df.columns:
            parent_id = str(row[parent_col]) if pd.notna(row[parent_col]) else "ROOT"
            if parent_id == node_id:  # self-loop → treat as root
                parent_id = "ROOT"
        else:
            parent_id = "ROOT"

        lookup[node_id] = {}
        child_to_parent[node_id] = parent_id

    for node_id, parent_id in child_to_parent.items():
        if parent_id == "ROOT":
            continue  # don’t add ROOT as a real node
        if parent_id not in lookup:
            lookup[parent_id] = {}
        lookup[parent_id][node_id] = lookup[node_id]


    roots = {}
    for node_id, parent_id in child_to_parent.items():
        if parent_id == "ROOT" or parent_id not in child_to_parent:
            roots[node_id] = lookup[node_id]

    logger.info("Hierarchy built with %d root nodes", len(roots))
    return roots


# def build_internal_hierarchy(df, id_col, parent_col=None):
#     logger.debug("Building hierarchy for id_col=%s, parent_col=%s", id_col, parent_col)
#     lookup = {}
#     child_to_parent = {}

#     for _, row in df.iterrows():
#         node_id = row[id_col]
#         if parent_col and parent_col in df.columns:
#             parent_id = row[parent_col]
#             if parent_id == node_id:  # self-loop → treat as root
#                 parent_id = "ROOT"
#         else:
#             parent_id = "ROOT"
#         lookup[node_id] = {}
#         child_to_parent[node_id] = parent_id

#     for node_id, parent_id in child_to_parent.items():
#         if parent_id not in lookup:
#             lookup[parent_id] = {}
#         lookup[parent_id][node_id] = lookup[node_id]

#     roots = {}
#     for node_id, parent_id in child_to_parent.items():
#         if parent_id == "ROOT" or parent_id not in child_to_parent:
#             roots[node_id] = lookup[node_id]

#     logger.info("Hierarchy built with %d root nodes", len(roots))
#     return roots



def build_internal_hierarchy(df, id_col, parent_col=None):
    logger.debug("Building hierarchy for id_col=%s, parent_col=%s", id_col, parent_col)

    children_map = defaultdict(list)
    roots = []

    for _, row in df.iterrows():
        node_id = str(row[id_col]) if pd.notna(row[id_col]) else "ROOT"

        if parent_col and parent_col in df.columns:
            parent_id = str(row[parent_col]) if pd.notna(row[parent_col]) else "ROOT"
            if parent_id == node_id:  # self-loop → treat as root
                parent_id = "ROOT"
        else:
            parent_id = "ROOT"

        if parent_id == "ROOT":
            roots.append(node_id)
        else:
            children_map[parent_id].append(node_id)

    # Recursive builder → creates fresh dicts every time
    def build_tree(node_id):
        return {child: build_tree(child) for child in children_map.get(node_id, [])}

    hierarchy = {root: build_tree(root) for root in roots}

    logger.info("Hierarchy built with %d root nodes", len(roots))
    return hierarchy







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
            [grouped_files.get(filename, pd.DataFrame()), row_df],
            ignore_index=True
        )

    if not grouped_files:
        raise FileNotFoundError("❌ No valid dimension files found after filtering")

    col_names_collection = db["table_metadata"]
    dimensions = []

    for fname, df in grouped_files.items():
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

        dimensions.append({
            "dim_name": fname,
            "dim_df": df,
            "id_col": id_col,
            "parent_col": hierarchy_col
        })

    logger.info("✅ Prepared %d dimensions", len(dimensions))
    return dimensions


def upload_hierarchies_nested(db, user_id, project_id, scenarios):
    """
    Upload or update hierarchies for multiple planning scenarios and files
    in a single document per (user_id, project_id).

    Args:
        mongo_uri (str): MongoDB connection URI
        db_name (str): Database name
        collection_name (str): Collection name
        user_id (int/str): User ID
        project_id (str): Project ID
        scenarios (list of dict): Each dict should have:
            - planning_scenario_id (str)
            - files: list of dicts with
                - file_name (str)
                - hierarchy_json (dict)
    """
    try:

        collection = db["hierarchies"]

        # Build update query
        update_dict = {}
        for scenario in scenarios:
            scenario_id = scenario.get("planning_scenario_id")
            files = scenario.get("files", [])
            if not scenario_id or not files:
                continue
            update_dict[f"planning_scenarios.{scenario_id}"] = files

        if not update_dict:
            logger.warning("No valid scenarios/files to upload.")
            return

        # Upsert: if document exists, update; else create
        result = collection.update_one(
            {"user_id": user_id, "project_id": project_id},
            {
                "$set": update_dict,
                "$setOnInsert": {"user_id": user_id, "project_id": project_id, "uploaded_at": datetime.datetime.utcnow()}
            },
            upsert=True
        )

        if result.matched_count:
            logger.info("Updated existing document for user_id=%s, project_id=%s", user_id, project_id)
        else:
            logger.info("Inserted new document for user_id=%s, project_id=%s", user_id, project_id)

    except Exception as e:
        logger.exception("Error uploading hierarchies to MongoDB: %s", e)



def lambda_handler(event, context):
    """
    AWS Lambda handler to build hierarchies for all planning scenarios for a user/project.
    """
    try:
        logger.info("🚀 Lambda execution started")

        # Extract inputs from event
        user_id = event.get("user_id")
        project_id = event.get("project_id")
        mongo_uri = event.get("mongo_uri")

        if not all([user_id, project_id, mongo_uri]):
            logger.error("❌ Missing required parameters in event")
            return {
                "statusCode": 400,
                "body": {"error": "Missing parameters"}
            }

        client = MongoClient(mongo_uri)
        db = client["devfpadb"]
        fcollection = db["fileuploaddata"]

        # ---- Get all planning scenario IDs for this user/project ----
        planning_scenario_ids = fcollection.distinct(
            "planning_scenario_id",
            {"user_id": user_id, "project_id": project_id}
        )

        if not planning_scenario_ids:
            logger.warning("⚠️ No planning scenarios found for this user/project")
            return {
                "statusCode": 404,
                "body": {"error": "No planning scenarios found"}
            }

        scenarios_data = []

        # Loop through all planning scenarios
        for scenario_id in planning_scenario_ids:
            logger.info(f"Processing scenario: {scenario_id}")
            scenario_entry = {"planning_scenario_id": scenario_id, "files": []}

            # Get dimensions/files for this scenario
            dimensions = prepare_dimension_list(db, user_id, project_id, scenario_id)
            logger.info("dimesnions: %s", dimensions)


            for dim in dimensions:
                dim_name = dim["dim_name"] 
                dim_df = dim["dim_df"]
                id_col = dim["id_col"]
                parent_col = dim["parent_col"]

                hierarchy_json = build_internal_hierarchy(dim_df, id_col, parent_col)
                # hierarchy_json = json.dumps(hierarchy, indent=2)
                logger.info("hierarchy_json: %s", hierarchy_json)
                scenario_entry["files"].append({
                    "file_name": dim_name,
                    "hierarchy_json": hierarchy_json
                })

            scenarios_data.append(scenario_entry)

        logger.info("✅ All scenarios processed successfully")

        # ---- Store in MongoDB ----

        upload_hierarchies_nested(db, user_id, project_id, scenarios_data)

        return {
            "statusCode": 200,
            "body":{"message": "Hierarchies built and stored successfully"}
        }

    except Exception as e:
        logger.exception("💥 Error occurred during Lambda execution")
        return {"statusCode": 500, "body": {"error": str(e)}}


if __name__ == "__main__":
    event = {
  "user_id": 18,
  "project_id": "01K4CB7NWE9TKZZJ4P5RXBAWGB",
  "mongo_uri": "mongodb://devfpauser:Ok74c3YE7GLN@13.202.247.111:27017/devfpadb"
}
    response = lambda_handler(event, None)
    print(response)

