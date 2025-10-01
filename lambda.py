import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import sys
import boto3
from io import BytesIO
from pymongo import MongoClient

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



# get the transaction data
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


# get the names of tables created by user in model for accoun, generic dimensions and thier mappings
def get_account_and_generic_mappings(mongo_uri, db_name, project_id, planning_scenario_id):
    """
    Fetch Account and Generic table names with their mappings 
    for a given project_id and planning_scenario_id.
    
    Date type is skipped.

    Returns
    -------
    dict
        Example:
        {
            "Account": {
                "table_name": "glaccount",
                "mapping": "gl_account_id"
            },
            "Generic": [
                {"table_name": "cost_center", "mapping": "cost_center_id"},
                {"table_name": "profit_center", "mapping": "profit_center_id"}
            ]
        }
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db["table_metadata"]
    mapping_collection = db["mapping_results"]

    # Default structure
    result = {
        "Account": {"table_name": None, "mapping": None},
        "Generic": []
    }

    # --- Step 1: get table names from scenarios.tables ---
    doc_tables = collection.find_one({"project_id": project_id}, {"scenarios": 1, "_id": 0})
    if doc_tables:
        for scenario in doc_tables.get("scenarios", []):
            if scenario.get("planning_scenario_id") == planning_scenario_id:
                for table in scenario.get("tables", []):
                    ttype = table.get("Type")
                    tname = table.get("table_name")

                    if ttype == "Account":
                        result["Account"]["table_name"] = tname
                    elif ttype == "Generic":  # allow multiple generic types
                        result["Generic"].append({"table_name": tname, "mapping": None})

    # --- Step 2: get field mappings from planning_scenarios.result ---
    doc_mappings = mapping_collection.find_one({"project_id": project_id}, {"planning_scenarios": 1, "_id": 0})
    if doc_mappings:
        scenario = doc_mappings.get("planning_scenarios", {}).get(planning_scenario_id, {})
        result_map = scenario.get("result", {})

        # Account mapping
        if result["Account"]["table_name"]:
            result["Account"]["mapping"] = result_map.get(result["Account"]["table_name"])

        # Generic mappings
        for gen in result["Generic"]:
            if gen["table_name"] in result_map:
                gen["mapping"] = result_map[gen["table_name"]]

    client.close()
    return result


# to get measure and date dimensions names
def get_measure_and_date_dimension(mongo_uri, db_name, project_id, planning_scenario_id):
    """
    Fetch 'measure' and 'date_dimension' from the columns field 
    for a given project_id and planning_scenario_id.
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db["recommendeddimensions"]

    # Query filter
    query = {
        "project_id": project_id,
        "planning_scenario_id": planning_scenario_id
    }

    # Projection (only fetch the required fields)
    projection = {
        "columns.measure": 1,
        "columns.date_dimension": 1,
        "_id": 0
    }

    # Fetch document
    result = collection.find_one(query, projection)

    client.close()

    if result and "columns" in result:
        return {
            "measure": result["columns"].get("measure"),
            "date_dimension": result["columns"].get("date_dimension")
        }
    return None



def fallback_with_weights(series, n_periods):
    # Base value = last value or mean
    base = series.iloc[-1] if len(series) > 0 else 0
    
    # Example weights (can be tuned or learned from history)
    # Adds variation across months
    month_weights = [1.00, 1.02, 0.98, 1.05, 1.01, 0.97, 
                     1.03, 1.04, 0.96, 1.02, 1.00, 0.99]
    
    preds = []
    for i in range(n_periods):
        w = month_weights[i % 12]   # cycle over 12 months
        preds.append(base * w)
    
    idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(),
                        periods=n_periods, freq='MS')
    
    return pd.Series(preds, index=idx)



def auto_forecast_fpna(df, date, measure, account_id_name, generic_cols_list,
                       project_id, planning_scenario_id, bucket_name, transaction_filename,
                       forecast_periods=12, min_history=3, fallback_window=3, max_lag=12,
                       model_path="trained_models.pkl"):
    """
    AutoForecast FP&A with SARIMA and XGBoost.
    Trains both models, compares metrics, selects the best, stores the model objects 
    in a single pickle file on S3, and returns forecasted DataFrame with Actual + Forecast.

    Returns
    -------
    final_df : pd.DataFrame
        Actual + Forecast combined.
    summary_df : pd.DataFrame
        Model comparison summary.
    """

    # --- Step 0: Preprocess ---
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df[measure] = df[measure].astype(float)
    df['month'] = df[date].dt.to_period('M').dt.to_timestamp()

    # --- Step 1: Aggregate GL × Month ---
    gl_monthly = df.groupby([account_id_name, 'month'])[measure].sum().reset_index()

    # --- Helper functions for SARIMA/XGB forecasting ---
    def forecast_sarima(series, n_periods):
        if len(series) < min_history:
            avg = series[-fallback_window:].mean() if len(series) > 0 else 0
            idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=n_periods, freq='MS')
            return pd.Series([avg]*n_periods, index=idx), None
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        fc = results.forecast(steps=n_periods)
        return fc, results  # return model object

    def forecast_xgb(series, n_periods):
        if len(series) < min_history:
            if len(series) > 1:
                growth = (series.iloc[-1] - series.iloc[0]) / max(1, len(series)-1)
            else:
                growth = 0
            preds = [series.iloc[-1] + (i+1)*growth for i in range(n_periods)]
            idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=n_periods, freq='MS')
            return pd.Series(preds, index=idx), None

        df_feat = pd.DataFrame({measure: series})
        for lag in range(1, max_lag+1):
            df_feat[f'lag_{lag}'] = df_feat[measure].shift(lag)
        df_feat.dropna(inplace=True)
        X = df_feat.drop(columns=[measure])
        y = df_feat[measure]
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X, y)
        preds = []
        history = list(series.values)
        for step in range(n_periods):
            lags = history[-max_lag:]
            if len(lags) < max_lag:
                lags = [0]*(max_lag - len(lags)) + lags
            yhat = model.predict(np.array(lags).reshape(1,-1))[0]
            preds.append(yhat)
            history.append(yhat)
        idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=n_periods, freq='MS')
        return pd.Series(preds, index=idx), model  # return model object

    # --- Step 2: Evaluate models & store trained objects ---
    results_summary = []
    forecasts_list = []
    trained_models = {}  # <-- dictionary to hold SARIMA/XGB objects per GL

    for gl_id in tqdm(gl_monthly[account_id_name].unique(), desc="Evaluating Models"):
        gl_series = gl_monthly[gl_monthly[account_id_name]==gl_id].set_index('month')[measure].asfreq('MS').fillna(0)

        if len(gl_series) < min_history + forecast_periods:
            # ❌ Old flat fallback
            # avg = gl_series[-fallback_window:].mean() if len(gl_series) > 0 else 0
            # fc_idx = pd.date_range(gl_series.index[-1]+pd.offsets.MonthBegin(), periods=forecast_periods, freq='MS')
            # final_fc = pd.Series([avg]*forecast_periods, index=fc_idx)

            # ✅ New weighted fallback
            final_fc = fallback_with_weights(gl_series, forecast_periods)

            results_summary.append({
                "gl_account_id": gl_id, "sarima_rmse": None, "xgb_rmse": None,
                "sarima_mape": None, "xgb_mape": None, "best_model": "Fallback"
            })
            trained_models[gl_id] = {"SARIMA": None, "XGB": None}

        else:
            train = gl_series.iloc[:-forecast_periods]
            test = gl_series.iloc[-forecast_periods:]

            sarima_fc, sarima_model = forecast_sarima(train, forecast_periods)
            sarima_rmse = np.sqrt(mean_squared_error(test, sarima_fc))
            sarima_mape = mean_absolute_percentage_error(test, sarima_fc)

            xgb_fc, xgb_model = forecast_xgb(train, forecast_periods)
            xgb_rmse = np.sqrt(mean_squared_error(test, xgb_fc))
            xgb_mape = mean_absolute_percentage_error(test, xgb_fc)

            if sarima_rmse < xgb_rmse:
                best_model = "SARIMA"
                final_fc, _ = forecast_sarima(gl_series, forecast_periods)
            else:
                best_model = "XGB"
                final_fc, _ = forecast_xgb(gl_series, forecast_periods)

            results_summary.append({
                "gl_account_id": gl_id, "sarima_rmse": sarima_rmse, "xgb_rmse": xgb_rmse,
                "sarima_mape": sarima_mape, "xgb_mape": xgb_mape, "best_model": best_model
            })

            trained_models[gl_id] = {"SARIMA": sarima_model, "XGB": xgb_model}

        fc_df = final_fc.reset_index()
        fc_df.columns = ['month', 'forecast_amount']
        fc_df[account_id_name] = gl_id
        forecasts_list.append(fc_df)

    forecast_df = pd.concat(forecasts_list, ignore_index=True)

    # --- Step 3 & 4: Allocation & final_df (same as original) ---
    group_cols = [account_id_name]
    detail_cols = generic_cols_list
    last_year = df['month'].max().year - 1

    weights_df = (df[df['month'].dt.year==last_year].groupby(group_cols+detail_cols)[measure].sum().reset_index(name='detail_amount'))
    totals = (df[df['month'].dt.year==last_year].groupby(group_cols)[measure].sum().reset_index(name='total_amount'))
    weights_df = weights_df.merge(totals, on=group_cols, how="left")
    weights_df['weight'] = weights_df['detail_amount']/weights_df['total_amount']

    forecast_alloc = forecast_df.merge(weights_df[group_cols+detail_cols+['weight']], on=account_id_name, how='left')
    forecast_alloc['weight'] = forecast_alloc['weight'].fillna(0)
    forecast_alloc['allocated_forecast'] = forecast_alloc['forecast_amount']*forecast_alloc['weight']

    forecast_full = forecast_alloc[['month', account_id_name]+detail_cols].copy()
    forecast_full[measure] = forecast_alloc['allocated_forecast']
    forecast_full['scenario'] = 'Forecast'

    df['scenario'] = 'Actual'
    final_df = pd.concat([df[['month', account_id_name]+detail_cols+[measure,'scenario']], forecast_full], ignore_index=True)

    # --- Step 5: Save trained models to S3 as single pickle ---
    try:
        s3 = boto3.client("s3")
        buffer = BytesIO()
        pickle.dump(trained_models, buffer)
        buffer.seek(0)
        s3_key = f"fpa/forecast_model/{project_id}/{planning_scenario_id}/{transaction_filename}_{model_path}"
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=buffer.getvalue())
        logger.info("✅ All trained models stored on S3 at s3://%s/%s", bucket_name, s3_key)
    except Exception as e:
        logger.error("❌ Failed to store trained models to S3: %s", e, exc_info=True)

    return final_df, pd.DataFrame(results_summary)



def insert_template_to_s3(df, project_id, planning_scenario_id, file_name, bucket_name="dev-ai-analytics-private"):
    logger.info("▶ Storing template to S3 for project_id=%s, planning_scenario_id=%s", project_id, planning_scenario_id)

    if df.empty:
        logger.warning("⚠️ DataFrame is empty. Nothing to store.")
        return None
    
    # if "month" in df.columns:
    #     df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").dt.to_timestamp()

    # Generate deterministic S3 path
    s3_key = f"fpa/forecast_data/{project_id}/{planning_scenario_id}/{file_name}_forecast.parquet"
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



def lambda_handler(event, context):
    """
    AWS Lambda handler to run AutoForecast FP&A pipeline.

    Expects `event` with:
    {
        "project_id": "01K4CZ0YJWT7DS3BHVGVQSJSAD",
        "planning_scenario_id": "01K1TAXXNV4GP7FAKYJSWCCEZP",
        "new_filename_base": "trans",   # parquet filename base
        "mongo_uri": "mongodb+srv://...",
        "db_name": "mydb",
        "bucket_name": "dev-ai-analytics-private"
    }
    """
    try:
        project_id = event["project_id"]
        planning_scenario_id = event["planning_scenario_id"]
        new_filename_base = event["transaction_filename"]
        mongo_uri = event["mongo_uri"]
        db_name = event["db_name"]
        user_id = event["user_id"]
        data_type = event["data_type"]
        budget_type = event["budget_type"] 
        term = event["term"]
        bucket_name = event.get("bucket_name", "dev-ai-analytics-private")

        logger.info("▶ Starting forecast for project=%s, scenario=%s", project_id, planning_scenario_id)

        # --- Step 1: Get transaction data ---
        df = get_transaction_data(project_id, planning_scenario_id, new_filename_base, bucket_name=bucket_name)
        if df.empty:
            raise ValueError("❌ Transaction data not found or empty")

        # --- Step 2: Get measure & date names ---
        measure_date = get_measure_and_date_dimension(mongo_uri, db_name, project_id, planning_scenario_id)
        if not measure_date:
            raise ValueError("❌ Measure and Date Dimension not found in Mongo")

        measure = measure_date["measure"]
        date = measure_date["date_dimension"]

        # --- Step 3: Get account + generic mappings ---
        mappings = get_account_and_generic_mappings(mongo_uri, db_name, project_id, planning_scenario_id)
        account_id_name = mappings["Account"]["mapping"]
        generic_cols_list = [g["mapping"] for g in mappings["Generic"]]

        if not account_id_name or not generic_cols_list:
            raise ValueError("❌ Missing account_id or generic mappings")

        logger.info("✅ Mappings: account_id=%s, generics=%s", account_id_name, generic_cols_list)

        # --- Step 4: Run AutoForecast ---
        final_df, summary_df = auto_forecast_fpna(project_id = project_id,planning_scenario_id = planning_scenario_id, bucket_name = bucket_name, transaction_filename = new_filename_base ,df=df,date=date, measure=measure, account_id_name=account_id_name, generic_cols_list=generic_cols_list)
        forecast_only_df = final_df[final_df["scenario"] == "Forecast"].reset_index(drop=True)
        # return forecast_only_df, pd.DataFrame(results_summary)
        forecast_only_df.to_csv("forecasted.csv")

        print(forecast_only_df.head(6))
        insert_template_to_s3(forecast_only_df, project_id, planning_scenario_id, file_name = new_filename_base)

        logger.info("✅ Forecasting completed. Final rows=%d, Summary rows=%d", forecast_only_df.shape[0], summary_df.shape[0])

        # --- Step 5: Return result (can be stored to S3 or DB instead) ---
        return {
            "statusCode": 200,
            "body": {
                "project_id": project_id,
                "planning_scenario_id": planning_scenario_id,
                "final_rows": forecast_only_df.shape[0],
                "summary_rows": summary_df.shape[0],
                "summary": summary_df.to_dict(orient="records"),
                "mongo_uri" : mongo_uri,
                "db_name" : db_name,
                "user_id": user_id,
                "transaction_filename": new_filename_base,
                "data_type" : data_type,
                "budget_type": budget_type,
                "term": term 
     
            }
        }

    except Exception as e:
        logger.error("❌ Lambda execution failed: %s", str(e), exc_info=True)
        return {
            "statusCode": 500,
            "error": str(e)
        }


if __name__ == "__main__":
    event = {
  "project_id": "01K5DMJ4KE5WSK9BE3KM3M0931",
  "planning_scenario_id": "01K1TAZPVGNR5KK6BBRHZFYWQ7",
  "transaction_filename": "trans",
  "mongo_uri": "mongodb://devfpauser:Ok74c3YE7GLN@13.202.247.111:27017/devfpadb",
  "db_name": "devfpadb",
  "user_id" : 18,
  "data_type" : "forecast",
  "budget_type" : "AI"
}
    response = lambda_handler(event, None)
    print(response)

