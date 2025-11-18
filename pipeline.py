import pandas as pd
from LLM_Ingestor.ingestor import *
from validator import *
from transform.transform import transform_pipeline
import json 


def run_pipeline():
    ingestor = Ingestor()
    prompt = "Show me apple stock prices from last 30 days with sma 10 days"
    dataframes, enrichment_features, key_features, validation_reports = ingestor.process(prompt)
    print("Ingestion completed.")

    validator = Validator()
    Val_Out, report = validator.process(dataframes)

    for idx, df in enumerate(Val_Out):
        print(f"\nOutput DataFrame {idx}:")
        print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)[:10]}" + ("..." if len(df.columns) > 10 else ""))
    
    print("Validation completed.")

    with open("enrichment_features.txt", "r") as f:
        enrichment_features = f.read().splitlines()
    Trans_Out, trans_rep= transform_pipeline(Val_Out, enrichment_features)
    
    for i, df in enumerate(Trans_Out):
        df.to_csv(f"df_2{i}.csv", index=False)
    
    print("Transformation completed.")

if __name__ == "__main__":
    run_pipeline()
