"""
Main entry point - Demonstrates the financial data ETL pipeline.
UPDATED: Handles new return signature (dataframes, enrichment_features, key_features, validation_reports)
UPDATED: Demonstrates proceed flag handling for non-finance queries
"""
import os
from ingestor import Ingestor


def main():
    """Main demonstration function"""
    
    # Configuration
    config = {
        
    }
    
    print("Initializing Financial Data Ingestor...")
    print("Building FAISS index for semantic validation...\n")
    
    ingestor = Ingestor(
        openai_api_key=config['openai_api_key'],
        polygon_api_key=config['polygon_api_key'],
        alpha_vantage_api_key=config['alpha_vantage_api_key'],
        openai_model=config['openai_model'],
        semantic_threshold=0.7  # FAISS threshold
    )
    
    print("✓ Ingestor ready!\n")
    
    # Example queries demonstrating various features
    examples = [
        # Non-finance query (should return proceed=false)
    #    "What's the best pizza in New York?",
        
        # Vague stock query (should use defaults)
    #    "I want stock data",
        
        # Vague query with ticker (1-month daily default)
    #    "Show me AAPL",
        
        # Vague economic query (should use CPI/INFLATION defaults)
    #   "Show me economic indicators",
        
        # Specific stock query
     #  "Get Apple's daily stock price for the last 30 days",
        
        # Stock data with technical indicators
     #   "Show me NVDA daily prices with 20-day and 50-day moving averages",
        
        # Intraday (user specified)
        #"TSLA 5-minute data for today",
      #  "show me all the financial data for apple",
      #  "i want the income statement for google, and the stock price for microsoft",
        
        # Economic indicator
      # "Show me US GDP growth for the last 5 years",

        # Economic indicator with multi-column output
      #  "Show recent US Treasury yield curve data",
     # "Compare US inflation with AAPL's monthly adjusted returns over the last 10 years. I want datasets for both"
     "i want the income statement for coca cola yearly."
    ]
    
    print("=" * 80)
    print("RUNNING EXAMPLE QUERIES")
    print("=" * 80)
    
    for i, prompt in enumerate(examples, 1):
        print(f"\n\nEXAMPLE {i}/{len(examples)}")
        print("-" * 80)
        
        try:
            # ENTRY POINT: Natural language prompt
            # Returns 5 outputs
            proceed, dataframes, enrichment_features, key_features, validation_reports = ingestor.process(prompt, verbose=True)
            
            # Check if we got results
            if not proceed:
                print("\n📊 RESULTS: No data returned (non-finance query or error)")
                continue
            
            if not dataframes and not enrichment_features and not key_features:
                print("\n📊 RESULTS: No data returned (non-finance query or error)")
                continue
            
            # EXIT POINT: Display results
            print("\n📊 RESULTS:")
            print(f"\nEnrichment Features: {enrichment_features}")
            print(f"Key Features (Query Intent): {key_features}")
            
            # Display LLM's parameter selections
            if ingestor.last_llm_response and ingestor.last_llm_response.api_requests:
                print(f"\n🔧 LLM Parameter Selections:")
                for j, api_req in enumerate(ingestor.last_llm_response.api_requests, 1):
                    print(f"\n  Request {j}:")
                    print(f"    API: {api_req.api_name}")
                    print(f"    Endpoint: {api_req.endpoint_name}")
                    print(f"    Parameters:")
                    for param_key, param_value in api_req.parameters.items():
                        print(f"      - {param_key}: {param_value}")
                    if api_req.reasoning:
                        print(f"    Reasoning: {api_req.reasoning}")
            
            print(f"\n📋 Validation Reports:")
            for j, report in enumerate(validation_reports, 1):
                status = "✓ PASSED" if report.validation_passed else "✗ FAILED"
                print(f"\n  Report {j}: {status}")
                print(f"    API: {report.api_name}")
                print(f"    Endpoint: {report.endpoint_name}")
                if report.ticker:
                    print(f"    Ticker: {report.ticker}")
                print(f"    Found features: {report.found_features}")
                print(f"    Actual columns: {report.actual_columns}")
                if report.missing_features:
                    print(f"    ⚠  Missing features: {report.missing_features}")
                if report.fuzzy_matched_features:
                    print(f"    ~ Fuzzy matched: {report.fuzzy_matched_features}")
            
            print(f"\n📈 DataFrames:")
            for j, df in enumerate(dataframes, 1):
                print(f"\n  Dataset {j}:")
                print(f"    Shape: {df.shape}")
                print(f"    Columns: {list(df.columns)}")
                print("\n    First 5 rows:")
                print(df.to_string())
        
        except Exception as e:
            print(f"\n❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 80)
    
    # Usage guide
    print("\n\n" + "=" * 80)
    print("USAGE GUIDE")
    print("=" * 80)
    print("\nYou can now use the ingestor with any custom query:")
    print("\n  proceed, dataframes, enrichment, key_features, reports = ingestor.process('your query here')")
    print("\nExamples:")
    print("  • proceed, dataframes, enrichment, key_features, reports = ingestor.process('Get IBM daily prices')")
    print("  • proceed, dataframes, enrichment, key_features, reports = ingestor.process('Show GDP data')")
    print("  • proceed, dataframes, enrichment, key_features, reports = ingestor.process('Apple with SMA_20')")
    print("\nThe function returns:")
    print("  1. proceed: Boolean flag indicating if the query was finance-related")
    print("  2. dataframes: List of pandas DataFrames with the data")
    print("  3. enrichment_features: List of technical indicators (e.g., ['SMA_20', 'RSI_14'])")
    print("  4. key_features: List modeling query intent (tickers + keywords + enrichment)")
    print("  5. validation_reports: List of ValidationReport objects")
    print("\nDefault behaviors:")
    print("  • Non-finance queries return empty results (proceed=false)")
    print("  • Vague stock queries without ticker → Polygon FULL_MARKET_SNAPSHOT")
    print("  • Vague stock queries with ticker → 1-month daily data from both APIs")
    print("  • Vague economic queries → Polygon INFLATION + Alpha Vantage CPI")
    print("  • All timestamps are date-only (no time component)")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()