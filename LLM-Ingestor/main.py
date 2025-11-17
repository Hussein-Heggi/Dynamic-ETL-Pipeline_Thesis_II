"""
Main entry point - Demonstrates the financial data ETL pipeline.
UPDATED: Handles new return signature (dataframes, key_features, validation_reports)
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
    
    # Example queries

    examples = [
        # Stock data examples
      #  "Show me Apple's daily stock price for the last 30 days",
      #  "Get Tesla and Microsoft weekly data for the past 3 months",
      #  "I need AAPL intraday prices from January 1, 2024 to January 15, 2024",
        
        # Stock data with technical indicators
        "Show me NVDA daily prices with 20-day and 50-day moving averages",
        "Get Apple stock with RSI and MACD indicators for the past month",
        
        # Economic indicator examples
      #  "Show me US GDP growth for the last 5 years",
      #  "Get the unemployment rate data quarterly",
    ]
    
    
    print("=" * 80)
    print("RUNNING EXAMPLE QUERIES")
    print("=" * 80)
    
    for i, prompt in enumerate(examples, 1):  # Run first 2 to avoid rate limits
        print(f"\n\nEXAMPLE {i}/{len(examples)}")
        print("-" * 80)
        
        try:
            # ENTRY POINT: Natural language prompt
            # UPDATED: Now returns (dataframes, key_features, validation_reports)
            dataframes, key_features, validation_reports = ingestor.process(prompt, verbose=True)
            
            # EXIT POINT: Display results
            print("\n📊 RESULTS:")
            print(f"\nKey Features: {key_features}")
            
            print(f"\n🔍 Validation Reports:")
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
                print(df.head())
        
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
    print("\n  dataframes, features, reports = ingestor.process('your query here')")
    print("\nExamples:")
    print("  • dataframes, features, reports = ingestor.process('Get IBM daily prices')")
    print("  • dataframes, features, reports = ingestor.process('Show GDP data')")
    print("  • dataframes, features, reports = ingestor.process('Apple with SMA_20')")
    print("\nThe function returns:")
    print("  1. dataframes: List of pandas DataFrames with the data")
    print("  2. key_features: List of features (tickers, native, enrichment)")
    print("  3. validation_reports: List of ValidationReport objects")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()