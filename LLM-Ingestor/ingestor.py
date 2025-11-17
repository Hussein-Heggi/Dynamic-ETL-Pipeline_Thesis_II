"""
Ingestor - Main orchestrator for the financial data ETL pipeline.
MAJOR UPDATE: LLM-driven endpoint selection with FAISS validation and output validation
UPDATED: Semantic validation now uses native features instead of raw query
UPDATED: Returns 4 outputs - (dataframes, enrichment_features, key_features, validation_reports)
         key_features now models query intent with tickers + semantic_keywords + enrichment
"""
import time
from typing import List, Tuple
import pandas as pd

from contracts import (
    APIRequest, ExecutionPlan, ExecutionResults, APIResult, ValidationReport
)
from query_analyzer import QueryAnalyzer
from endpoint_validator import EndpointValidator
from parameter_validator import ParameterValidator
from output_validator import OutputValidator


class Ingestor:
    """
    Main orchestrator for financial data ETL pipeline.
    
    Pipeline flow:
    1. User Prompt → QueryAnalyzer (LLM) → LLMResponse {features, semantic_keywords, api_requests}
    2. LLMResponse → Convert to APIRequests → ExecutionPlan
    3. ExecutionPlan → EndpointValidator (FAISS) → Semantic scores (validates native features)
    4. ExecutionPlan → ParameterValidator → Schema validation
    5. Validated Plan → API Execution → ExecutionResults
    6. ExecutionResults → OutputValidator → ValidationReports
    7. Return (dataframes, enrichment_features, key_features, validation_reports)
    """
    
    def __init__(
        self,
        openai_api_key: str,
        polygon_api_key: str,
        alpha_vantage_api_key: str,
        openai_model: str = "gpt-4o-mini",
        temperature: float = None,
        semantic_threshold: float = 0.7
    ):
        """
        Initialize Ingestor with API credentials
        
        Args:
            openai_api_key: OpenAI API key for query analysis
            polygon_api_key: Polygon API key
            alpha_vantage_api_key: Alpha Vantage API key
            openai_model: OpenAI model to use
            temperature: LLM temperature (None=default)
            semantic_threshold: FAISS similarity threshold (default 0.7)
        """
        # Initialize pipeline components
        self.query_analyzer = QueryAnalyzer(openai_api_key, openai_model, temperature)
        self.endpoint_validator = EndpointValidator(semantic_threshold)
        self.parameter_validator = ParameterValidator()
        self.output_validator = OutputValidator()
        
        # Store original query for validation
        self.current_query = ""
        
        # Initialize API clients
        from polygon_client import PolygonClient
        from alpha_vantage_client import AlphaVantageClient
        
        self.clients = {
            'polygon': PolygonClient(api_key=polygon_api_key),
            'alpha_vantage': AlphaVantageClient(api_key=alpha_vantage_api_key),
        }
    
    def process(
        self, 
        prompt: str, 
        verbose: bool = True
    ) -> Tuple[List[pd.DataFrame], List[str], List[str], List[ValidationReport]]:
        """
        MAIN ENTRY POINT - Process natural language prompt
        UPDATED: Now returns 4 outputs instead of 3
        
        Args:
            prompt: Natural language query for financial data
            verbose: Print pipeline progress
            
        Returns:
            Tuple of:
                - List of DataFrames containing requested financial data
                - List of enrichment features (technical indicators with parameters)
                - List of key features (query intent: tickers + semantic_keywords + enrichment)
                - List of ValidationReports (one per successful dataset)
        """
        start_time = time.time()
        self.current_query = prompt
        
        if verbose:
            print("=" * 80)
            print(f"INGESTING: {prompt}")
            print("=" * 80)
        
        # Step 1: Query Analysis (LLM selects endpoints)
        if verbose:
            print("\n[1/5] Analyzing query with LLM...")
        
        llm_response = self.query_analyzer.analyze(prompt)
        
        if verbose:
            print(f"  ✓ Features:")
            print(f"    - Native: {llm_response.features.native}")
            print(f"    - Enrichment: {llm_response.features.enrichment}")
            print(f"  ✓ Semantic Keywords: {llm_response.semantic_keywords}")
            print(f"  ✓ LLM selected {len(llm_response.api_requests)} endpoint(s):")
            for req in llm_response.api_requests:
                print(f"    - {req.api_name}.{req.endpoint_name}")
        
        # Step 2: Convert to ExecutionPlan
        if verbose:
            print("\n[2/5] Building execution plan...")
        
        execution_plan = self._build_execution_plan(llm_response)
        
        if verbose:
            print(f"  ✓ Created {len(execution_plan.ranked_requests)} request(s)")
        
        # Step 3: Semantic Validation (FAISS validates native features against endpoints)
        if verbose:
            print("\n[3/5] Validating endpoints semantically (FAISS)...")
            print(f"  Validating that endpoints can provide: {llm_response.features.native}")
        
        # UPDATED: Pass native features instead of raw query
        validated_plan = self.endpoint_validator.validate_plan(
            llm_response.features.native, 
            execution_plan
        )
        
        if verbose:
            for req in validated_plan.ranked_requests:
                score_str = f"{req.semantic_score:.2f}" if req.semantic_score else "N/A"
                status_symbol = "✓" if req.validation_status == "VALID" else "⚠ " if req.validation_status == "WARNING" else "✗"
                print(f"  {status_symbol} {req.api_name}.{req.endpoint_name}: score={score_str}, status={req.validation_status}")
        
        # Step 4: Parameter Validation
        if verbose:
            print("\n[4/5] Validating parameters...")
        
        validated_plan = self.parameter_validator.validate_plan(validated_plan)
        
        if verbose:
            for req in validated_plan.ranked_requests:
                if req.validation_errors:
                    for error in req.validation_errors:
                        print(f"      Error: {error}")
                if req.validation_warnings:
                    for warning in req.validation_warnings:
                        print(f"      Warning: {warning}")
        
        # Step 5: API Execution
        if verbose:
            print("\n[5/5] Executing API calls...")
        
        execution_results = self._execute_plan_sequential(validated_plan)
        
        if verbose:
            print(f"  ✓ Status: {execution_results.overall_status}")
            print(f"  ✓ Successful: {len(execution_results.results)}")
            if execution_results.failed_requests:
                print(f"  ✗ Failed: {len(execution_results.failed_requests)}")
        
        # Step 6: Output Validation
        dataframes = [result.data for result in execution_results.results if result.status == "SUCCESS"]
        
        validation_reports = self.output_validator.validate_multiple(
            execution_results.results,
            llm_response.features.native
        )
        
        # Extract enrichment features and key features
        enrichment_features = llm_response.features.enrichment
        key_features = self._extract_key_features(llm_response)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"COMPLETED in {elapsed_time:.0f}ms")
            print(f"Returned {len(dataframes)} dataset(s)")
            
            # Show validation results
            if validation_reports:
                print(f"\nValidation Reports:")
                for i, report in enumerate(validation_reports, 1):
                    status = "✓ PASSED" if report.validation_passed else "✗ FAILED"
                    print(f"  {i}. {report.api_name}.{report.endpoint_name} {status}")
                    if report.ticker:
                        print(f"     Ticker: {report.ticker}")
                    print(f"     Found: {report.found_features}")
                    if report.missing_features:
                        print(f"     Missing: {report.missing_features}")
            
            print(f"\nEnrichment features: {enrichment_features}")
            print(f"Key features (query intent): {key_features}")
            print(f"{'=' * 80}\n")
        
        return dataframes, enrichment_features, key_features, validation_reports
    
    def _build_execution_plan(self, llm_response) -> ExecutionPlan:
        """Convert LLM response to ExecutionPlan with APIRequests"""
        requests = []
        
        for llm_req in llm_response.api_requests:
            api_request = APIRequest(
                api_name=llm_req.api_name,
                endpoint_name=llm_req.endpoint_name,
                parameters=llm_req.parameters,
                validation_status="PENDING"
            )
            requests.append(api_request)
        
        return ExecutionPlan(ranked_requests=requests)
    
    def _extract_key_features(self, llm_response) -> List[str]:
        """
        Extract key features that model query intent.
        UPDATED: key_features = tickers + semantic_keywords + enrichment_features
        
        This represents what the user ASKED FOR, not the infrastructure columns.
        Example: "Show me NVDA daily prices with 20-day SMA"
          → key_features = ["NVDA", "daily", "price", "stock", "SMA_20"]
        
        Args:
            llm_response: LLM response containing tickers, semantic_keywords, and enrichment features
            
        Returns:
            List of features representing query intent
        """
        key_features = []
        
        # Add tickers
        key_features.extend(llm_response.tickers)
        
        # Add semantic keywords (query intent descriptors)
        key_features.extend(llm_response.semantic_keywords)
        
        # Add enrichment features (technical indicators)
        key_features.extend(llm_response.features.enrichment)
        
        # Deduplicate while preserving order
        seen = set()
        deduplicated = []
        for feature in key_features:
            if feature not in seen:
                seen.add(feature)
                deduplicated.append(feature)
        
        return deduplicated
    
    def _execute_plan_sequential(self, execution_plan: ExecutionPlan) -> ExecutionResults:
        """Execute the validated plan sequentially"""
        start_time = time.time()
        
        results = []
        failed_requests = []
        
        # Filter out requests with errors
        valid_requests = [
            req for req in execution_plan.ranked_requests
            if req.validation_status != "ERROR"
        ]
        
        # Execute sequentially
        for request in valid_requests:
            result = self._execute_request(request)
            if result.status == "SUCCESS":
                results.append(result)
            else:
                failed_requests.append(request)
        
        # Determine overall status
        if len(results) == len(valid_requests):
            overall_status = "COMPLETE"
        elif len(results) > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return ExecutionResults(
            results=results,
            failed_requests=failed_requests,
            overall_status=overall_status,
            execution_time_ms=execution_time_ms
        )
    
    def _execute_request(self, request: APIRequest) -> APIResult:
        """Execute a single API request"""
        api_name = request.api_name
        endpoint_name = request.endpoint_name
        parameters = request.parameters.copy()
        
        # Get client
        client = self.clients.get(api_name)
        if not client:
            return APIResult(
                api_name=api_name,
                endpoint_name=endpoint_name,
                status="FAILED",
                error_message=f"No client found for API: {api_name}",
                used_parameters=parameters
            )
        
        try:
            # Remove internal parameters
            client_params = {k: v for k, v in parameters.items() if k not in ['endpoint_type', 'function']}
            
            print(f"  → Calling {api_name}.{endpoint_name}")
            
            # Fetch data
            raw_response = client.fetch_data(parameters)
            
            # Parse response
            parsed_data = client.parse_response(raw_response)
            
            # Handle tuple response (old format)
            if isinstance(parsed_data, tuple):
                df = parsed_data[0]
            else:
                df = parsed_data
            
            # Ensure it's a DataFrame
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            
            # Check if DataFrame is empty
            if df.empty:
                print(f"  ⚠  API returned empty DataFrame")
                return APIResult(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    status="FAILED",
                    error_message="API returned empty dataset",
                    used_parameters=parameters
                )
            
            # Build metadata
            metadata = {
                "rows": df.shape[0],
                "columns": list(df.columns),
                "date_range": {
                    "from": parameters.get("from", parameters.get("date")),
                    "to": parameters.get("to")
                }
            }
            
            print(f"  ✓ Success: {df.shape[0]} rows × {df.shape[1]} columns")
            
            return APIResult(
                api_name=api_name,
                endpoint_name=endpoint_name,
                status="SUCCESS",
                data=df,
                metadata=metadata,
                used_parameters=parameters,
                response_code=200
            )
        
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Error: {error_msg}")
            
            return APIResult(
                api_name=api_name,
                endpoint_name=endpoint_name,
                status="FAILED",
                error_message=error_msg,
                used_parameters=parameters
            )