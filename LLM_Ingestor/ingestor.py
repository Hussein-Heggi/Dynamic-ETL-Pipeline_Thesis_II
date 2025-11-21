"""
Ingestor - Main orchestrator using semantic_keywords for validation
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
    """Main orchestrator for financial data ETL pipeline"""
    
    def __init__(
        self,
        openai_api_key: str= "",
        polygon_api_key: str='',
        alpha_vantage_api_key: str= '',
        openai_model: str = "gpt-5-nano",
        temperature: float = None,
        semantic_threshold: float = 0.7
    ):
        self.query_analyzer = QueryAnalyzer(openai_api_key, openai_model, temperature)
        self.endpoint_validator = EndpointValidator(semantic_threshold)
        self.parameter_validator = ParameterValidator()
        self.output_validator = OutputValidator()
        
        # Store last LLM response for inspection
        self.last_llm_response = None
        
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
    ) -> Tuple[bool, List[pd.DataFrame], List[str], List[str], List[ValidationReport]]:
        """Process natural language prompt"""
        start_time = time.time()
        
        if verbose:
            print("=" * 80)
            print(f"INGESTING: {prompt}")
            print("=" * 80)
        
        # Step 1: Query Analysis
        if verbose:
            print("\n[1/5] Analyzing query with LLM...")
        
        llm_response = self.query_analyzer.analyze(prompt)
        
        # Store for inspection
        self.last_llm_response = llm_response
        
        # Check proceed flag
        if not llm_response.proceed:
            if verbose:
                print("\n  ✗ Query is not finance-related. Aborting.")
                print(f"\n{'=' * 80}")
                print("ABORTED: Non-finance query")
                print(f"{'=' * 80}\n")
            return llm_response.proceed, [], [], [], []
        
        if verbose:
            print(f"  ✓ Proceed: {llm_response.proceed}")
            print(f"  ✓ Features:")
            print(f"    - Native: {llm_response.features.native}")
            print(f"    - Enrichment: {llm_response.features.enrichment}")
            print(f"  ✓ Semantic Keywords: {llm_response.semantic_keywords}")
            print(f"  ✓ LLM selected {len(llm_response.api_requests)} endpoint(s):")
            for req in llm_response.api_requests:
                print(f"    - {req.api_name}.{req.endpoint_name}")
        
        # Step 2: Build ExecutionPlan
        if verbose:
            print("\n[2/5] Building execution plan...")
        
        execution_plan = self._build_execution_plan(llm_response)
        
        if verbose:
            print(f"  ✓ Created {len(execution_plan.ranked_requests)} request(s)")
        
        # Step 3: Semantic Validation (using semantic_keywords)
        if verbose:
            print("\n[3/5] Validating endpoints semantically (FAISS)...")
            print(f"  Validating with keywords: {llm_response.semantic_keywords}")
        
        validated_plan = self.endpoint_validator.validate_plan(
            llm_response.semantic_keywords, 
            execution_plan
        )
        
        if verbose:
            for req in validated_plan.ranked_requests:
                score_str = f"{req.semantic_score:.2f}" if req.semantic_score else "N/A"
                status_symbol = "✓" if req.validation_status == "VALID" else "⚠" if req.validation_status == "WARNING" else "✗"
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
        
        # Extract features
        enrichment_features = llm_response.features.enrichment
        key_features = self._extract_key_features(llm_response)
        
        elapsed_time = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"\n{'=' * 80}")
            print(f"COMPLETED in {elapsed_time:.0f}ms")
            print(f"Returned {len(dataframes)} dataset(s)")
            
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
        
        return llm_response.proceed, dataframes, enrichment_features, key_features, validation_reports
    
    def _build_execution_plan(self, llm_response) -> ExecutionPlan:
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
        key_features = []
        key_features.extend(llm_response.tickers)
        key_features.extend(llm_response.semantic_keywords)
        key_features.extend(llm_response.features.enrichment)
        
        seen = set()
        deduplicated = []
        for feature in key_features:
            if feature not in seen:
                seen.add(feature)
                deduplicated.append(feature)
        
        return deduplicated
    
    def _execute_plan_sequential(self, execution_plan: ExecutionPlan) -> ExecutionResults:
        start_time = time.time()
        results = []
        failed_requests = []
        
        valid_requests = [
            req for req in execution_plan.ranked_requests
            if req.validation_status != "ERROR"
        ]
        
        for request in valid_requests:
            result = self._execute_request(request)
            if result.status == "SUCCESS":
                results.append(result)
            else:
                failed_requests.append(request)
        
        if len(results) == len(valid_requests):
            overall_status = "COMPLETE"
        elif len(results) > 0:
            overall_status = "PARTIAL"
        else:
            overall_status = "FAILED"
        
        return ExecutionResults(
            results=results,
            failed_requests=failed_requests,
            overall_status=overall_status,
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
    
    def _execute_request(self, request: APIRequest) -> APIResult:
        api_name = request.api_name
        endpoint_name = request.endpoint_name
        parameters = request.parameters.copy()
        
        client = self.clients.get(api_name)
        if not client:
            return APIResult(
                api_name=api_name,
                endpoint_name=endpoint_name,
                status="FAILED",
                error_message=f"No client for: {api_name}",
                used_parameters=parameters
            )
        
        try:
            print(f"  → Calling {api_name}.{endpoint_name}")
            
            raw_response = client.fetch_data(parameters)
            parsed_data = client.parse_response(raw_response)
            
            if isinstance(parsed_data, tuple):
                df = parsed_data[0]
            else:
                df = parsed_data
            
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame()
            
            if df.empty:
                print(f"  ⚠  Empty DataFrame")
                return APIResult(
                    api_name=api_name,
                    endpoint_name=endpoint_name,
                    status="FAILED",
                    error_message="Empty dataset",
                    used_parameters=parameters
                )
            
            metadata = {
                "rows": df.shape[0],
                "columns": list(df.columns)
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