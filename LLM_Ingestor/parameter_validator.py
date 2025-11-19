"""
Parameter Validator - Validates and enriches API request parameters
"""
from datetime import datetime
from typing import Dict, Any, List
import re

from contracts import APIRequest, ExecutionPlan
from api_registry import registry, ParameterType


class ParameterValidator:
    def __init__(self):
        self.registry = registry
    
    def validate_plan(self, execution_plan: ExecutionPlan) -> ExecutionPlan:
        validated_requests = []
        for request in execution_plan.ranked_requests:
            validated_request = self.validate_request(request)
            validated_requests.append(validated_request)
        execution_plan.ranked_requests = validated_requests
        return execution_plan
    
    def validate_request(self, request: APIRequest) -> APIRequest:
        endpoint_spec = self.registry.get_endpoint_spec(request.api_name, request.endpoint_name)
        
        if not endpoint_spec:
            request.validation_status = "ERROR"
            request.validation_errors.append(f"Endpoint {request.endpoint_name} not found")
            return request
        
        errors = []
        warnings = []
        enriched_params = request.parameters.copy()
        
        # Check required parameters
        for param_schema in endpoint_spec.parameters:
            if param_schema.required:
                param_exists = (
                    param_schema.name in enriched_params or
                    any(alias in enriched_params for alias in param_schema.aliases)
                )
                
                if not param_exists:
                    if param_schema.default_value is not None:
                        enriched_params[param_schema.name] = param_schema.default_value
                        warnings.append(f"Using default for {param_schema.name}: {param_schema.default_value}")
                    else:
                        errors.append(f"Missing required: {param_schema.name}")
        
        # Validate provided parameters
        for param_name, param_value in list(enriched_params.items()):
            param_schema = self._get_parameter_schema(endpoint_spec, param_name)
            if not param_schema:
                continue
            
            validation_result = self._validate_parameter(param_name, param_value, param_schema)
            errors.extend(validation_result["errors"])
            warnings.extend(validation_result["warnings"])
            if validation_result["transformed_value"] is not None:
                enriched_params[param_name] = validation_result["transformed_value"]
        
        # API-specific enrichment
        enriched_params = self._enrich_parameters(enriched_params, request.api_name, request.endpoint_name)
        
        request.parameters = enriched_params
        request.validation_errors.extend(errors)
        request.validation_warnings.extend(warnings)
        
        if errors:
            request.validation_status = "ERROR"
        elif warnings:
            request.validation_status = "WARNING"
        elif request.validation_status == "PENDING":
            request.validation_status = "VALID"
        
        return request
    
    def _get_parameter_schema(self, endpoint_spec, param_name: str):
        for param_schema in endpoint_spec.parameters:
            if param_schema.name == param_name or param_name in param_schema.aliases:
                return param_schema
        return None
    
    def _validate_parameter(self, param_name: str, param_value: Any, param_schema) -> Dict[str, Any]:
        result = {"errors": [], "warnings": [], "transformed_value": None}
        
        if param_schema.type == ParameterType.DATE:
            if param_schema.format == "YYYY-MM-DD":
                try:
                    date_obj = datetime.strptime(param_value, "%Y-%m-%d")
                    if date_obj.date() > datetime.now().date():
                        result["warnings"].append(f"{param_name} is in future")
                        param_value = datetime.now().strftime("%Y-%m-%d")
                        result["transformed_value"] = param_value
                except ValueError:
                    result["errors"].append(f"{param_name} must be YYYY-MM-DD")
        
        elif param_schema.type == ParameterType.ENUM:
            if param_schema.valid_values and param_value not in param_schema.valid_values:
                result["errors"].append(f"{param_name} must be one of {param_schema.valid_values}")
        
        if result["transformed_value"] is None:
            result["transformed_value"] = param_value
        
        return result
    
    def _enrich_parameters(self, parameters: Dict[str, Any], api_name: str, endpoint_name: str) -> Dict[str, Any]:
        enriched = parameters.copy()
        endpoint_spec = self.registry.get_endpoint_spec(api_name, endpoint_name)
        
        if api_name == "polygon":
            endpoint_type_map = {
                'get_aggs': 0,
                'get_grouped_daily_aggs': 1,
                'get_daily_open_close_agg': 2,
                'get_previous_close_agg': 3,
                'FULL_MARKET_SNAPSHOT': 4,
            }
            if endpoint_name in endpoint_type_map:
                enriched['endpoint_type'] = endpoint_type_map[endpoint_name]
            elif endpoint_spec and endpoint_spec.data_category == "economic_indicator":
                # Economic indicators don't require tickers; pass indicator hint downstream
                enriched['endpoint_type'] = 'economic_indicator'
                enriched.setdefault('indicator', endpoint_name)
                if 'limit' not in enriched or enriched['limit'] is None:
                    enriched['limit'] = 30
                sort_value = enriched.get('sort')
                if sort_value:
                    if '.' not in sort_value:
                        enriched['sort'] = f"date.{sort_value}"
                else:
                    enriched['sort'] = 'date.desc'
            
            if 'date' in enriched and 'from' not in enriched:
                enriched['from'] = enriched['date']
        
        elif api_name == "alpha_vantage":
            enriched['function'] = endpoint_name
            
            if 'ticker' in enriched:
                enriched['symbol'] = enriched['ticker']
            
            if endpoint_name == "TIME_SERIES_INTRADAY":
                if 'timespan' in enriched:
                    enriched['interval'] = enriched['timespan']
                else:
                    enriched['interval'] = '5min'
            
            elif endpoint_name == "TIME_SERIES_DAILY_ADJUSTED":
                if 'outputsize' not in enriched:
                    enriched['outputsize'] = 'full'
        
        return enriched
