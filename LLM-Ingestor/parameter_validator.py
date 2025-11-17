"""
Parameter Validator - Validates and enriches APIRequests against parameter schemas.
UPDATED: Skips validation for enrichment_features (they will be calculated later)
"""
from datetime import datetime
from typing import Dict, Any, List
import re

from contracts import APIRequest, ExecutionPlan
from api_registry import registry, ParameterType


class ParameterValidator:
    """Validates and enriches API request parameters"""
    
    def __init__(self):
        """Initialize Parameter Validator"""
        self.registry = registry
    
    def validate_plan(self, execution_plan: ExecutionPlan) -> ExecutionPlan:
        """
        Validate all APIRequests in an ExecutionPlan
        
        Args:
            execution_plan: Plan to validate
            
        Returns:
            ExecutionPlan with validated and enriched requests
        """
        validated_requests = []
        
        for request in execution_plan.ranked_requests:
            validated_request = self.validate_request(request)
            validated_requests.append(validated_request)
        
        execution_plan.ranked_requests = validated_requests
        return execution_plan
    
    def validate_request(self, request: APIRequest) -> APIRequest:
        """
        Validate and enrich a single APIRequest
        
        Args:
            request: Request to validate
            
        Returns:
            Validated and enriched APIRequest
        """
        # Get endpoint spec
        endpoint_spec = self.registry.get_endpoint_spec(
            request.api_name,
            request.endpoint_name
        )
        
        if not endpoint_spec:
            request.validation_status = "ERROR"
            request.validation_errors.append(f"Endpoint {request.endpoint_name} not found in registry")
            return request
        
        # Validate each parameter
        errors = []
        warnings = []
        enriched_params = request.parameters.copy()
        
        # Check required parameters
        for param_schema in endpoint_spec.parameters:
            if param_schema.required:
                # Check if parameter exists (including aliases)
                param_exists = (
                    param_schema.name in enriched_params or
                    any(alias in enriched_params for alias in param_schema.aliases)
                )
                
                if not param_exists:
                    # Add default if available
                    if param_schema.default_value is not None:
                        enriched_params[param_schema.name] = param_schema.default_value
                        warnings.append(f"Using default value for {param_schema.name}: {param_schema.default_value}")
                    else:
                        errors.append(f"Missing required parameter: {param_schema.name}")
        
        # Validate each provided parameter
        for param_name, param_value in list(enriched_params.items()):
            param_schema = self._get_parameter_schema(endpoint_spec, param_name)
            
            if not param_schema:
                warnings.append(f"Unknown parameter: {param_name}")
                continue
            
            # Validate type and constraints
            validation_result = self._validate_parameter(
                param_name,
                param_value,
                param_schema
            )
            
            if validation_result["errors"]:
                errors.extend(validation_result["errors"])
            if validation_result["warnings"]:
                warnings.extend(validation_result["warnings"])
            if validation_result["transformed_value"] is not None:
                enriched_params[param_name] = validation_result["transformed_value"]
        
        # Cross-parameter validation
        cross_validation = self._cross_validate_parameters(enriched_params, request.api_name)
        errors.extend(cross_validation["errors"])
        warnings.extend(cross_validation["warnings"])
        
        # API-specific enrichment
        enriched_params = self._enrich_parameters(
            enriched_params,
            request.api_name,
            request.endpoint_name
        )
        
        # Update request
        request.parameters = enriched_params
        request.validation_errors = errors
        request.validation_warnings = warnings
        
        if errors:
            request.validation_status = "ERROR"
        elif warnings:
            request.validation_status = "WARNING"
        else:
            request.validation_status = "VALID"
        
        return request
    
    def _get_parameter_schema(self, endpoint_spec, param_name: str):
        """Get parameter schema by name or alias"""
        for param_schema in endpoint_spec.parameters:
            if param_schema.name == param_name or param_name in param_schema.aliases:
                return param_schema
        return None
    
    def _validate_parameter(
        self,
        param_name: str,
        param_value: Any,
        param_schema
    ) -> Dict[str, Any]:
        """
        Validate a single parameter against its schema
        
        Returns:
            Dict with 'errors', 'warnings', and 'transformed_value'
        """
        result = {
            "errors": [],
            "warnings": [],
            "transformed_value": None
        }
        
        # Type validation
        if param_schema.type == ParameterType.STRING:
            if not isinstance(param_value, str):
                param_value = str(param_value)
                result["warnings"].append(f"{param_name} converted to string")
            
            # Pattern validation
            if param_schema.pattern and not re.match(param_schema.pattern, param_value):
                result["errors"].append(f"{param_name} does not match pattern {param_schema.pattern}")
        
        elif param_schema.type == ParameterType.INTEGER:
            if not isinstance(param_value, int):
                try:
                    param_value = int(param_value)
                    result["warnings"].append(f"{param_name} converted to integer")
                except ValueError:
                    result["errors"].append(f"{param_name} must be an integer")
                    return result
            
            # Range validation
            if param_schema.min_value is not None and param_value < param_schema.min_value:
                result["errors"].append(f"{param_name} must be >= {param_schema.min_value}")
            if param_schema.max_value is not None and param_value > param_schema.max_value:
                result["errors"].append(f"{param_name} must be <= {param_schema.max_value}")
        
        elif param_schema.type == ParameterType.DATE:
            # Validate date format
            if param_schema.format == "YYYY-MM-DD":
                try:
                    date_obj = datetime.strptime(param_value, "%Y-%m-%d")
                    
                    # Check if date is in the future
                    if date_obj.date() > datetime.now().date():
                        result["warnings"].append(f"{param_name} is in the future, may be invalid")
                        # Replace with today
                        param_value = datetime.now().strftime("%Y-%m-%d")
                        result["transformed_value"] = param_value
                except ValueError:
                    result["errors"].append(f"{param_name} must be in YYYY-MM-DD format")
        
        elif param_schema.type == ParameterType.ENUM:
            if param_value not in param_schema.valid_values:
                result["errors"].append(
                    f"{param_name} must be one of {param_schema.valid_values}, got '{param_value}'"
                )
        
        if result["transformed_value"] is None:
            result["transformed_value"] = param_value
        
        return result
    
    def _cross_validate_parameters(
        self,
        parameters: Dict[str, Any],
        api_name: str
    ) -> Dict[str, List[str]]:
        """Cross-parameter validation rules"""
        result = {"errors": [], "warnings": []}
        
        # Date range validation
        if "from" in parameters and "to" in parameters:
            try:
                from_date = datetime.strptime(parameters["from"], "%Y-%m-%d")
                to_date = datetime.strptime(parameters["to"], "%Y-%m-%d")
                
                if from_date > to_date:
                    result["errors"].append("'from' date must be before 'to' date")
            except ValueError:
                pass  # Individual date validation will catch this
        
        return result
    
    def _enrich_parameters(
        self,
        parameters: Dict[str, Any],
        api_name: str,
        endpoint_name: str
    ) -> Dict[str, Any]:
        """Add API-specific parameters and transformations"""
        enriched = parameters.copy()
        
        if api_name == "polygon":
            # Add endpoint_type mapping
            endpoint_type_map = {
                'get_aggs': 0,
                'get_grouped_daily_aggs': 1,
                'get_daily_open_close_agg': 2,
                'get_previous_close_agg': 3
            }
            
            if endpoint_name in endpoint_type_map:
                enriched['endpoint_type'] = endpoint_type_map[endpoint_name]
            
            # Map 'date' to 'from' if needed
            if 'date' in enriched and 'from' not in enriched:
                enriched['from'] = enriched['date']
        
        elif api_name == "alpha_vantage":
            # Add function name for Alpha Vantage
            enriched['function'] = endpoint_name
            
            # Keep ticker but also add symbol (AlphaVantage client accepts both)
            if 'ticker' in enriched:
                enriched['symbol'] = enriched['ticker']
                # Don't remove ticker, client can use either
            
            # Handle intraday interval parameter
            if endpoint_name == "TIME_SERIES_INTRADAY":
                # Ensure timespan is a valid intraday interval
                if 'timespan' in enriched:
                    valid_intervals = ['1min', '5min', '15min', '30min', '60min']
                    if enriched['timespan'] not in valid_intervals:
                        # Transform "intraday" to default "5min"
                        enriched['timespan'] = '5min'  # Default
                    # Also set interval (Alpha Vantage client accepts both)
                    enriched['interval'] = enriched['timespan']
                else:
                    # No timespan provided, use default
                    enriched['timespan'] = '5min'
                    enriched['interval'] = '5min'
        
        return enriched