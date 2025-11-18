"""
EndpointValidator - FAISS-based semantic validation of LLM endpoint selections
UPDATED: Now validates endpoint descriptions against extracted native features
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from .contracts import APIRequest, ExecutionPlan
from .api_registry import registry


class EndpointValidator:
    """
    Validates LLM-selected endpoints using FAISS semantic matching.
    UPDATED: Compares native features against endpoint descriptions (not user query)
    Threshold-based: scores must be >= 0.7 to pass.
    """
    
    def __init__(self, semantic_threshold: float = 0.7):
        """
        Initialize validator with FAISS index
        
        Args:
            semantic_threshold: Minimum similarity score (0-1)
        """
        self.threshold = semantic_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Build FAISS index from registry
        self.endpoint_list = registry.list_all_endpoints()
        self.descriptions = [desc for _, _, desc in self.endpoint_list]
        
        # Encode descriptions
        embeddings = self.model.encode(self.descriptions, convert_to_numpy=True)
        
        # Build index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def validate_request(self, native_features: List[str], request: APIRequest) -> APIRequest:
        """
        Validate a single API request against extracted native features using FAISS
        
        Args:
            native_features: List of native features extracted by LLM (e.g., ["open", "high", "low", "close", "volume"])
            request: API request to validate
            
        Returns:
            Validated APIRequest with semantic_score and validation_status updated
        """
        # Find the endpoint description
        endpoint_desc = self._get_endpoint_description(request.api_name, request.endpoint_name)
        
        if not endpoint_desc:
            request.validation_errors.append(f"Endpoint {request.endpoint_name} not found in registry")
            request.validation_status = "ERROR"
            return request
        
        # Convert native features to searchable text
        # e.g., ["open", "high", "low", "close", "volume"] -> "open high low close volume data"
        if native_features:
            features_text = " ".join(native_features) + " data"
        else:
            features_text = "financial data"
        
        # Encode native features and endpoint description
        features_embedding = self.model.encode([features_text], convert_to_numpy=True)
        faiss.normalize_L2(features_embedding)
        
        endpoint_embedding = self.model.encode([endpoint_desc], convert_to_numpy=True)
        faiss.normalize_L2(endpoint_embedding)
        
        # Compute similarity
        similarity = float(np.dot(features_embedding[0], endpoint_embedding[0].T))
        request.semantic_score = similarity
        
        # Apply threshold
        if similarity >= self.threshold:
            if request.validation_status != "ERROR":
                request.validation_status = "VALID"
        elif similarity >= 0.5:
            request.validation_warnings.append(
                f"Low semantic match ({similarity:.2f}). Native features may not align well with this endpoint."
            )
            if request.validation_status == "PENDING":
                request.validation_status = "WARNING"
        else:
            request.validation_errors.append(
                f"Endpoint doesn't match native features semantically (score: {similarity:.2f}, threshold: {self.threshold})"
            )
            request.validation_status = "ERROR"
        
        return request
    
    def validate_plan(self, native_features: List[str], plan: ExecutionPlan) -> ExecutionPlan:
        """
        Validate all requests in an execution plan against native features
        
        Args:
            native_features: List of native features extracted by LLM
            plan: Execution plan to validate
            
        Returns:
            Validated ExecutionPlan
        """
        validated_requests = []
        
        for request in plan.ranked_requests:
            validated_request = self.validate_request(native_features, request)
            validated_requests.append(validated_request)
        
        plan.ranked_requests = validated_requests
        return plan
    
    def _get_endpoint_description(self, api_name: str, endpoint_name: str) -> str:
        """Get description for an endpoint"""
        endpoint_spec = registry.get_endpoint_spec(api_name, endpoint_name)
        return endpoint_spec.description if endpoint_spec else None