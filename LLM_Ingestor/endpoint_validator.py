"""
EndpointValidator - Use semantic keywords for validation instead of native features
"""
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from contracts import APIRequest, ExecutionPlan
from api_registry import registry


class EndpointValidator:
    """Validates endpoints using FAISS semantic matching on query intent"""
    
    def __init__(self, semantic_threshold: float = 0.7):
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
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
    
    def validate_request(self, semantic_keywords: List[str], request: APIRequest) -> APIRequest:
        """Validate request using semantic keywords"""
        # Get endpoint description
        endpoint_desc = self._get_endpoint_description(request.api_name, request.endpoint_name)
        
        if not endpoint_desc:
            request.validation_errors.append(f"Endpoint {request.endpoint_name} not found")
            request.validation_status = "ERROR"
            return request
        
        # Build query text from semantic keywords
        if semantic_keywords:
            query_text = " ".join(semantic_keywords)
        else:
            query_text = "financial data"
        
        # Encode query and endpoint
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        endpoint_embedding = self.model.encode([endpoint_desc], convert_to_numpy=True)
        faiss.normalize_L2(endpoint_embedding)
        
        # Compute similarity
        similarity = float(np.dot(query_embedding[0], endpoint_embedding[0].T))
        request.semantic_score = similarity
        
        # Apply threshold - semantic validation should not block execution
        if similarity >= self.threshold:
            if request.validation_status != "ERROR":
                request.validation_status = "VALID"
        else:
            severity = "Low" if similarity >= 0.5 else "Very low"
            request.validation_warnings.append(
                f"{severity} semantic match ({similarity:.2f}, threshold: {self.threshold})"
            )
            if request.validation_status == "PENDING":
                request.validation_status = "WARNING"
        
        return request
    
    def validate_plan(self, semantic_keywords: List[str], plan: ExecutionPlan) -> ExecutionPlan:
        """Validate all requests in plan"""
        validated_requests = []
        
        for request in plan.ranked_requests:
            validated_request = self.validate_request(semantic_keywords, request)
            validated_requests.append(validated_request)
        
        plan.ranked_requests = validated_requests
        return plan
    
    def _get_endpoint_description(self, api_name: str, endpoint_name: str) -> str:
        endpoint_spec = registry.get_endpoint_spec(api_name, endpoint_name)
        return endpoint_spec.description if endpoint_spec else None
