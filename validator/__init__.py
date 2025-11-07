"""
Validator Component for ETL Pipeline

Performs intelligent UNION and JOIN operations on dataframes using ML models.

Main Components:
- UnionEngine: Column matching and vertical concatenation
- JoinEngine: Row matching and horizontal concatenation  
- Validator: Main orchestrator class

Usage:
    from validator import Validator
    
    validator = Validator()
    output_dfs, report = validator.process(input_dataframes)
    print(validator.get_summary(report))
"""

from .validator import Validator
from .config import ValidatorConfig

__all__ = ['Validator', 'ValidatorConfig']
__version__ = '1.0.0'