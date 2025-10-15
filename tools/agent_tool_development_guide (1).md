# Agentic System Tool Development Guide

## Table of Contents
1. [Understanding Agentic Tools](#understanding-agentic-tools)
2. [Tool Design Principles](#tool-design-principles)
3. [Tool Implementation Guidelines](#tool-implementation-guidelines)
4. [Code Optimization Strategies](#code-optimization-strategies)
5. [System Analysis Framework](#system-analysis-framework)
6. [Testing and Validation](#testing-and-validation)
7. [Common Patterns and Examples](#common-patterns-and-examples)

## Understanding Agentic Tools

### What are Agentic Tools?
Agentic tools are discrete functions that extend an AI agent's capabilities beyond text generation. They allow agents to:
- Interact with external APIs and services
- Process and manipulate data
- Perform calculations and analysis
- Access databases and file systems
- Control external systems and workflows

### Tool Categories
**Data Tools**: File readers, database connectors, web scrapers
**Computation Tools**: Calculators, analyzers, processors
**Communication Tools**: Email senders, API clients, notification systems
**Control Tools**: System controllers, workflow managers
**Integration Tools**: Third-party service connectors

## Tool Design Principles

### 1. Single Responsibility
Each tool should have one clear, well-defined purpose:
```python
# Good: Specific, focused tool
def calculate_compound_interest(principal, rate, time, compounds_per_year):
    return principal * (1 + rate/compounds_per_year) ** (compounds_per_year * time)

# Bad: Tool trying to do too much
def financial_calculator(operation, **kwargs):
    # Handles loans, interest, taxes, budgets, etc.
```

### 2. Clear Input/Output Contracts
Define explicit schemas for inputs and outputs:
```python
from typing import Dict, List, Optional
from pydantic import BaseModel

class WeatherRequest(BaseModel):
    location: str
    units: Optional[str] = "metric"
    days: Optional[int] = 1

class WeatherResponse(BaseModel):
    location: str
    temperature: float
    conditions: str
    forecast: List[Dict]
```

### 3. Error Handling and Graceful Degradation
Tools should handle errors gracefully and provide meaningful feedback:
```python
def get_stock_price(symbol: str) -> Dict:
    try:
        response = api_client.get(f"/stocks/{symbol}")
        return {"success": True, "price": response.json()["price"]}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"API request failed: {str(e)}"}
    except KeyError:
        return {"success": False, "error": f"Invalid symbol: {symbol}"}
```

### 4. Idempotency
When possible, tools should be idempotent (same input = same output):
```python
# Good: Idempotent
def hash_string(input_string: str) -> str:
    return hashlib.sha256(input_string.encode()).hexdigest()

# Problematic: Non-idempotent without proper handling
def create_user_account(username: str):
    # Should check if user exists first
    pass
```

## Tool Implementation Guidelines

### Tool Structure Template
```python
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Define your input schema here"""
    parameter1: str
    parameter2: Optional[int] = None

class ToolOutput(BaseModel):
    """Define your output schema here"""
    success: bool
    result: Any
    message: Optional[str] = None

class YourTool:
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize tool with configuration"""
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self, input_data: ToolInput) -> ToolOutput:
        """Main execution method"""
        try:
            # Validate input
            self._validate_input(input_data)
            
            # Perform core logic
            result = self._perform_operation(input_data)
            
            # Return success response
            return ToolOutput(
                success=True,
                result=result,
                message="Operation completed successfully"
            )
            
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            return ToolOutput(
                success=False,
                result=None,
                message=f"Error: {str(e)}"
            )
    
    def _validate_input(self, input_data: ToolInput) -> None:
        """Validate input parameters"""
        # Add validation logic here
        pass
    
    def _perform_operation(self, input_data: ToolInput) -> Any:
        """Core business logic"""
        # Implement your tool's main functionality here
        pass
```

### Configuration Management
```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolConfig:
    api_key: Optional[str] = None
    base_url: str = "https://api.example.com"
    timeout: int = 30
    retry_count: int = 3
    
    @classmethod
    def from_env(cls):
        return cls(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL", cls.base_url),
            timeout=int(os.getenv("TIMEOUT", cls.timeout)),
            retry_count=int(os.getenv("RETRY_COUNT", cls.retry_count))
        )
```

### Logging and Monitoring
```python
import logging
import time
from functools import wraps

def monitor_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__name__)
        
        try:
            logger.info(f"Starting execution with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Execution completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Execution failed after {execution_time:.2f}s: {str(e)}")
            raise
    
    return wrapper
```

## Code Optimization Strategies

### 1. Caching and Memoization
```python
from functools import lru_cache
import redis

# In-memory caching
@lru_cache(maxsize=128)
def expensive_calculation(n: int) -> int:
    # Expensive operation here
    return result

# Redis caching for distributed systems
class CachedTool:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
    
    def get_cached_or_compute(self, key: str, compute_func, *args, **kwargs):
        cached_result = self.redis_client.get(key)
        if cached_result:
            return json.loads(cached_result)
        
        result = compute_func(*args, **kwargs)
        self.redis_client.setex(key, self.cache_ttl, json.dumps(result))
        return result
```

### 2. Async Operations
```python
import asyncio
import aiohttp
from typing import List

class AsyncWebTool:
    async def fetch_multiple_urls(self, urls: List[str]) -> List[Dict]:
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
    
    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Dict:
        try:
            async with session.get(url) as response:
                return {
                    "url": url,
                    "status": response.status,
                    "data": await response.json()
                }
        except Exception as e:
            return {"url": url, "error": str(e)}
```

### 3. Resource Management
```python
from contextlib import contextmanager
import psycopg2

@contextmanager
def database_connection(connection_string: str):
    conn = None
    try:
        conn = psycopg2.connect(connection_string)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

# Usage
def query_database(query: str) -> List[Dict]:
    with database_connection(DB_CONNECTION_STRING) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        return cursor.fetchall()
```

### 4. Batch Processing
```python
from typing import Iterator, List, TypeVar

T = TypeVar('T')

def batch_process(items: List[T], batch_size: int = 100) -> Iterator[List[T]]:
    """Process items in batches to avoid memory issues"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def process_large_dataset(data: List[Dict]) -> List[Dict]:
    results = []
    for batch in batch_process(data, batch_size=50):
        batch_results = process_batch(batch)
        results.extend(batch_results)
    return results
```

## System Analysis Framework

### 1. Performance Metrics
```python
import time
import psutil
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    execution_time: float
    memory_usage: float
    cpu_usage: float
    success_rate: float
    error_count: int

class PerformanceAnalyzer:
    def __init__(self):
        self.metrics = {}
    
    def measure_tool_performance(self, tool_name: str, execution_func, *args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            result = execution_func(*args, **kwargs)
            success = True
        except Exception as e:
            result = None
            success = False
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=psutil.cpu_percent(),
            success_rate=1.0 if success else 0.0,
            error_count=0 if success else 1
        )
        
        self._record_metrics(tool_name, metrics)
        return result
```

### 2. Tool Usage Analytics
```python
class ToolAnalytics:
    def __init__(self):
        self.usage_stats = {}
        self.performance_history = {}
    
    def track_usage(self, tool_name: str, input_size: int, success: bool):
        if tool_name not in self.usage_stats:
            self.usage_stats[tool_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "average_input_size": 0,
                "last_used": None
            }
        
        stats = self.usage_stats[tool_name]
        stats["total_calls"] += 1
        if success:
            stats["successful_calls"] += 1
        stats["average_input_size"] = (
            (stats["average_input_size"] * (stats["total_calls"] - 1) + input_size) 
            / stats["total_calls"]
        )
        stats["last_used"] = time.time()
    
    def get_analytics_report(self) -> Dict[str, Any]:
        return {
            "most_used_tools": sorted(
                self.usage_stats.items(),
                key=lambda x: x[1]["total_calls"],
                reverse=True
            )[:10],
            "tool_reliability": {
                name: stats["successful_calls"] / stats["total_calls"]
                for name, stats in self.usage_stats.items()
                if stats["total_calls"] > 0
            }
        }
```

### 3. System Health Monitoring
```python
class SystemHealthMonitor:
    def __init__(self):
        self.health_checks = []
        self.alerts = []
    
    def add_health_check(self, name: str, check_func):
        self.health_checks.append({
            "name": name,
            "check": check_func,
            "last_status": None,
            "last_check": None
        })
    
    def run_health_checks(self) -> Dict[str, Any]:
        results = {}
        for check in self.health_checks:
            try:
                status = check["check"]()
                check["last_status"] = status
                check["last_check"] = time.time()
                results[check["name"]] = {"status": "healthy" if status else "unhealthy"}
            except Exception as e:
                results[check["name"]] = {"status": "error", "error": str(e)}
        
        return results
```

## Testing and Validation

### Unit Testing Template
```python
import unittest
from unittest.mock import Mock, patch
from your_tool import YourTool, ToolInput

class TestYourTool(unittest.TestCase):
    def setUp(self):
        self.tool = YourTool()
        self.valid_input = ToolInput(parameter1="test", parameter2=42)
    
    def test_successful_execution(self):
        result = self.tool.execute(self.valid_input)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result)
    
    def test_invalid_input_handling(self):
        invalid_input = ToolInput(parameter1="", parameter2=-1)
        result = self.tool.execute(invalid_input)
        self.assertFalse(result.success)
        self.assertIn("Error", result.message)
    
    @patch('your_tool.external_api_call')
    def test_external_dependency_failure(self, mock_api):
        mock_api.side_effect = Exception("API Error")
        result = self.tool.execute(self.valid_input)
        self.assertFalse(result.success)
```

### Integration Testing
```python
class IntegrationTestSuite:
    def __init__(self, agent_system):
        self.agent = agent_system
        self.test_scenarios = []
    
    def add_test_scenario(self, name: str, input_data: Dict, expected_outcome: Dict):
        self.test_scenarios.append({
            "name": name,
            "input": input_data,
            "expected": expected_outcome
        })
    
    def run_integration_tests(self) -> Dict[str, Any]:
        results = []
        for scenario in self.test_scenarios:
            try:
                result = self.agent.execute_with_tools(scenario["input"])
                success = self._validate_result(result, scenario["expected"])
                results.append({
                    "scenario": scenario["name"],
                    "success": success,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "scenario": scenario["name"],
                    "success": False,
                    "error": str(e)
                })
        
        return {"test_results": results, "success_rate": self._calculate_success_rate(results)}
```

## Common Patterns and Examples

### 1. API Integration Tool
```python
import requests
from typing import Dict, Any

class APIIntegrationTool:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            return {
                "success": True,
                "data": response.json(),
                "status_code": response.status_code
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "status_code": getattr(e.response, 'status_code', None)
            }
```

### 2. Data Processing Tool
```python
import pandas as pd
from typing import List, Dict, Any

class DataProcessingTool:
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'excel']
    
    def process_data(self, data: Any, operations: List[Dict]) -> Dict:
        try:
            # Convert to DataFrame
            if isinstance(data, str):  # File path
                df = self._load_data_file(data)
            elif isinstance(data, list):  # List of records
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported data format")
            
            # Apply operations
            for operation in operations:
                df = self._apply_operation(df, operation)
            
            return {
                "success": True,
                "result": df.to_dict('records'),
                "shape": df.shape,
                "columns": list(df.columns)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _apply_operation(self, df: pd.DataFrame, operation: Dict) -> pd.DataFrame:
        op_type = operation.get('type')
        
        if op_type == 'filter':
            return df.query(operation['condition'])
        elif op_type == 'sort':
            return df.sort_values(operation['column'], ascending=operation.get('ascending', True))
        elif op_type == 'group':
            return df.groupby(operation['column']).agg(operation['aggregation']).reset_index()
        else:
            raise ValueError(f"Unsupported operation: {op_type}")
```

### 3. File System Tool
```python
import os
import shutil
from pathlib import Path
from typing import List, Dict, Union

class FileSystemTool:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
    
    def list_files(self, directory: str = "", pattern: str = "*") -> Dict:
        try:
            target_path = self.base_path / directory
            if not target_path.exists():
                return {"success": False, "error": "Directory not found"}
            
            files = []
            for file_path in target_path.glob(pattern):
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.base_path)),
                    "size": file_path.stat().st_size if file_path.is_file() else None,
                    "is_directory": file_path.is_dir(),
                    "modified": file_path.stat().st_mtime
                })
            
            return {"success": True, "files": files}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def read_file(self, file_path: str, encoding: str = "utf-8") -> Dict:
        try:
            full_path = self.base_path / file_path
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return {
                "success": True,
                "content": content,
                "size": len(content)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
```

## Best Practices Checklist

### Design Phase
- [ ] Tool has single, clear responsibility
- [ ] Input/output schemas are well-defined
- [