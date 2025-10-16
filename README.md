# MetisOS Agent Framework

An advanced agentic AI system for intelligent query processing and complex task execution.

## Overview

MetisOS is a production-ready AI agent framework that combines natural language understanding, strategic planning, and specialized tool orchestration to deliver powerful task automation capabilities. Built with a sophisticated multi-component architecture, it provides:

- **Intelligent Intent Classification**: Automatically distinguishes between questions and tasks
- **Advanced Task Planning**: Creates DAG-based execution plans for complex requests
- **Adaptive Memory System**: SQLite + Titans adaptive memory for context retention
- **Multi-Tool Orchestration**: Dynamically selects and chains appropriate tools
- **Production Monitoring**: Built-in tracing, evaluation, and health monitoring

## Architecture

### Core Components

- **SingleAgent** (`agent.py`): Central orchestrator managing identity, memory, and workflow
- **Intent Router** (`components/intent_router.py`): Classifies user input as question or task
- **Planner** (`components/planner.py`): Creates structured subtask decomposition (DAG)
- **Task Manager** (`components/task_manager.py`): Manages task lifecycle and state
- **Tool Selector** (`components/tool_selector.py`): Dynamically selects appropriate tools
- **Memory Systems** (`memory/`): SQLite + Titans adaptive memory integration
- **Response Evaluator** (`components/response_evaluator.py`): Quality assessment and auto-improvement
- **Tracer** (`components/tracer.py`): Comprehensive execution tracing

### Specialized Tools

Located in `tools/` directory:
- **Code Generation**: Multi-language code creation with game dev specialization
- **Content Generation**: Research papers and written content
- **E2B Tool**: Secure code execution in cloud sandboxes
- **Search Tools**: Google Search and Firecrawl web scraping
- **Email Tools**: Gmail integration for email management
- **Math Tool**: Advanced mathematical computations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- API Keys (optional, for enhanced functionality):
  - Groq API for LLM access
  - E2B API for code execution
  - Google API for search
  - Firecrawl API for web scraping

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/metisos/Metis-Agents-Framework-Open-Source.git
cd Metis-Agents-Framework-Open-Source
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Run the agent**:
```python
from agent import SingleAgent

# Initialize agent (all parameters are optional)
agent = SingleAgent()

# Process a query
response = agent.process_query("Write a Python function to calculate Fibonacci numbers")
print(response)
```

## Usage Examples

### Basic Question Answering

```python
from agent import SingleAgent

agent = SingleAgent()
response = agent.process_query("What is machine learning?")
print(response)
```

### Complex Task Execution

```python
from agent import SingleAgent

agent = SingleAgent(enable_titans_memory=True)

# The agent will plan, decompose, and execute the task
response = agent.process_query(
    "Create a Python snake game with score tracking and difficulty levels"
)

# Response includes task breakdown and results
print(response['summary'])
print(response['steps'])
```

### With Memory Context

```python
from agent import SingleAgent

agent = SingleAgent(enable_titans_memory=True)

# First interaction
response1 = agent.process_query(
    "Explain the concept of neural networks",
    session_id="learning_session"
)

# Follow-up with context
response2 = agent.process_query(
    "Can you give me an example?",
    session_id="learning_session"
)
```

### Production Deployment

```python
from agent import deploy_production_agent, health_check_endpoint

# Deploy with monitoring
agent = deploy_production_agent(user_id="production")

# Health check
health = health_check_endpoint(agent)
print(f"Status: {health['status']}")
print(f"Performance Score: {health['performance_score']}/100")

# Get production metrics
metrics = agent.get_production_metrics()
print(f"Uptime: {metrics['uptime_hours']} hours")
```

## Configuration

### Environment Variables

Create a `.env` file with the following:

```bash
# LLM Provider
GROQ_API_KEY=your_groq_api_key_here

# Code Execution (optional)
E2B_API_KEY=your_e2b_api_key_here

# Search (optional)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_search_engine_id

# Web Scraping (optional)
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

### Agent Configuration

```python
from agent import SingleAgent

# Configure agent behavior
agent = SingleAgent(
    user_id="custom_user",
    enable_titans_memory=True  # Enable adaptive memory
)

# Configure evaluation
agent.configure_evaluation(
    enabled=True,
    auto_improve=True,
    threshold=7.0
)

# Configure tracing
agent.configure_tracing(
    enabled=True,
    trace_tools=True,
    trace_evaluations=True
)

# Configure adaptive memory
agent.configure_adaptive_memory(
    surprise_threshold=0.6,
    chunk_size=4
)
```

## Features

### Intelligent Query Processing
- Multi-layer intent classification
- Context-aware query understanding
- Automatic clarification requests when needed

### Advanced Memory Systems
- **SQLite Memory**: Persistent conversation storage
- **Titans Adaptive Memory**: Learning-based context retention
- **Surprise-based Learning**: Automatically adapts to novel information

### Task Execution
- DAG-based task planning
- Parallel and sequential execution strategies
- Automatic tool selection and chaining
- Error handling and recovery

### Quality Assurance
- Automatic response evaluation
- Self-improvement attempts
- Configurable quality thresholds
- Comprehensive evaluation metrics

### Production Features
- Real-time health monitoring
- Performance scoring
- Auto-save memory states
- Production metrics and analytics
- Execution tracing and debugging

## API Reference

### SingleAgent Class

```python
SingleAgent(
    user_id="default_user",      # Optional: identifier for memory storage (defaults to "default_user")
    enable_titans_memory=True    # Optional: enable adaptive memory system (defaults to True)
)
```

**Parameters:**
- `user_id` (optional): String identifier for separating user data in multi-user applications. Defaults to `"default_user"`. Only needed if you're building a system with multiple users who need separate conversation histories.
- `enable_titans_memory` (optional): Boolean to enable/disable the adaptive memory system. Defaults to `True`.

**Methods:**
- `process_query(query, session_id=None)`: Process user query
- `get_agent_identity()`: Get agent identity information
- `get_memory_insights()`: Get memory system insights
- `configure_evaluation(...)`: Configure response evaluation
- `configure_tracing(...)`: Configure execution tracing
- `get_production_metrics()`: Get comprehensive metrics

## Development

### Adding Custom Tools

1. Create a new tool in `tools/` directory:

```python
class CustomTool:
    def can_handle(self, task):
        # Determine if tool can handle the task
        return "custom_keyword" in task.lower()

    def execute(self, task):
        # Execute the task
        result = perform_custom_operation(task)
        return {
            "status": "success",
            "result": result
        }
```

2. Register in `components/tool_selector.py`

### Customizing Agent Behavior

- **System Message**: Modify in `components/llm_interface.py`
- **Personality Traits**: Adjust in `agent.py` constructor
- **Task Routing**: Configure in `components/intent_router.py`

## Testing

Run basic tests:

```python
from agent import SingleAgent

# Initialize agent
agent = SingleAgent()

# Test question handling
q_response = agent.process_query("What is 2 + 2?")
assert q_response is not None

# Test task handling
t_response = agent.process_query("Create a hello world function in Python")
assert t_response.get('summary') is not None

print("All tests passed!")
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

Please report security vulnerabilities privately following our [Security Policy](SECURITY.md).

## Support

- **Documentation**: Full guides in this README
- **Issues**: Report bugs via [GitHub Issues](https://github.com/metisos/Metis-Agents-Framework-Open-Source/issues)
- **Discussions**: Community support via [GitHub Discussions](https://github.com/metisos/Metis-Agents-Framework-Open-Source/discussions)

## Acknowledgments

Built with contributions from the open source community and powered by:
- Groq for LLM inference
- E2B for secure code execution
- SQLite for data persistence

## Project Status

**Version**: 2.0.0
**Status**: Production Ready
**Python**: 3.8+

---

**MetisOS** - Intelligent AI Agent Orchestration
