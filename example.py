"""
MetisOS Agent Framework - Example Usage

This example demonstrates the basic usage of the MetisOS Agent Framework.
"""

from agent import SingleAgent


def main():
    print("=" * 60)
    print("MetisOS Agent Framework - Example Usage")
    print("=" * 60)
    print()

    # Example 1: Initialize Agent
    print("Example 1: Initializing Agent")
    print("-" * 60)

    agent = SingleAgent(
        user_id="example_user",
        enable_titans_memory=True  # Enable adaptive memory
    )

    # Get agent identity
    identity = agent.get_agent_identity()
    print(f"Agent Name: {identity['name']}")
    print(f"Agent ID: {identity['id']}")
    print(f"Version: {identity['version']}")
    print(f"Personality Traits: {', '.join(identity['personality'])}")
    print()

    # Example 2: Simple Question
    print("Example 2: Asking a Simple Question")
    print("-" * 60)

    response = agent.process_query("What is machine learning?")
    print(f"Response: {response}")
    print()

    # Example 3: Complex Task with Planning
    print("Example 3: Complex Task Execution")
    print("-" * 60)

    response = agent.process_query(
        "Create a Python function that calculates factorial with memoization"
    )

    if isinstance(response, dict):
        print(f"Task Summary: {response.get('summary', 'N/A')}")

        if 'steps' in response:
            print(f"\nExecution Steps:")
            for i, step in enumerate(response['steps'], 1):
                print(f"  {i}. {step.get('description', step.get('task', 'N/A'))[:80]}...")
    else:
        print(f"Response: {response}")
    print()

    # Example 4: Session-based Conversation
    print("Example 4: Session-based Conversation with Memory")
    print("-" * 60)

    # First query in a session
    response1 = agent.process_query(
        "Tell me about neural networks",
        session_id="learning_session"
    )
    print(f"Response 1: {str(response1)[:150]}...")
    print()

    # Follow-up query in the same session (agent remembers context)
    response2 = agent.process_query(
        "Can you give me a practical example?",
        session_id="learning_session"
    )
    print(f"Response 2 (with context): {str(response2)[:150]}...")
    print()

    # Example 5: Memory Insights
    print("Example 5: Memory System Insights")
    print("-" * 60)

    insights = agent.get_memory_insights()
    print(f"Standard Memory Enabled: {insights['standard_memory']['enabled']}")
    print(f"Adaptive Memory Enabled: {insights['adaptive_memory']['enabled']}")

    if insights['adaptive_memory']['enabled'] and insights['adaptive_memory']['insights']:
        titans_insights = insights['adaptive_memory']['insights']
        if 'health_indicators' in titans_insights:
            health = titans_insights['health_indicators']
            print(f"Memory Utilization: {health.get('memory_utilization', 0):.2%}")
            print(f"Learning Active: {health.get('learning_active', False)}")
    print()

    # Example 6: Configure Agent Behavior
    print("Example 6: Configuring Agent")
    print("-" * 60)

    # Configure evaluation
    agent.configure_evaluation(
        enabled=True,
        auto_improve=True,
        threshold=7.0
    )
    print("Evaluation configured: enabled=True, auto_improve=True, threshold=7.0")

    # Configure tracing
    agent.configure_tracing(
        enabled=True,
        trace_tools=True,
        trace_evaluations=True
    )
    print("Tracing configured: enabled=True")
    print()

    # Example 7: Production Monitoring
    print("Example 7: Production Metrics")
    print("-" * 60)

    try:
        metrics = agent.get_production_metrics()
        print(f"System Health: {metrics['system_health']}")
        print(f"Performance Score: {metrics['performance_score']:.2f}/100")
        print(f"Uptime: {metrics['uptime_hours']:.2f} hours")
    except Exception as e:
        print(f"Production metrics not available: {e}")
    print()

    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
