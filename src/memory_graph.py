"""
Memory-Enabled Graph Builder
Import this instead of direct graph.py to get memory-augmented agents
This maintains clean separation of concerns
"""

import logging
from clera_agents import graph as base_graph
from clera_agents.reinforcement_learning.agent_wrapper import create_memory_augmented_agent

logger = logging.getLogger(__name__)

# Re-export the base graph components
from clera_agents.graph import State, checkpointer, workflow, main_llm, current_datetime

# Import the original agent executors BEFORE wrapping
from clera_agents.graph import (
    financial_analyst_agent as base_financial_analyst_agent,
    portfolio_management_agent as base_portfolio_management_agent,
    trade_execution_agent as base_trade_execution_agent
)

# Wrap agents with memory augmentation
# This is a non-intrusive way to add RL without modifying graph.py
financial_analyst_agent = create_memory_augmented_agent(
    base_financial_analyst_agent,
    agent_name="Financial Analyst",
    agent_type="financial_analyst",
    enabled=True  # Set to False to disable memory for testing
)

portfolio_management_agent = create_memory_augmented_agent(
    base_portfolio_management_agent,
    agent_name="Portfolio Manager",
    agent_type="portfolio_manager",
    enabled=True
)

trade_execution_agent = create_memory_augmented_agent(
    base_trade_execution_agent,
    agent_name="Trade Executor",
    agent_type="trade_executor",
    enabled=True
)

# Create memory-augmented graph
# We rebuild the supervisor with wrapped agents
from langgraph_supervisor import create_supervisor
from utils.personalization_service import create_personalized_supervisor_prompt

memory_workflow = create_supervisor(
    [financial_analyst_agent, portfolio_management_agent, trade_execution_agent],
    model=main_llm,
    prompt=create_personalized_supervisor_prompt,
    output_mode="full_history",
    supervisor_name="Clera",
    state_schema=State
)

# Compile memory-augmented graph
memory_graph = memory_workflow.compile()
memory_graph.name = "CleraAgentsWithMemory"

logger.info("Memory-augmented graph initialized successfully")

# Export both for flexibility
__all__ = [
    'memory_graph',  # Use this for RL-enabled agents
    'financial_analyst_agent',
    'portfolio_management_agent',
    'trade_execution_agent',
    'State',
    'checkpointer',
    'workflow',
    'memory_workflow'
]

