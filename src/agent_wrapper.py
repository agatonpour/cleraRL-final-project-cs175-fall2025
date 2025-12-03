"""
Memory-Augmented Agent Wrapper for RL Integration
Wraps existing LangGraph agents with memory retrieval and storage
This implements the BEHAVIORAL CLONING aspect of the RL system
"""

import logging
from typing import Dict, List, Any, Callable
from uuid import UUID
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.types import RunnableConfig
from clera_agents.reinforcement_learning.memory_manager import ConversationMemoryManager, create_memory_manager

logger = logging.getLogger(__name__)


class MemoryAugmentedAgentWrapper:
    """
    Wraps a LangGraph agent with memory retrieval and storage capabilities.
    
    This implements the Decorator Pattern, adding RL functionality without
    modifying the original agent.
    
    The wrapper performs:
    1. EXPERIENCE RETRIEVAL: Fetch similar past successful interactions
    2. CONTEXT INJECTION: Add retrieved memories to agent context
    3. EXPERIENCE STORAGE: Save new interactions for future learning
    4. RETRIEVAL LOGGING: Track what memories were used (for evaluation)
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_type: str,
        memory_manager: ConversationMemoryManager = None,
        enabled: bool = True
    ):
        """
        Initialize wrapper.
        
        Args:
            agent_name: Display name of the agent
            agent_type: Type identifier ('financial_analyst', 'portfolio_manager', etc.)
            memory_manager: Memory manager instance (defaults to singleton)
            enabled: Whether memory augmentation is enabled (allows easy toggle)
        """
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.memory_manager = memory_manager or create_memory_manager()
        self.enabled = enabled
        logger.info(f"Initialized MemoryAugmentedAgentWrapper for {agent_name} (enabled={enabled})")
    
    def wrap(self, agent_executor: Callable) -> Callable:
        """
        Wrap an agent executor with memory augmentation.
        
        Args:
            agent_executor: The original agent executor function
            
        Returns:
            Wrapped executor function with memory capabilities
        """
        def memory_augmented_executor(state: Dict[str, Any], config: RunnableConfig = None) -> Dict[str, Any]:
            """
            Execute agent with memory augmentation.
            
            Flow:
            1. Extract user query from state
            2. Retrieve similar successful past experiences (EXPERIENCE REPLAY)
            3. Inject memories into agent context
            4. Execute original agent
            5. Store new experience
            6. Log retrieval for evaluation
            """
            # Skip if memory not enabled or no config
            if not self.enabled or not config:
                return agent_executor(state, config)
            
            try:
                # Extract user context from config
                user_id = self._get_user_id(config)
                thread_id = self._get_thread_id(config)
                
                if not user_id or not thread_id:
                    logger.debug(f"No user/thread context for {self.agent_name}, skipping memory")
                    return agent_executor(state, config)
                
                # Extract latest user query
                latest_query = self._extract_latest_query(state)
                if not latest_query:
                    logger.debug(f"No query found in state for {self.agent_name}, skipping memory")
                    return agent_executor(state, config)
                
                # STEP 1: EXPERIENCE REPLAY - Retrieve similar successful past interactions
                logger.debug(f"Retrieving memories for {self.agent_name}: {latest_query[:50]}...")
                similar_experiences = self.memory_manager.retrieve_relevant_memories(
                    query_text=latest_query,
                    user_id=user_id,
                    agent_type=self.agent_type,
                    top_k=3,
                    min_feedback_score=0  # Retrieve neutral or positive experiences only
                )
                
                # STEP 2: CONTEXT INJECTION - Add memories to agent prompt
                if similar_experiences:
                    memory_context = self._format_memory_context(similar_experiences)
                    state = self._inject_memory_context(state, memory_context)
                    logger.info(f"Injected {len(similar_experiences)} memories into {self.agent_name}")
                
                # STEP 3: Execute original agent
                result = agent_executor(state, config)
                
                # STEP 4: Store new experience for future learning
                agent_response = self._extract_agent_response(result)
                if agent_response:
                    experience_id = self.memory_manager.store_interaction(
                        user_id=user_id,
                        thread_id=thread_id,
                        query_text=latest_query,
                        agent_response=agent_response,
                        agent_type=self.agent_type,
                        context_snapshot=self._create_context_snapshot(state, config)
                    )
                    
                    # STEP 5: Log retrieval for evaluation
                    if similar_experiences:
                        self.memory_manager.log_memory_retrieval(experience_id, similar_experiences)
                
                return result
                
            except Exception as e:
                # Never fail - memory is enhancement, not requirement
                logger.error(f"Error in memory augmentation for {self.agent_name}: {e}")
                return agent_executor(state, config)
        
        return memory_augmented_executor
    
    def _get_user_id(self, config: RunnableConfig) -> str:
        """Extract user ID from config"""
        if not config or not isinstance(config, dict):
            return None
        configurable = config.get('configurable', {})
        return configurable.get('user_id')
    
    def _get_thread_id(self, config: RunnableConfig) -> str:
        """Extract thread ID from config"""
        if not config or not isinstance(config, dict):
            return None
        configurable = config.get('configurable', {})
        return configurable.get('thread_id') or configurable.get('checkpoint_id', 'unknown')
    
    def _extract_latest_query(self, state: Dict[str, Any]) -> str:
        """Extract the latest user query from state"""
        messages = state.get('messages', [])
        
        # Find last HumanMessage
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        
        return None
    
    def _extract_agent_response(self, result: Dict[str, Any]) -> str:
        """Extract agent response from execution result"""
        if not isinstance(result, dict):
            return None
        
        messages = result.get('messages', [])
        
        # Find last AIMessage
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return None
    
    def _format_memory_context(self, similar_experiences: List) -> str:
        """
        Format retrieved memories into context string for agent.
        
        This implements BEHAVIORAL CLONING - showing the agent
        successful past interactions to mimic.
        """
        context = "\n\n═══════════════════════════════════════════════════════\n"
        context += "📚 RELEVANT PAST SUCCESSFUL INTERACTIONS (Learn from these patterns)\n"
        context += "═══════════════════════════════════════════════════════\n\n"
        
        for i, similar_exp in enumerate(similar_experiences, 1):
            exp = similar_exp.experience
            similarity = similar_exp.similarity
            
            context += f"**Example {i}** (Similarity: {similarity:.2f}, "
            context += f"Feedback: {'👍 Positive' if exp.feedback_score > 0 else '👌 Neutral'})\n\n"
            
            context += f"**User Query:** {exp.query_text}\n\n"
            
            # Truncate long responses
            response_preview = exp.agent_response[:300]
            if len(exp.agent_response) > 300:
                response_preview += "..."
            
            context += f"**Successful Response:** {response_preview}\n\n"
            context += "---\n\n"
        
        context += "Use these examples to inform your response style, level of detail, and approach.\n"
        context += "Adapt the successful patterns to the current query.\n"
        context += "═══════════════════════════════════════════════════════\n"
        
        return context
    
    def _inject_memory_context(self, state: Dict[str, Any], memory_context: str) -> Dict[str, Any]:
        """
        Inject memory context into state.
        
        We add it as a system message so the agent sees it as part of
        the conversation context.
        """
        from langchain_core.messages import SystemMessage
        
        # Create a copy of state to avoid mutation
        new_state = state.copy()
        
        # Add memory context as system message
        memory_msg = SystemMessage(content=memory_context)
        
        if 'messages' in new_state:
            # Insert after the first system message (agent prompt)
            messages = new_state['messages']
            if messages and isinstance(messages[0], SystemMessage):
                new_state['messages'] = [messages[0], memory_msg] + messages[1:]
            else:
                new_state['messages'] = [memory_msg] + messages
        else:
            new_state['messages'] = [memory_msg]
        
        return new_state
    
    def _create_context_snapshot(self, state: Dict[str, Any], config: RunnableConfig) -> Dict:
        """Create snapshot of context for storage"""
        return {
            'state_keys': list(state.keys()),
            'message_count': len(state.get('messages', [])),
            'agent_name': self.agent_name,
            'agent_type': self.agent_type
        }


def create_memory_augmented_agent(
    agent_executor: Callable,
    agent_name: str,
    agent_type: str,
    enabled: bool = True
) -> Callable:
    """
    Factory function to create a memory-augmented agent.
    
    Usage:
        financial_analyst_agent = create_memory_augmented_agent(
            financial_analyst_agent,
            "Financial Analyst",
            "financial_analyst"
        )
    
    Args:
        agent_executor: The agent executor to wrap
        agent_name: Display name
        agent_type: Type identifier
        enabled: Whether to enable memory (defaults to True)
        
    Returns:
        Wrapped agent executor with memory capabilities
    """
    wrapper = MemoryAugmentedAgentWrapper(agent_name, agent_type, enabled=enabled)
    return wrapper.wrap(agent_executor)

