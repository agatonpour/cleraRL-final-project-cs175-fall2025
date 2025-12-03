"""
Base interfaces for memory management (Dependency Inversion Principle)
Allows for different implementations and easy testing with mocks
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConversationExperience:
    """
    Represents a single conversation experience in the memory system.
    Immutable data class for type safety and clarity.
    """
    id: Optional[UUID]
    user_id: str
    thread_id: str
    query_text: str
    agent_response: str
    agent_type: str
    context_snapshot: Dict
    feedback_score: int = 0
    outcome_score: Optional[float] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate experience data"""
        if self.feedback_score not in [-1, 0, 1]:
            raise ValueError(f"feedback_score must be -1, 0, or 1, got {self.feedback_score}")
        if self.agent_type not in ['financial_analyst', 'portfolio_manager', 'trade_executor', 'supervisor']:
            raise ValueError(f"Invalid agent_type: {self.agent_type}")


@dataclass
class SimilarExperience:
    """
    Represents a retrieved experience with similarity score.
    Used for experience replay in RL system.
    """
    experience: ConversationExperience
    similarity: float
    
    def __post_init__(self):
        """Validate similarity score"""
        if not 0 <= self.similarity <= 1:
            raise ValueError(f"similarity must be between 0 and 1, got {self.similarity}")


@dataclass
class MemoryStats:
    """Statistics about a user's memory bank"""
    total_experiences: int
    positive_feedback: int
    negative_feedback: int
    neutral_feedback: int
    average_feedback: float
    experiences_by_agent: Dict[str, Dict]


class IEmbeddingProvider(ABC):
    """
    Interface for embedding generation (Strategy Pattern)
    Allows swapping between different embedding models
    """
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return dimensionality of embeddings"""
        pass


class IMemoryStore(ABC):
    """
    Interface for memory storage backend (Repository Pattern)
    Allows swapping between Supabase, PostgreSQL, or other stores
    """
    
    @abstractmethod
    def store_experience(self, experience: ConversationExperience, embedding: List[float]) -> UUID:
        """Store a conversation experience with its embedding"""
        pass
    
    @abstractmethod
    def retrieve_similar(
        self,
        query_embedding: List[float],
        user_id: str,
        agent_type: Optional[str],
        top_k: int,
        min_feedback_score: int,
        similarity_threshold: float
    ) -> List[SimilarExperience]:
        """Retrieve similar experiences using vector similarity"""
        pass
    
    @abstractmethod
    def update_feedback(self, experience_id: UUID, feedback_score: int):
        """Update feedback score for an experience"""
        pass
    
    @abstractmethod
    def get_stats(self, user_id: str) -> MemoryStats:
        """Get memory statistics for a user"""
        pass
    
    @abstractmethod
    def log_retrieval(
        self,
        query_id: UUID,
        retrieved_ids: List[UUID],
        scores: List[float],
        ranks: List[int]
    ):
        """Log a memory retrieval for evaluation"""
        pass


class IMemoryManager(ABC):
    """
    High-level interface for memory management operations
    This is what agents will interact with
    """
    
    @abstractmethod
    def store_interaction(
        self,
        user_id: str,
        thread_id: str,
        query_text: str,
        agent_response: str,
        agent_type: str,
        context_snapshot: Dict = None
    ) -> UUID:
        """Store a new conversation interaction"""
        pass
    
    @abstractmethod
    def retrieve_relevant_memories(
        self,
        query_text: str,
        user_id: str,
        agent_type: Optional[str] = None,
        top_k: int = 3,
        min_feedback_score: int = 0
    ) -> List[SimilarExperience]:
        """Retrieve relevant past experiences for a query (experience replay)"""
        pass
    
    @abstractmethod
    def record_feedback(self, experience_id: UUID, is_positive: bool):
        """Record user feedback on an experience (reward signal)"""
        pass
    
    @abstractmethod
    def get_user_stats(self, user_id: str) -> MemoryStats:
        """Get statistics about user's memory bank"""
        pass

