"""
Supabase Memory Store Implementation
Handles all database operations for conversation experiences
"""

import logging
from typing import List, Dict, Optional
from uuid import UUID
from datetime import datetime
from clera_agents.reinforcement_learning import (
    IMemoryStore,
    ConversationExperience,
    SimilarExperience,
    MemoryStats
)
from utils.supabase.db_client import get_supabase_client

logger = logging.getLogger(__name__)


class SupabaseMemoryStore(IMemoryStore):
    """
    Concrete implementation using Supabase/PostgreSQL.
    This is a Repository Pattern implementation for data access.
    """
    
    EXPERIENCES_TABLE = "conversation_experiences"
    RETRIEVAL_LOG_TABLE = "memory_retrieval_log"
    
    def __init__(self):
        """Initialize Supabase client"""
        try:
            self.supabase = get_supabase_client()
            logger.info("Initialized SupabaseMemoryStore")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def store_experience(self, experience: ConversationExperience, embedding: List[float]) -> UUID:
        """
        Store a conversation experience with its embedding.
        
        Args:
            experience: The conversation experience to store
            embedding: Vector embedding of the query text
            
        Returns:
            UUID of the stored experience
            
        Raises:
            Exception: If database operation fails
        """
        try:
            data = {
                'user_id': experience.user_id,
                'thread_id': experience.thread_id,
                'query_text': experience.query_text,
                'query_embedding': embedding,
                'agent_response': experience.agent_response,
                'agent_type': experience.agent_type,
                'context_snapshot': experience.context_snapshot,
                'feedback_score': experience.feedback_score
            }
            
            if experience.outcome_score is not None:
                data['outcome_score'] = experience.outcome_score
            
            result = self.supabase.table(self.EXPERIENCES_TABLE).insert(data).execute()
            
            if not result.data or len(result.data) == 0:
                raise Exception("Failed to insert experience: no data returned")
            
            experience_id = UUID(result.data[0]['id'])
            logger.info(f"Stored experience {experience_id} for user {experience.user_id}, agent {experience.agent_type}")
            return experience_id
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            raise
    
    def retrieve_similar(
        self,
        query_embedding: List[float],
        user_id: str,
        agent_type: Optional[str],
        top_k: int,
        min_feedback_score: int,
        similarity_threshold: float
    ) -> List[SimilarExperience]:
        """
        Retrieve similar experiences using vector similarity search.
        This implements the experience replay mechanism for RL.
        
        Args:
            query_embedding: Vector embedding of the query
            user_id: User ID to filter experiences
            agent_type: Optional agent type filter
            top_k: Number of similar experiences to retrieve
            min_feedback_score: Minimum feedback score to consider (reward threshold)
            similarity_threshold: Minimum cosine similarity threshold
            
        Returns:
            List of SimilarExperience objects, ordered by reward then similarity
        """
        try:
            # Call the stored procedure for efficient search
            rpc_params = {
                'query_embedding': query_embedding,
                'match_threshold': similarity_threshold,
                'match_count': top_k * 2,  # Get more, then filter
                'target_user_id': user_id,
                'target_agent_type': agent_type,
                'min_feedback_score': min_feedback_score
            }
            
            result = self.supabase.rpc('search_similar_experiences', rpc_params).execute()
            
            if not result.data:
                logger.debug(f"No similar experiences found for user {user_id}")
                return []
            
            # Convert to SimilarExperience objects
            similar_experiences = []
            for row in result.data[:top_k]:  # Take only top_k
                exp = ConversationExperience(
                    id=UUID(row['id']),
                    user_id=user_id,  # We know this from the query
                    thread_id="",  # Not needed for retrieval
                    query_text=row['query_text'],
                    agent_response=row['agent_response'],
                    agent_type=row['agent_type'],
                    context_snapshot=row.get('context_snapshot', {}),
                    feedback_score=row['feedback_score'],
                    created_at=row.get('created_at')
                )
                
                similar_experiences.append(SimilarExperience(
                    experience=exp,
                    similarity=float(row['similarity'])
                ))
            
            logger.debug(f"Retrieved {len(similar_experiences)} similar experiences for user {user_id}")
            return similar_experiences
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar experiences: {e}")
            raise
    
    def update_feedback(self, experience_id: UUID, feedback_score: int):
        """
        Update feedback score for an experience (reward signal).
        
        Args:
            experience_id: ID of the experience to update
            feedback_score: New feedback score (-1, 0, or 1)
            
        Raises:
            ValueError: If feedback_score is invalid
            Exception: If database operation fails
        """
        if feedback_score not in [-1, 0, 1]:
            raise ValueError(f"feedback_score must be -1, 0, or 1, got {feedback_score}")
        
        try:
            result = self.supabase.table(self.EXPERIENCES_TABLE).update({
                'feedback_score': feedback_score,
                'updated_at': datetime.utcnow().isoformat()
            }).eq('id', str(experience_id)).execute()
            
            if not result.data or len(result.data) == 0:
                logger.warning(f"No experience found with ID {experience_id}")
            else:
                logger.info(f"Updated feedback score for experience {experience_id}: {feedback_score}")
                
        except Exception as e:
            logger.error(f"Failed to update feedback: {e}")
            raise
    
    def get_stats(self, user_id: str) -> MemoryStats:
        """
        Get memory statistics for a user.
        
        Args:
            user_id: User ID to get stats for
            
        Returns:
            MemoryStats object with comprehensive statistics
        """
        try:
            # Try the stored procedure first
            try:
                result = self.supabase.rpc('get_user_memory_stats', {'target_user_id': user_id}).execute()
                
                if result.data and len(result.data) > 0:
                    data = result.data[0]
                    return MemoryStats(
                        total_experiences=data['total_experiences'],
                        positive_feedback=data['positive_feedback'],
                        negative_feedback=data['negative_feedback'],
                        neutral_feedback=data['neutral_feedback'],
                        average_feedback=float(data['average_feedback']) if data['average_feedback'] else 0.0,
                        experiences_by_agent=data.get('experiences_by_agent', {})
                    )
            except Exception as e:
                logger.warning(f"Stored procedure failed, using direct query: {e}")
                # Fallback to direct query
                pass
            
            # Direct query fallback
            result = self.supabase.table(self.EXPERIENCES_TABLE)\
                .select('*')\
                .eq('user_id', user_id)\
                .execute()
            
            experiences = result.data
            
            if not experiences:
                return MemoryStats(
                    total_experiences=0,
                    positive_feedback=0,
                    negative_feedback=0,
                    neutral_feedback=0,
                    average_feedback=0.0,
                    experiences_by_agent={}
                )
            
            # Calculate stats manually
            total = len(experiences)
            positive = sum(1 for e in experiences if e.get('feedback_score', 0) > 0)
            negative = sum(1 for e in experiences if e.get('feedback_score', 0) < 0)
            neutral = sum(1 for e in experiences if e.get('feedback_score', 0) == 0)
            avg_feedback = sum(e.get('feedback_score', 0) for e in experiences) / total if total > 0 else 0.0
            
            # Calculate by agent
            by_agent = {}
            for exp in experiences:
                agent = exp.get('agent_type', 'unknown')
                if agent not in by_agent:
                    by_agent[agent] = {'count': 0, 'total_feedback': 0}
                by_agent[agent]['count'] += 1
                by_agent[agent]['total_feedback'] += exp.get('feedback_score', 0)
            
            # Calculate averages
            experiences_by_agent = {}
            for agent, stats in by_agent.items():
                experiences_by_agent[agent] = {
                    'count': stats['count'],
                    'avg_feedback': stats['total_feedback'] / stats['count'] if stats['count'] > 0 else 0
                }
            
            return MemoryStats(
                total_experiences=total,
                positive_feedback=positive,
                negative_feedback=negative,
                neutral_feedback=neutral,
                average_feedback=avg_feedback,
                experiences_by_agent=experiences_by_agent
            )
            
        except Exception as e:
            logger.error(f"Failed to get user stats: {e}")
            raise
    
    def log_retrieval(
        self,
        query_id: UUID,
        retrieved_ids: List[UUID],
        scores: List[float],
        ranks: List[int]
    ):
        """
        Log a memory retrieval for evaluation.
        
        Args:
            query_id: ID of the query experience
            retrieved_ids: IDs of retrieved experiences
            scores: Similarity scores of retrieved experiences
            ranks: Ranking positions of retrieved experiences
        """
        try:
            data = {
                'query_id': str(query_id),
                'retrieved_experience_ids': [str(id) for id in retrieved_ids],
                'retrieval_scores': scores,
                'retrieval_rank': ranks
            }
            
            self.supabase.table(self.RETRIEVAL_LOG_TABLE).insert(data).execute()
            logger.debug(f"Logged retrieval for query {query_id}")
            
        except Exception as e:
            # Log but don't fail - retrieval logging is for evaluation only
            logger.warning(f"Failed to log retrieval: {e}")

