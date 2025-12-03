"""
Feedback API Routes for RL System
Handles thumbs up/down feedback and memory statistics
This is the reward signal interface for the RL system
"""

import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from uuid import UUID
from typing import Optional
from clera_agents.reinforcement_learning.memory_manager import create_memory_manager, ConversationMemoryManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rl", tags=["Reinforcement Learning"])

# Dependency injection for memory manager (testable)
def get_memory_manager() -> ConversationMemoryManager:
    """Dependency injection for memory manager"""
    return create_memory_manager()


# ============================================================================
# Request/Response Models
# ============================================================================

class FeedbackRequest(BaseModel):
    """Request model for submitting conversation feedback"""
    experience_id: UUID = Field(..., description="ID of the conversation experience")
    feedback: str = Field(..., description="Feedback type: 'thumbs_up' or 'thumbs_down'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "experience_id": "550e8400-e29b-41d4-a716-446655440000",
                "feedback": "thumbs_up"
            }
        }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool
    message: str
    experience_id: UUID


class MemoryStatsResponse(BaseModel):
    """Response model for memory statistics"""
    total_experiences: int
    positive_feedback: int
    negative_feedback: int
    neutral_feedback: int
    average_feedback: float
    experiences_by_agent: dict
    learning_rate: float  # Positive / Total (0-1)


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    rl_system_enabled: bool
    message: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/feedback", response_model=FeedbackResponse)
async def submit_conversation_feedback(
    request: FeedbackRequest,
    memory_manager: ConversationMemoryManager = Depends(get_memory_manager)
):
    """
    Submit thumbs up/down feedback for a conversation.
    
    This provides the **immediate reward signal** in the RL system.
    Feedback is stored and used to weight memory retrieval.
    
    Args:
        request: Feedback submission with experience ID and feedback type
        
    Returns:
        Confirmation of feedback recording
        
    Raises:
        HTTPException 400: If feedback type is invalid
        HTTPException 404: If experience ID not found
        HTTPException 500: If storage fails
    """
    # Validate feedback type
    if request.feedback not in ['thumbs_up', 'thumbs_down']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid feedback type: {request.feedback}. Must be 'thumbs_up' or 'thumbs_down'"
        )
    
    try:
        # Record feedback (reward signal)
        is_positive = request.feedback == 'thumbs_up'
        memory_manager.record_feedback(request.experience_id, is_positive)
        
        logger.info(f"Recorded {request.feedback} for experience {request.experience_id}")
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded successfully",
            experience_id=request.experience_id
        )
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")


@router.get("/stats/{user_id}", response_model=MemoryStatsResponse)
async def get_user_memory_stats(
    user_id: str,
    memory_manager: ConversationMemoryManager = Depends(get_memory_manager)
):
    """
    Get memory statistics for a user.
    
    Used for evaluation and monitoring the RL system.
    Shows memory accumulation and learning progress.
    
    Args:
        user_id: User ID to get stats for
        
    Returns:
        Comprehensive memory statistics
        
    Raises:
        HTTPException 500: If stats retrieval fails
    """
    try:
        stats = memory_manager.get_user_stats(user_id)
        
        # Calculate learning rate (positive feedback ratio)
        learning_rate = 0.0
        if stats.total_experiences > 0:
            learning_rate = stats.positive_feedback / stats.total_experiences
        
        logger.info(f"Retrieved stats for user {user_id}: {stats.total_experiences} experiences")
        
        return MemoryStatsResponse(
            total_experiences=stats.total_experiences,
            positive_feedback=stats.positive_feedback,
            negative_feedback=stats.negative_feedback,
            neutral_feedback=stats.neutral_feedback,
            average_feedback=stats.average_feedback,
            experiences_by_agent=stats.experiences_by_agent,
            learning_rate=learning_rate
        )
        
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check for RL system.
    
    Verifies that the RL components are operational.
    
    Returns:
        Health status
    """
    try:
        # Try to create memory manager
        memory_manager = create_memory_manager()
        
        return HealthCheckResponse(
            status="healthy",
            rl_system_enabled=True,
            message="Reinforcement learning system is operational"
        )
        
    except Exception as e:
        logger.error(f"RL system health check failed: {e}")
        return HealthCheckResponse(
            status="degraded",
            rl_system_enabled=False,
            message=f"RL system error: {str(e)}"
        )


# Export router for registration in main API server
__all__ = ['router']

