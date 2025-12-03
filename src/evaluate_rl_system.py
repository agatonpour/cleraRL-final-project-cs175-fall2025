"""
Evaluation Framework for RL System
Measures memory accumulation, retrieval relevance, and behavioral improvement
"""

import logging
import json
from typing import List, Dict
from datetime import datetime
from clera_agents.reinforcement_learning.memory_manager import create_memory_manager, MemoryStats

logger = logging.getLogger(__name__)


class RLSystemEvaluator:
    """
    Comprehensive evaluator for the RL memory system.
    Measures key metrics for the CS 175 project.
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_manager = create_memory_manager()
    
    def evaluate_memory_accumulation(self) -> Dict:
        """
        Metric 1: Memory Accumulation Over Time
        Measures how the memory bank grows as users interact
        """
        stats = self.memory_manager.get_user_stats(self.user_id)
        
        return {
            "metric": "Memory Accumulation",
            "user_id": self.user_id,
            "total_experiences": stats.total_experiences,
            "positive_experiences": stats.positive_feedback,
            "negative_experiences": stats.negative_feedback,
            "neutral_experiences": stats.neutral_feedback,
            "learning_rate": (stats.positive_feedback / stats.total_experiences 
                            if stats.total_experiences > 0 else 0),
            "experiences_by_agent": stats.experiences_by_agent,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def evaluate_retrieval_relevance(self, test_queries: List[str]) -> Dict:
        """
        Metric 2: Retrieval Relevance
        Measures how relevant retrieved memories are to queries
        """
        relevance_scores = []
        
        for query in test_queries:
            try:
                retrieved = self.memory_manager.retrieve_relevant_memories(
                    query_text=query,
                    user_id=self.user_id,
                    top_k=3,
                    min_feedback_score=0
                )
                
                if retrieved:
                    avg_similarity = sum(exp.similarity for exp in retrieved) / len(retrieved)
                    avg_feedback = sum(exp.experience.feedback_score for exp in retrieved) / len(retrieved)
                    
                    relevance_scores.append({
                        "query": query,
                        "num_retrieved": len(retrieved),
                        "avg_similarity": avg_similarity,
                        "avg_feedback": avg_feedback,
                        "top_similarity": retrieved[0].similarity if retrieved else 0
                    })
                else:
                    relevance_scores.append({
                        "query": query,
                        "num_retrieved": 0,
                        "avg_similarity": 0,
                        "avg_feedback": 0,
                        "top_similarity": 0
                    })
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                continue
        
        if not relevance_scores:
            return {
                "metric": "Retrieval Relevance",
                "error": "No queries could be evaluated"
            }
        
        return {
            "metric": "Retrieval Relevance",
            "user_id": self.user_id,
            "num_test_queries": len(test_queries),
            "avg_similarity": sum(r['avg_similarity'] for r in relevance_scores) / len(relevance_scores),
            "avg_retrieved_feedback": sum(r['avg_feedback'] for r in relevance_scores) / len(relevance_scores),
            "avg_num_retrieved": sum(r['num_retrieved'] for r in relevance_scores) / len(relevance_scores),
            "details": relevance_scores,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def evaluate_feedback_distribution(self) -> Dict:
        """
        Metric 3: Feedback Distribution
        Proxy for user satisfaction over time
        """
        stats = self.memory_manager.get_user_stats(self.user_id)
        
        total = stats.total_experiences
        if total == 0:
            return {
                "metric": "Feedback Distribution",
                "error": "No experiences yet",
                "user_id": self.user_id
            }
        
        return {
            "metric": "Feedback Distribution",
            "user_id": self.user_id,
            "total_experiences": total,
            "thumbs_up_rate": stats.positive_feedback / total,
            "thumbs_down_rate": stats.negative_feedback / total,
            "neutral_rate": stats.neutral_feedback / total,
            "average_score": stats.average_feedback,
            "target_thumbs_up_rate": 0.80,  # Project target
            "meets_target": (stats.positive_feedback / total) >= 0.70,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def generate_full_report(self, test_queries: List[str]) -> Dict:
        """
        Generate comprehensive evaluation report
        """
        return {
            "evaluation_report": {
                "user_id": self.user_id,
                "generated_at": datetime.utcnow().isoformat(),
                "metrics": [
                    self.evaluate_memory_accumulation(),
                    self.evaluate_retrieval_relevance(test_queries),
                    self.evaluate_feedback_distribution()
                ]
            }
        }
    
    def print_report(self, test_queries: List[str]):
        """Print human-readable evaluation report"""
        report = self.generate_full_report(test_queries)
        
        print("\n" + "="*80)
        print("RL SYSTEM EVALUATION REPORT")
        print("="*80)
        print(f"User ID: {self.user_id}")
        print(f"Generated: {report['evaluation_report']['generated_at']}")
        print("="*80 + "\n")
        
        for metric in report['evaluation_report']['metrics']:
            print(f"\n{metric['metric'].upper()}")
            print("-"*80)
            
            for key, value in metric.items():
                if key not in ['metric', 'details', 'timestamp', 'experiences_by_agent']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
            
            if 'experiences_by_agent' in metric:
                print(f"\n  Experiences by Agent:")
                for agent, stats in metric['experiences_by_agent'].items():
                    print(f"    {agent}: {stats}")
        
        print("\n" + "="*80 + "\n")
        
        return report


# Test queries for evaluation
DEFAULT_TEST_QUERIES = [
    "Should I invest in NVIDIA?",
    "Should I invest in tech stocks?",
    "What's a good investment right now?",
    "How's my portfolio performing?",
    "Should I rebalance my portfolio?",
    "What stocks do I own?",
    "Buy $500 of Apple stock",
    "Sell $1000 worth of Tesla"
]


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RL system performance")
    parser.add_argument("--user-id", required=True, help="User ID to evaluate")
    parser.add_argument("--output", help="Output file for JSON report (optional)")
    parser.add_argument("--queries", nargs="+", help="Custom test queries (optional)")
    
    args = parser.parse_args()
    
    test_queries = args.queries if args.queries else DEFAULT_TEST_QUERIES
    
    print(f"\nEvaluating RL system for user: {args.user_id}")
    print(f"Test queries: {len(test_queries)}\n")
    
    evaluator = RLSystemEvaluator(args.user_id)
    report = evaluator.print_report(test_queries)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.output}")

