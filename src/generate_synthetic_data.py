"""
Synthetic Conversation Data Generator
Creates realistic conversation examples for testing the RL system
This provides initial training data before real user interactions accumulate
"""

import logging
from uuid import uuid4
from typing import List, Dict
from clera_agents.reinforcement_learning.memory_manager import create_memory_manager

logger = logging.getLogger(__name__)


# ============================================================================
# Synthetic Conversation Examples
# ============================================================================

SYNTHETIC_CONVERSATIONS = [
    # ========== FINANCIAL ANALYST QUERIES ==========
    {
        "agent_type": "financial_analyst",
        "query": "Should I invest in NVIDIA right now?",
        "response": """Based on current market data:

**Current Status**: NVDA is trading at $140.50 (+2.3% today). 
- 52-week range: $112-$180
- P/E ratio: 45.2x (premium to tech sector average of 32x)

**Analyst Consensus**: 
- Strong Buy rating from 28 of 35 analysts
- Average price target: $165 (17% upside)
- Recent upgrades from Goldman Sachs and Morgan Stanley

**Investment Thesis**:
1. **AI Leadership**: NVIDIA dominates the AI chip market with 80%+ market share
2. **Strong Fundamentals**: Q3 earnings beat by 15%, revenue up 94% YoY
3. **Data Center Growth**: Data center segment grew 279% YoY to $14.5B

**Key Risks**:
- High valuation (45x P/E vs. historical average of 38x)
- Competition from AMD and custom chips (Google TPU, Amazon Trainium)
- China export restrictions could impact 20-25% of revenue

**Recommendation**: NVIDIA remains a strong long-term AI play. Consider dollar-cost averaging to manage valuation risk. Wait for pullbacks below $135 for better entry points.""",
        "feedback": 1
    },
    {
        "agent_type": "financial_analyst",
        "query": "What's happening with Tesla stock lately?",
        "response": """Here's the latest on Tesla:

**Recent Performance**: TSLA at $245.30 (-3.2% today, +8% past month)

**Key Developments**:
1. **Q4 Deliveries**: Beat expectations with 484,507 vehicles delivered (vs. 473,000 expected)
2. **Price Cuts**: Reduced Model 3/Y prices in China by 5-8% to compete with BYD
3. **Cybertruck Launch**: Started deliveries, but production ramp slower than expected

**Analyst Views**: Mixed sentiment
- Bull case: Strong delivery growth, energy business momentum, AI/robotics optionality
- Bear case: Margin compression from price cuts, increased competition, valuation concerns

**Recommendation**: Tesla is in a transitional phase. Strong for long-term believers in Musk's vision, but near-term volatility expected. Consider waiting for clarity on margins in the next earnings call (Jan 24).""",
        "feedback": 1
    },
    {
        "agent_type": "financial_analyst",
        "query": "Are bonds a good investment right now with rates so high?",
        "response": """Bonds are becoming more attractive as rates stabilize:

**Current Environment**:
- 10-year Treasury yield: 4.45% (near recent highs)
- Fed has signaled potential rate cuts in 2025
- Inflation trending toward 2% target

**Why Bonds Make Sense Now**:
1. **Attractive Yields**: 4-5% risk-free rates best since 2007
2. **Rate Cut Potential**: If Fed cuts, bond prices rise (inverse relationship)
3. **Portfolio Diversification**: Bonds reduce equity volatility

**Recommendations**:
- **Short-term (<2 years)**: High-quality corporate bonds or Treasury bonds
- **Medium-term (3-7 years)**: Investment-grade corporate bonds (BBB+ or higher)
- **Bond ETFs**: Consider AGG (total market), BND (Vanguard total bond), or TLT (long-term Treasury)

**Strategy**: Lock in today's yields before potential rate cuts. A 60/40 stock/bond allocation is regaining appeal.""",
        "feedback": 1
    },
    {
        "agent_type": "financial_analyst",
        "query": "Should I buy GameStop? I heard it's going to moon!",
        "response": """GameStop (GME) is a highly speculative meme stock. Here's the reality:

**Current Situation**: GME at $18.50
- Down 75% from 2021 peak of $81
- Negative earnings: -$0.14 per share last quarter
- Business struggling: Retail gaming shift to digital

**Investment Case**: Weak
- Company hasn't successfully pivoted from declining physical game sales
- No clear path to profitability
- Price driven by social media hype, not fundamentals

**Recommendation**: **Not recommended** for serious investors. GameStop remains speculative with poor fundamentals. If you want tech exposure, consider established companies like Microsoft or Sony who actually profit from gaming.""",
        "feedback": -1  # Bad advice (too speculative recommendation)
    },
    
    # ========== PORTFOLIO MANAGER QUERIES ==========
    {
        "agent_type": "portfolio_manager",
        "query": "How's my portfolio performing?",
        "response": """**Portfolio Performance Summary:**

**Overall Performance**:
- Current Value: $52,450
- YTD Return: +12.3%
- S&P 500 YTD: +9.1%
- **Outperformance**: +3.2% vs market

**Allocation Breakdown**:
- Stocks: 55% ($28,848)
- Bonds: 35% ($18,358)
- Cash: 10% ($5,245)

**Top Performers**:
1. NVDA: +45% ($3,200 gain)
2. VOO (S&P 500 ETF): +18% ($1,850 gain)
3. MSFT: +22% ($1,100 gain)

**Underperformers**:
- BND (Bond ETF): -2% ($180 loss) - expected in rising rate environment

**Assessment**: Your portfolio is performing well, beating the market through strong tech exposure. Consider rebalancing into bonds to lock in gains and reduce volatility.""",
        "feedback": 1
    },
    {
        "agent_type": "portfolio_manager",
        "query": "Should I rebalance my portfolio?",
        "response": """**Rebalancing Recommendation: YES**

**Current vs Target Allocation**:
```
Asset Class    Current    Target    Drift
────────────────────────────────────────
Stocks         62%        50%       +12%  ⚠️
Bonds          28%        40%       -12%  ⚠️
Cash           10%        10%       0%     ✓
```

**Why Rebalance**:
1. Your stocks have grown to 62% (target: 50%) - increases risk
2. Bonds below target - missing diversification benefits
3. Strong stock gains = good time to take profits

**Recommended Actions**:
1. Sell $6,300 in stocks (sell highest gainers first for tax efficiency)
2. Buy $6,300 in bonds (consider BND, AGG, or intermediate-term corporate bonds)

**Tax Consideration**: If selling stocks with gains, hold >1 year for long-term capital gains rate (15-20% vs. ordinary income rate).

**Next Steps**: Execute trades within the next week. Review again in 6 months.""",
        "feedback": 1
    },
    {
        "agent_type": "portfolio_manager",
        "query": "What stocks do I currently own?",
        "response": """**Your Current Stock Holdings:**

**Individual Stocks** (35% of portfolio):
1. **NVDA** - 15 shares @ $140.50 avg = $2,108
2. **AAPL** - 25 shares @ $195.30 avg = $4,883
3. **MSFT** - 12 shares @ $378.20 avg = $4,538
4. **TSLA** - 8 shares @ $245.10 avg = $1,961

**ETFs** (27% of portfolio):
1. **VOO** (S&P 500) - 45 shares @ $420.15 = $18,907
2. **QQQ** (Nasdaq 100) - 10 shares @ $385.40 = $3,854

**Total Stock Value**: $28,848 (55% of portfolio)

**Observations**:
- Heavy tech concentration (NVDA, AAPL, MSFT, TSLA = 47% of stocks)
- Good core with VOO for broad market exposure
- Consider adding non-tech sectors for better diversification""",
        "feedback": 1
    },
    
    # ========== TRADE EXECUTION QUERIES ==========
    {
        "agent_type": "trade_executor",
        "query": "Buy $500 of Apple stock",
        "response": """**Trade Executed Successfully** ✓

**Order Details**:
- **Action**: BUY
- **Ticker**: AAPL (Apple Inc.)
- **Amount**: $500.00
- **Execution Price**: $195.45 per share
- **Shares Acquired**: 2.56 shares
- **Order Type**: Market order
- **Status**: Filled
- **Timestamp**: 2025-01-20 10:32:15 EST

**Updated Position**:
- Previous AAPL shares: 25.00
- New AAPL shares: 27.56
- Total AAPL value: $5,386.60

Your order has been executed. Confirmation email sent.""",
        "feedback": 1
    },
    {
        "agent_type": "trade_executor",
        "query": "Sell $1000 worth of Tesla",
        "response": """**Trade Executed Successfully** ✓

**Order Details**:
- **Action**: SELL
- **Ticker**: TSLA (Tesla Inc.)
- **Amount**: $1,000.00
- **Execution Price**: $245.30 per share
- **Shares Sold**: 4.08 shares
- **Order Type**: Market order
- **Status**: Filled
- **Timestamp**: 2025-01-20 11:15:42 EST

**Updated Position**:
- Previous TSLA shares: 8.00
- New TSLA shares: 3.92
- Remaining TSLA value: $961.58

**Proceeds**: $1,000 added to cash balance.

Your order has been executed. Confirmation email sent.""",
        "feedback": 1
    },
    
    # ========== EDGE CASES ==========
    {
        "agent_type": "trade_executor",
        "query": "Buy some Apple stock",
        "response": """**Cannot Execute Trade** ❌

**Issue**: Missing dollar amount

To execute a trade, I need:
1. ✓ Ticker symbol (AAPL provided)
2. ❌ Dollar amount (not specified)

**Please specify**: How much would you like to invest in Apple?

**Example**: "Buy $500 of Apple stock" or "Buy $1000 of AAPL" """,
        "feedback": 1  # Good error handling
    },
]


# ============================================================================
# Generator Functions
# ============================================================================

def generate_synthetic_data(
    test_user_id: str,
    num_copies: int = 1,
    add_noise: bool = False
) -> Dict[str, int]:
    """
    Generate synthetic conversation data for a test user.
    
    This creates a realistic memory bank for testing the RL system
    before real user interactions accumulate.
    
    Args:
        test_user_id: User ID to generate data for
        num_copies: Number of times to replicate the conversation set
        add_noise: Whether to add slight variations to conversations
        
    Returns:
        Dictionary with generation statistics
    """
    memory_manager = create_memory_manager()
    
    stats = {
        'total_generated': 0,
        'positive_feedback': 0,
        'negative_feedback': 0,
        'neutral_feedback': 0
    }
    
    for copy_num in range(num_copies):
        for i, conv in enumerate(SYNTHETIC_CONVERSATIONS):
            # Generate unique thread ID for each conversation
            thread_id = f"synthetic-thread-{copy_num}-{i}"
            
            # Store conversation
            try:
                experience_id = memory_manager.store_interaction(
                    user_id=test_user_id,
                    thread_id=thread_id,
                    query_text=conv["query"],
                    agent_response=conv["response"],
                    agent_type=conv["agent_type"],
                    context_snapshot={
                        "synthetic": True,
                        "copy_number": copy_num,
                        "conversation_index": i
                    }
                )
                
                # Add feedback (reward signal)
                is_positive = conv["feedback"] > 0
                memory_manager.record_feedback(experience_id, is_positive)
                
                # Update stats
                stats['total_generated'] += 1
                if conv["feedback"] > 0:
                    stats['positive_feedback'] += 1
                elif conv["feedback"] < 0:
                    stats['negative_feedback'] += 1
                else:
                    stats['neutral_feedback'] += 1
                
                logger.debug(f"Generated synthetic experience {i+1}/{len(SYNTHETIC_CONVERSATIONS)}")
                
            except Exception as e:
                logger.error(f"Failed to generate synthetic experience: {e}")
                continue
    
    logger.info(f"Generated {stats['total_generated']} synthetic experiences for user {test_user_id}")
    return stats


def print_synthetic_data_summary():
    """Print summary of available synthetic conversations"""
    print("\n" + "="*80)
    print("SYNTHETIC CONVERSATION DATA SUMMARY")
    print("="*80)
    
    by_agent = {}
    total_positive = 0
    total_negative = 0
    
    for conv in SYNTHETIC_CONVERSATIONS:
        agent = conv['agent_type']
        if agent not in by_agent:
            by_agent[agent] = {'count': 0, 'positive': 0, 'negative': 0}
        
        by_agent[agent]['count'] += 1
        if conv['feedback'] > 0:
            by_agent[agent]['positive'] += 1
            total_positive += 1
        elif conv['feedback'] < 0:
            by_agent[agent]['negative'] += 1
            total_negative += 1
    
    print(f"\nTotal Conversations: {len(SYNTHETIC_CONVERSATIONS)}")
    print(f"Positive Feedback: {total_positive} ({total_positive/len(SYNTHETIC_CONVERSATIONS)*100:.1f}%)")
    print(f"Negative Feedback: {total_negative} ({total_negative/len(SYNTHETIC_CONVERSATIONS)*100:.1f}%)")
    
    print("\nBreakdown by Agent:")
    for agent, stats in by_agent.items():
        print(f"  {agent}:")
        print(f"    Total: {stats['count']}")
        print(f"    Positive: {stats['positive']}")
        print(f"    Negative: {stats['negative']}")
    
    print("\n" + "="*80 + "\n")


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic conversation data for RL system testing")
    parser.add_argument("--user-id", required=True, help="User ID to generate data for")
    parser.add_argument("--copies", type=int, default=5, help="Number of copies of conversation set (default: 5)")
    parser.add_argument("--summary", action="store_true", help="Print summary of synthetic data")
    
    args = parser.parse_args()
    
    if args.summary:
        print_synthetic_data_summary()
    
    print(f"\nGenerating synthetic data for user: {args.user_id}")
    print(f"Copies: {args.copies}\n")
    
    stats = generate_synthetic_data(args.user_id, num_copies=args.copies)
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total experiences: {stats['total_generated']}")
    print(f"Positive feedback: {stats['positive_feedback']}")
    print(f"Negative feedback: {stats['negative_feedback']}")
    print(f"Neutral feedback: {stats['neutral_feedback']}")
    print("="*80 + "\n")

