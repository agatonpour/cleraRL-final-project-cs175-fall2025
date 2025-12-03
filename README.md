# CS 175 Final Project: Memory-Augmented Reinforcement Learning for Clera

## An AI Investment Advisor That Learns From User Feedback

**Team Members:**
- Cristian Mendoza (cfmendo1@uci.edu)
- Delphine Tai-Beauchamp
- Agaton Pourshahidi

**Course:** CS 175 - Reinforcement Learning, Fall 2025  
**University of California, Irvine**

---

## Project Overview

We implement a reinforcement learning system for **Clera**, an AI-powered investment advisor platform. Our approach uses **experience replay** and **reward-weighted retrieval** to enable the system to learn from user feedback (thumbs up/down) without expensive model retraining.

### Key RL Concepts Applied
- **Experience Replay** (Mnih et al., 2013) - Store past conversations with embeddings
- **Reward-Based Learning** (Sutton & Barto) - User feedback guides retrieval
- **Behavioral Cloning** (Pomerleau, 1991) - Show successful examples to agents

---

## Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/cmendoza1031/cleraRL-final-project-cs175-fall2025.git
cd cleraRL-final-project-cs175-fall2025
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Demo

### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook project.ipynb
```
Then click **Kernel → Restart & Run All** to execute all cells.

**Expected runtime:** < 30 seconds

### Option 2: View Pre-Rendered HTML
Open `project.html` in any web browser to see the executed notebook with all outputs.

---

## Project Structure

```
cleraRL-final-project-cs175-fall2025/
├── README.md                 # This file - setup instructions
├── requirements.txt          # Python dependencies
├── project.ipynb             # Main demo notebook (RUNNABLE)
├── project.html              # Pre-rendered notebook output
├── fall2025-cs175-cleraRL.pdf # Final report (PDF)
├── src/                      # Source code modules
│   ├── __init__.py           # Interfaces (IEmbeddingProvider, IMemoryStore)
│   ├── embedding_provider.py # OpenAI text-embedding-3-small integration
│   ├── memory_store.py       # PostgreSQL + pgvector storage
│   ├── memory_manager.py     # High-level memory orchestration
│   ├── agent_wrapper.py      # Decorator pattern for agent integration
│   ├── memory_graph.py       # LangGraph workflow integration
│   ├── rl_routes.py          # FastAPI endpoints for feedback
│   ├── generate_synthetic_data.py  # Training data generation
│   └── evaluate_rl_system.py # Evaluation metrics
└── *.png                     # Visualization outputs
```

---

## What the Notebook Demonstrates

1. **Synthetic Training Data** - 50 conversation experiences with feedback scores
2. **Memory Accumulation** - How experiences grow over a 2-week deployment
3. **User Satisfaction** - 74% positive feedback (exceeds 70% target)
4. **Experience Replay** - How similar past successes inform new responses
5. **Reward-Weighted Retrieval** - Prioritizing high-feedback experiences
6. **RL Loop Visualization** - State → Action → Reward → Learn → Improve

---

## Results Summary

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Total Experiences | 50+ | 50 | Met |
| User Satisfaction | >70% | 74% | Exceeds |
| Learning Rate | >0.7 | 0.74 | Exceeds |

---

## Technical Approach

### Core Algorithm: Reward-Weighted Experience Replay

```python
def process_query(user_query, user_id):
    # 1. Generate embedding for query
    embedding = embed(user_query)  # 1536-dim vector
    
    # 2. Retrieve similar past experiences, prioritizing high rewards
    experiences = db.query(
        "ORDER BY feedback_score DESC, similarity DESC"
    )
    
    # 3. Inject successful patterns (behavioral cloning)
    response = agent.generate(query, context=experiences)
    
    # 4. Store for future learning
    db.store(query, response, embedding)
    
    return response
```

---

## Data

The notebook generates **synthetic training data** based on real Clera production patterns. No external datasets are required - everything needed to run the demo is included.

---

## Contact

For questions about this project, please contact:
- Cristian Mendoza: cfmendo1@uci.edu

---

**Thank you for reviewing our project!**
