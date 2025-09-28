# LangGraph Investment Research Assistant

A comprehensive proof of concept demonstrating advanced checkpointing and time-travel debugging capabilities using LangGraph for building stateful AI agents.

## Overview

This project showcases how to build a sophisticated stateful agent for investment research that can be paused, resumed, modified, and debugged using LangGraph's powerful persistence layer. The agent analyzes stocks through multiple stages and demonstrates all key checkpointing features.

## =€ Features Demonstrated

### Core LangGraph Capabilities
-  **Complete State Persistence** with `InMemorySaver` (production-ready for `SqliteSaver`/`PostgresSaver`)
-  **Checkpoint History Tracking** and inspection with metadata
-  **Time-Travel Debugging** with state modification and replay
-  **State Forking** for alternative execution paths
-  **Human-in-the-Loop** workflows with interrupts
-  **Failure Recovery** and error handling
-  **Parallel Execution** tracking across different threads
-  **State Comparison** and analysis between execution paths

### Investment Research Workflow
- =Ê **Company Information Gathering** (mock data with real API integration points)
- =È **Technical Analysis** (price trends, RSI, moving averages)
- =° **Fundamental Analysis** (P/E ratios, revenue growth, financial health)
-   **Risk Assessment** (automated risk scoring and categorization)
- <¯ **Investment Recommendations** (buy/hold/sell with confidence scores)
- =d **Human Review Process** for high-risk investments

## =à Installation & Setup

### Prerequisites
- Python 3.11+
- `uv` package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd langgraph-stateful-agent-example

# Install dependencies using uv
uv sync

# Run the demonstration
uv run python main.py
```

## =Ë Dependencies

Key dependencies include:
- `langgraph>=0.2.0` - Core state graph functionality with checkpointing
- `langchain>=0.3.0` - Foundation for LLM integration
- `rich>=13.0.0` - Beautiful console output and formatting
- `colorama>=0.4.6` - Cross-platform colored terminal output
- `psycopg2-binary>=2.9.0` - PostgreSQL support for production persistence

## <× Architecture

### State Definition
The `InvestmentResearchState` TypedDict includes:

```python
class InvestmentResearchState(TypedDict):
    # Core identification
    ticker: str
    thread_id: str

    # Analysis results
    company_info: Optional[CompanyInfo]
    technical_analysis: Optional[TechnicalAnalysis]
    fundamental_analysis: Optional[FundamentalAnalysis]
    risk_assessment: Optional[RiskAssessment]
    recommendation: Optional[Recommendation]

    # Process tracking
    messages: Annotated[List[BaseMessage], add_messages]
    analysis_timestamp: str
    current_step: str
    steps_completed: List[str]

    # Human-in-the-loop controls
    human_approval_required: bool
    human_feedback: Optional[str]
    approved_by_human: bool

    # Debugging and recovery
    error_count: int
    last_error: Optional[str]
    recovery_attempts: int

    # Configuration
    risk_tolerance: str
    analysis_depth: str
```

### Graph Nodes
1. **`gather_company_info`** - Fetches basic company data
2. **`perform_technical_analysis`** - Analyzes price trends and indicators
3. **`perform_fundamental_analysis`** - Evaluates financial health
4. **`assess_risks`** - Identifies and scores investment risks
5. **`human_review`** - Interrupts for human approval on high-risk investments
6. **`generate_recommendation`** - Creates final investment advice

### Conditional Edges
- **Risk-based routing** to human review for high-risk investments
- **Automatic continuation** for low-risk investments
- **State-dependent execution** paths based on analysis results

## <® Demonstration Phases

The demo runs through 7 comprehensive phases:

### Phase 1: Complete Analysis
- Runs full investment analysis for AAPL
- Demonstrates automatic checkpointing at each step
- Shows state persistence and recovery

### Phase 2: Checkpoint History
- Displays all saved checkpoints with metadata
- Shows checkpoint IDs, steps completed, and message counts
- Demonstrates checkpoint inspection capabilities

### Phase 3: Time-Travel Debugging
- Goes back to an earlier checkpoint (after technical analysis)
- Modifies state (changes risk tolerance to 'conservative')
- Resumes execution from the modified state

### Phase 4: State Forking
- Creates an alternative execution path
- Modifies parameters (comprehensive analysis, aggressive risk tolerance)
- Demonstrates parallel state management

### Phase 5: Failure Recovery
- Simulates a network timeout failure
- Shows error injection and state tracking
- Demonstrates recovery mechanisms

### Phase 6: Fork Execution
- Resumes execution on the forked thread
- Shows parallel analysis with different parameters
- Compares results between execution paths

### Phase 7: Results Comparison
- Side-by-side comparison of original vs. forked execution
- Shows how different parameters affect outcomes
- Demonstrates the value of A/B testing with state forking

## =' Key Code Examples

### Creating the Graph with Checkpointing
```python
def create_investment_research_graph() -> StateGraph:
    # Initialize checkpointer
    checkpointer = MemorySaver()  # Use SqliteSaver for production

    # Create and configure the state graph
    workflow = StateGraph(InvestmentResearchState)
    workflow.add_node("gather_company_info", gather_company_info)
    # ... add other nodes

    # Add conditional edges for human review
    workflow.add_conditional_edges(
        "assess_risks",
        should_require_human_review,
        {
            "human_review": "human_review",
            "generate_recommendation": "generate_recommendation"
        }
    )

    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)
```

### Time-Travel Debugging
```python
def demonstrate_time_travel(graph, config):
    # Get checkpoint history
    checkpoints = list(graph.get_state_history(config))

    # Find target checkpoint
    target_checkpoint = find_checkpoint_after_technical_analysis(checkpoints)

    # Modify state
    modified_state = {
        **target_checkpoint.values,
        "risk_tolerance": "conservative",
        "messages": target_checkpoint.values.get("messages", []) + [
            HumanMessage(content="Modified during time-travel")
        ]
    }

    # Create new execution path
    new_config = {"configurable": {"thread_id": f"{thread_id}_modified"}}
    graph.update_state(new_config, modified_state)

    return new_config
```

### Human-in-the-Loop
```python
def human_review(state: InvestmentResearchState) -> InvestmentResearchState:
    review_summary = f"""
    Investment Review Required for {state['ticker']}
    Risk Level: {state['risk_assessment'].risk_level.value}
    Risk Score: {state['risk_assessment'].risk_score:.2f}
    """

    # This interrupts execution for human input
    raise NodeInterrupt(review_summary)
```

## =€ Production Considerations

### Persistent Storage
Replace `InMemorySaver` with production-grade persistence:

```python
# SQLite for single-machine deployment
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("investment_research.db")

# PostgreSQL for distributed deployment
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@localhost/db")
```

### Real API Integration
Replace mock functions with real financial APIs:
- **Alpha Vantage** for market data
- **Yahoo Finance** for stock information
- **Bloomberg API** for professional-grade data
- **SEC EDGAR** for fundamental analysis

### Security & Authentication
- Implement proper authentication for human review processes
- Secure API keys and database connections
- Add audit logging for compliance

### Scalability
- Use distributed checkpointing for multi-node deployments
- Implement proper error handling and retry mechanisms
- Add monitoring and alerting for production workflows

### Performance Optimization
- Optimize checkpoint storage for large state objects
- Implement checkpoint cleanup and archival strategies
- Add performance monitoring and metrics

## <¯ Use Cases

This pattern is valuable for:
- **Financial Analysis** - Complex multi-step investment research
- **Data Processing Pipelines** - Long-running analysis workflows
- **Customer Service** - Multi-agent conversation handling
- **Scientific Computing** - Reproducible research workflows
- **Business Process Automation** - Complex approval workflows
- **A/B Testing** - Comparing different execution strategies

## =Ê Output Example

The demo produces beautiful, formatted output showing:
- Real-time progress with colored emojis
- Detailed state summaries in formatted tables
- Checkpoint history with metadata
- Side-by-side execution comparisons
- Clear phase separation and progress tracking

## > Contributing

Contributions are welcome! Areas for enhancement:
- Additional analysis nodes (sentiment analysis, news integration)
- More sophisticated risk models
- Real-time data integration
- Advanced visualization features
- Performance optimizations

## =Ä License

This project is open source and available under the MIT License.

## <“ Learning Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Checkpointing Guide](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Human-in-the-Loop Patterns](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [State Management Best Practices](https://langchain-ai.github.io/langgraph/concepts/low_level/)

---

*This implementation serves as both a learning resource and a production-ready template for building sophisticated stateful AI agents with LangGraph.*