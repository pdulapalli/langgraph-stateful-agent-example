"""
LangGraph Investment Research Assistant with Advanced Checkpointing and Time-Travel Debugging

This module demonstrates a complete proof of concept for a stateful investment research agent
that showcases LangGraph's advanced checkpointing capabilities including:
- State persistence and restoration
- Time-travel debugging
- Human-in-the-loop workflows
- State modification and forking
- Recovery from failures

The agent analyzes stocks through multiple stages: company info gathering, technical analysis,
fundamental analysis, risk assessment, and recommendation generation.
"""

import logging
import random
import time
from datetime import datetime
from typing import TypedDict, List, Optional, Annotated
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langgraph.errors import GraphInterrupt
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from colorama import init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize rich console for beautiful output
console = Console()


class RiskLevel(Enum):
    """Risk level enumeration for investment recommendations."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RecommendationType(Enum):
    """Investment recommendation types."""

    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class CompanyInfo:
    """Company information data structure."""

    name: str
    sector: str
    market_cap: float
    employees: int
    description: str
    founded_year: int


@dataclass
class TechnicalAnalysis:
    """Technical analysis results."""

    current_price: float
    price_change_1d: float
    price_change_7d: float
    price_change_30d: float
    moving_average_20: float
    moving_average_50: float
    rsi: float
    volume: int
    trend: str


@dataclass
class FundamentalAnalysis:
    """Fundamental analysis metrics."""

    pe_ratio: float
    revenue_growth: float
    profit_margin: float
    debt_to_equity: float
    return_on_equity: float
    book_value: float
    dividend_yield: float


@dataclass
class RiskAssessment:
    """Risk assessment results."""

    risk_level: RiskLevel
    risk_score: float
    risk_factors: List[str]
    mitigation_strategies: List[str]


@dataclass
class Recommendation:
    """Investment recommendation."""

    recommendation_type: RecommendationType
    confidence: float
    target_price: float
    time_horizon: str
    reasoning: str


class InvestmentResearchState(TypedDict):
    """
    Comprehensive state definition for the Investment Research Assistant.

    This state tracks all information gathered during the research process,
    enabling complete state persistence and time-travel debugging capabilities.
    """

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
    risk_tolerance: str  # conservative, moderate, aggressive
    analysis_depth: str  # basic, standard, comprehensive


def create_mock_company_info(ticker: str) -> CompanyInfo:
    """Create mock company information for demonstration purposes."""
    companies = {
        "AAPL": CompanyInfo(
            "Apple Inc.",
            "Technology",
            3000000000000,
            154000,
            "Technology company focused on consumer electronics",
            1976,
        ),
        "GOOGL": CompanyInfo(
            "Alphabet Inc.",
            "Technology",
            2000000000000,
            174000,
            "Multinational technology company",
            1998,
        ),
        "TSLA": CompanyInfo(
            "Tesla Inc.",
            "Automotive",
            800000000000,
            127855,
            "Electric vehicle and clean energy company",
            2003,
        ),
        "MSFT": CompanyInfo(
            "Microsoft Corporation",
            "Technology",
            2800000000000,
            221000,
            "Multinational technology corporation",
            1975,
        ),
    }

    return companies.get(
        ticker,
        CompanyInfo(
            f"{ticker} Corp",
            "Unknown",
            1000000000,
            10000,
            f"Mock company for {ticker}",
            2000,
        ),
    )


def create_mock_technical_analysis(ticker: str) -> TechnicalAnalysis:
    """Generate mock technical analysis data."""
    base_price = random.uniform(50, 500)
    return TechnicalAnalysis(
        current_price=base_price,
        price_change_1d=random.uniform(-5, 5),
        price_change_7d=random.uniform(-15, 15),
        price_change_30d=random.uniform(-30, 30),
        moving_average_20=base_price * random.uniform(0.95, 1.05),
        moving_average_50=base_price * random.uniform(0.90, 1.10),
        rsi=random.uniform(20, 80),
        volume=random.randint(1000000, 100000000),
        trend=random.choice(["bullish", "bearish", "sideways"]),
    )


def create_mock_fundamental_analysis(ticker: str) -> FundamentalAnalysis:
    """Generate mock fundamental analysis data."""
    return FundamentalAnalysis(
        pe_ratio=random.uniform(10, 40),
        revenue_growth=random.uniform(-20, 50),
        profit_margin=random.uniform(5, 25),
        debt_to_equity=random.uniform(0.1, 2.0),
        return_on_equity=random.uniform(5, 30),
        book_value=random.uniform(20, 200),
        dividend_yield=random.uniform(0, 5),
    )


# Graph Node Functions
def gather_company_info(state: InvestmentResearchState) -> InvestmentResearchState:
    """
    Gather basic company information.

    In production, this would integrate with financial APIs like Alpha Vantage,
    Yahoo Finance, or Bloomberg. For demonstration, we use mock data.
    """
    console.print(
        f"[blue]Gathering company information for {state['ticker']}...[/blue]"
    )

    # Simulate API call delay
    time.sleep(1)

    try:
        company_info = create_mock_company_info(state["ticker"])

        message = AIMessage(
            content=f"Gathered company information for {company_info.name}"
        )

        return {
            **state,
            "company_info": company_info,
            "current_step": "company_info_complete",
            "steps_completed": state["steps_completed"] + ["company_info"],
            "messages": [message],
        }

    except Exception as e:
        logger.error(f"Error gathering company info: {e}")
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "last_error": str(e),
            "messages": [AIMessage(content=f"Error gathering company info: {e}")],
        }


def perform_technical_analysis(
    state: InvestmentResearchState,
) -> InvestmentResearchState:
    """
    Perform technical analysis on the stock.

    Analyzes price trends, moving averages, RSI, volume, and other technical indicators.
    """
    console.print(
        f"[green]Performing technical analysis for {state['ticker']}...[/green]"
    )

    # Simulate analysis time
    time.sleep(1.5)

    try:
        technical_analysis = create_mock_technical_analysis(state["ticker"])

        # Determine trend strength
        trend_strength = (
            "strong" if abs(technical_analysis.price_change_30d) > 20 else "moderate"
        )

        message = AIMessage(
            content=f"Technical analysis complete. Trend: {technical_analysis.trend} ({trend_strength}), "
            f"RSI: {technical_analysis.rsi:.1f}, Price: ${technical_analysis.current_price:.2f}"
        )

        return {
            **state,
            "technical_analysis": technical_analysis,
            "current_step": "technical_analysis_complete",
            "steps_completed": state["steps_completed"] + ["technical_analysis"],
            "messages": [message],
        }

    except Exception as e:
        logger.error(f"Error in technical analysis: {e}")
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "last_error": str(e),
            "messages": [AIMessage(content=f"Error in technical analysis: {e}")],
        }


def perform_fundamental_analysis(
    state: InvestmentResearchState,
) -> InvestmentResearchState:
    """
    Perform fundamental analysis of the company.

    Evaluates financial health, valuation metrics, and growth prospects.
    """
    console.print(
        f"[yellow]Performing fundamental analysis for {state['ticker']}...[/yellow]"
    )

    # Simulate analysis time
    time.sleep(2)

    try:
        fundamental_analysis = create_mock_fundamental_analysis(state["ticker"])

        # Assess financial health
        health_score = 0
        if fundamental_analysis.pe_ratio < 25:
            health_score += 1
        if fundamental_analysis.revenue_growth > 10:
            health_score += 1
        if fundamental_analysis.debt_to_equity < 1.0:
            health_score += 1
        if fundamental_analysis.return_on_equity > 15:
            health_score += 1

        health_rating = ["Poor", "Fair", "Good", "Excellent"][health_score]

        message = AIMessage(
            content=f"Fundamental analysis complete. Financial health: {health_rating}, "
            f"P/E: {fundamental_analysis.pe_ratio:.1f}, Revenue growth: {fundamental_analysis.revenue_growth:.1f}%"
        )

        return {
            **state,
            "fundamental_analysis": fundamental_analysis,
            "current_step": "fundamental_analysis_complete",
            "steps_completed": state["steps_completed"] + ["fundamental_analysis"],
            "messages": [message],
        }

    except Exception as e:
        logger.error(f"Error in fundamental analysis: {e}")
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "last_error": str(e),
            "messages": [AIMessage(content=f"Error in fundamental analysis: {e}")],
        }


def assess_risks(state: InvestmentResearchState) -> InvestmentResearchState:
    """
    Assess investment risks based on technical and fundamental analysis.

    Identifies potential risks and calculates an overall risk score.
    """
    console.print(f"[red]Assessing investment risks for {state['ticker']}...[/red]")

    # Simulate risk assessment time
    time.sleep(1)

    try:
        risk_factors = []
        risk_score = 0.0

        # Technical risk factors
        if state["technical_analysis"]:
            ta = state["technical_analysis"]
            if ta.rsi > 70:
                risk_factors.append("Stock may be overbought (RSI > 70)")
                risk_score += 0.2
            elif ta.rsi < 30:
                risk_factors.append("Stock may be oversold (RSI < 30)")
                risk_score += 0.1

            if abs(ta.price_change_30d) > 25:
                risk_factors.append("High price volatility in past 30 days")
                risk_score += 0.3

        # Fundamental risk factors
        if state["fundamental_analysis"]:
            fa = state["fundamental_analysis"]
            if fa.pe_ratio > 30:
                risk_factors.append("High P/E ratio indicates potential overvaluation")
                risk_score += 0.2

            if fa.debt_to_equity > 1.5:
                risk_factors.append("High debt-to-equity ratio")
                risk_score += 0.3

            if fa.revenue_growth < 0:
                risk_factors.append("Negative revenue growth")
                risk_score += 0.4

        # Determine risk level
        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Generate mitigation strategies
        mitigation_strategies = [
            "Diversify portfolio to reduce concentration risk",
            "Set stop-loss orders to limit downside",
            "Monitor key financial metrics regularly",
            "Consider position sizing based on risk tolerance",
        ]

        risk_assessment = RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors
            if risk_factors
            else ["No significant risks identified"],
            mitigation_strategies=mitigation_strategies,
        )

        message = AIMessage(
            content=f"Risk assessment complete. Risk level: {risk_level.value}, "
            f"Score: {risk_score:.2f}, Factors identified: {len(risk_factors)}"
        )

        # Set human approval flag for high-risk investments
        human_approval_required = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

        return {
            **state,
            "risk_assessment": risk_assessment,
            "human_approval_required": human_approval_required,
            "current_step": "risk_assessment_complete",
            "steps_completed": state["steps_completed"] + ["risk_assessment"],
            "messages": [message],
        }

    except Exception as e:
        logger.error(f"Error in risk assessment: {e}")
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "last_error": str(e),
            "messages": [AIMessage(content=f"Error in risk assessment: {e}")],
        }


def human_review(state: InvestmentResearchState) -> InvestmentResearchState:
    """
    Human-in-the-loop node for reviewing high-risk investments.

    This node interrupts execution to allow human review and approval.
    """
    console.print(
        f"[magenta]Human review required for {state['ticker']} (Risk: {state['risk_assessment'].risk_level.value})[/magenta]"
    )

    # Create a detailed review summary
    review_summary = f"""
    Investment Review Required for {state['ticker']}

    Risk Level: {state['risk_assessment'].risk_level.value}
    Risk Score: {state['risk_assessment'].risk_score:.2f}

    Risk Factors:
    {chr(10).join(f"  • {factor}" for factor in state['risk_assessment'].risk_factors)}

    Please review and approve/reject this analysis.
    """

    message = HumanMessage(content="Human review checkpoint - awaiting approval")

    # This will interrupt the graph execution
    interrupt(review_summary)


def generate_recommendation(state: InvestmentResearchState) -> InvestmentResearchState:
    """
    Generate final investment recommendation based on all analysis.

    Combines technical, fundamental, and risk analysis to create actionable recommendations.
    """
    console.print(
        f"[cyan]Generating investment recommendation for {state['ticker']}...[/cyan]"
    )

    # Simulate recommendation generation time
    time.sleep(1)

    try:
        # Score-based recommendation logic
        score = 0.0
        reasoning_points = []

        # Technical analysis contribution
        if state["technical_analysis"]:
            ta = state["technical_analysis"]
            if ta.trend == "bullish":
                score += 0.3
                reasoning_points.append("Bullish technical trend")
            elif ta.trend == "bearish":
                score -= 0.3
                reasoning_points.append("Bearish technical trend")

            if 30 <= ta.rsi <= 70:
                score += 0.1
                reasoning_points.append("RSI in neutral range")

        # Fundamental analysis contribution
        if state["fundamental_analysis"]:
            fa = state["fundamental_analysis"]
            if fa.revenue_growth > 15:
                score += 0.3
                reasoning_points.append("Strong revenue growth")
            elif fa.revenue_growth < 0:
                score -= 0.2
                reasoning_points.append("Declining revenue")

            if fa.pe_ratio < 20:
                score += 0.2
                reasoning_points.append("Reasonable valuation")
            elif fa.pe_ratio > 35:
                score -= 0.2
                reasoning_points.append("High valuation concerns")

        # Risk adjustment
        if state["risk_assessment"]:
            risk_penalty = state["risk_assessment"].risk_score * 0.5
            score -= risk_penalty
            reasoning_points.append(f"Risk adjustment: -{risk_penalty:.2f}")

        # Generate recommendation type
        if score >= 0.6:
            rec_type = RecommendationType.STRONG_BUY
        elif score >= 0.3:
            rec_type = RecommendationType.BUY
        elif score >= -0.2:
            rec_type = RecommendationType.HOLD
        elif score >= -0.5:
            rec_type = RecommendationType.SELL
        else:
            rec_type = RecommendationType.STRONG_SELL

        # Calculate confidence and target price
        confidence = min(0.95, max(0.5, abs(score) + 0.3))

        current_price = (
            state["technical_analysis"].current_price
            if state["technical_analysis"]
            else 100
        )
        if rec_type in [RecommendationType.STRONG_BUY, RecommendationType.BUY]:
            target_price = current_price * (1 + abs(score) * 0.3)
        else:
            target_price = current_price * (1 - abs(score) * 0.2)

        recommendation = Recommendation(
            recommendation_type=rec_type,
            confidence=confidence,
            target_price=target_price,
            time_horizon="12 months",
            reasoning=" | ".join(reasoning_points),
        )

        message = AIMessage(
            content=f"Investment recommendation: {rec_type.value} with {confidence:.1%} confidence. "
            f"Target price: ${target_price:.2f}"
        )

        return {
            **state,
            "recommendation": recommendation,
            "current_step": "recommendation_complete",
            "steps_completed": state["steps_completed"] + ["recommendation"],
            "messages": [message],
        }

    except Exception as e:
        logger.error(f"Error generating recommendation: {e}")
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "last_error": str(e),
            "messages": [AIMessage(content=f"Error generating recommendation: {e}")],
        }


def should_require_human_review(state: InvestmentResearchState) -> str:
    """
    Conditional edge function to determine if human review is required.

    Returns the next node to execute based on risk level.
    """
    if state.get("human_approval_required", False) and not state.get(
        "approved_by_human", False
    ):
        return "human_review"
    else:
        return "generate_recommendation"


def create_investment_research_graph() -> StateGraph:
    """
    Create and configure the Investment Research StateGraph with checkpointing.

    Returns a compiled graph with all nodes, edges, and checkpointer configured.
    """
    # Initialize checkpointer (InMemorySaver for demo)
    # In production, use SqliteSaver or PostgresSaver:
    # from langgraph.checkpoint.sqlite import SqliteSaver
    # checkpointer = SqliteSaver.from_conn_string("investment_research.db")
    #
    # from langgraph.checkpoint.postgres import PostgresSaver
    # checkpointer = PostgresSaver.from_conn_string("postgresql://user:pass@localhost/db")

    # Create a fresh checkpointer to ensure no state persists between runs
    checkpointer = MemorySaver()

    # Create the state graph
    workflow = StateGraph(InvestmentResearchState)

    # Add nodes
    workflow.add_node("gather_company_info", gather_company_info)
    workflow.add_node("perform_technical_analysis", perform_technical_analysis)
    workflow.add_node("perform_fundamental_analysis", perform_fundamental_analysis)
    workflow.add_node("assess_risks", assess_risks)
    workflow.add_node("human_review", human_review)
    workflow.add_node("generate_recommendation", generate_recommendation)

    # Add edges
    workflow.add_edge(START, "gather_company_info")
    workflow.add_edge("gather_company_info", "perform_technical_analysis")
    workflow.add_edge("perform_technical_analysis", "perform_fundamental_analysis")
    workflow.add_edge("perform_fundamental_analysis", "assess_risks")

    # Conditional edge for human review
    workflow.add_conditional_edges(
        "assess_risks",
        should_require_human_review,
        {
            "human_review": "human_review",
            "generate_recommendation": "generate_recommendation",
        },
    )

    workflow.add_edge("human_review", "generate_recommendation")
    workflow.add_edge("generate_recommendation", END)

    # Compile with checkpointer
    return workflow.compile(checkpointer=checkpointer)


# Time-Travel and Debugging Functions
def display_state_summary(state: InvestmentResearchState, title: str = "Current State"):
    """Display a formatted summary of the current state."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Ticker", state.get("ticker", "N/A"))
    table.add_row("Current Step", state.get("current_step", "N/A"))
    table.add_row("Steps Completed", ", ".join(state.get("steps_completed", [])))
    table.add_row(
        "Human Approval Required", str(state.get("human_approval_required", False))
    )
    table.add_row("Error Count", str(state.get("error_count", 0)))

    if state.get("risk_assessment"):
        table.add_row("Risk Level", state["risk_assessment"].risk_level.value)
        table.add_row("Risk Score", f"{state['risk_assessment'].risk_score:.2f}")

    if state.get("recommendation"):
        table.add_row(
            "Recommendation", state["recommendation"].recommendation_type.value
        )
        table.add_row("Confidence", f"{state['recommendation'].confidence:.1%}")

    console.print(table)


def display_checkpoints(graph, config):
    """Display all available checkpoints with metadata."""
    console.print("\n[bold blue]Available Checkpoints:[/bold blue]")

    checkpoints = list(graph.get_state_history(config))

    if not checkpoints:
        console.print("[red]No checkpoints found.[/red]")
        return

    table = Table(show_header=True, header_style="bold green")
    table.add_column("Step", style="cyan")
    table.add_column("Checkpoint ID", style="yellow")
    table.add_column("Current Step", style="white")
    table.add_column("Steps Completed", style="blue")
    table.add_column("Messages", style="magenta")

    for i, checkpoint in enumerate(checkpoints):
        state = checkpoint.values
        # Show full checkpoint ID or last 12 chars if too long
        full_checkpoint_id = checkpoint.config["configurable"]["checkpoint_id"]
        if len(full_checkpoint_id) > 12:
            # Show last 12 characters which are more likely to be unique
            checkpoint_id = "..." + full_checkpoint_id[-12:]
        else:
            checkpoint_id = full_checkpoint_id
        current_step = state.get("current_step", "N/A")
        steps_completed = len(state.get("steps_completed", []))
        message_count = len(state.get("messages", []))

        table.add_row(
            str(i + 1),
            checkpoint_id,
            current_step,
            str(steps_completed),
            str(message_count),
        )

    console.print(table)
    return checkpoints


def demonstrate_time_travel(graph, config):
    """Demonstrate time-travel debugging capabilities."""
    console.print("\n[bold yellow]Time-Travel Debugging Demonstration[/bold yellow]")

    # Get checkpoint history
    checkpoints = list(graph.get_state_history(config))

    if len(checkpoints) < 2:
        console.print(
            "[red]Not enough checkpoints for time-travel demonstration.[/red]"
        )
        return

    # Go back to an earlier checkpoint (e.g., after technical analysis)
    target_checkpoint = None
    for checkpoint in checkpoints:
        if "technical_analysis_complete" in checkpoint.values.get("current_step", ""):
            target_checkpoint = checkpoint
            break

    if not target_checkpoint:
        target_checkpoint = checkpoints[-2]  # Go back one step

    checkpoint_id = target_checkpoint.config['configurable']['checkpoint_id']
    display_id = checkpoint_id if len(checkpoint_id) <= 12 else "..." + checkpoint_id[-12:]
    console.print(
        f"\n[green]Traveling back to checkpoint: {display_id}[/green]"
    )

    # Display state at that checkpoint
    display_state_summary(target_checkpoint.values, "State at Selected Checkpoint")

    # Modify state (e.g., change risk tolerance)
    console.print(
        "\n[blue]Modifying state: Changing risk tolerance to 'conservative'[/blue]"
    )
    modified_state = {
        **target_checkpoint.values,
        "risk_tolerance": "conservative",
        "messages": target_checkpoint.values.get("messages", [])
        + [
            HumanMessage(
                content="Modified risk tolerance to conservative during time-travel"
            )
        ],
    }

    # Create new execution path from modified state
    console.print("\n[cyan]Resuming execution from modified state...[/cyan]")
    new_config = {
        "configurable": {
            "thread_id": f"{config['configurable']['thread_id']}_modified_{int(time.time())}"
        }
    }

    # Update the graph state with modified state
    graph.update_state(new_config, modified_state)

    return new_config


def demonstrate_state_forking(graph, original_config):
    """Demonstrate creating alternative execution paths (forking)."""
    console.print("\n[bold magenta]State Forking Demonstration[/bold magenta]")

    # Get current state
    current_state = graph.get_state(original_config)

    # Create a fork with different analysis parameters
    fork_config = {
        "configurable": {
            "thread_id": f"{original_config['configurable']['thread_id']}_fork_{int(time.time())}"
        }
    }

    # Modify state for the fork (e.g., different analysis depth)
    forked_state = {
        **current_state.values,
        "analysis_depth": "comprehensive",
        "risk_tolerance": "aggressive",
        "messages": current_state.values.get("messages", [])
        + [
            HumanMessage(
                content="Forked execution with comprehensive analysis and aggressive risk tolerance"
            )
        ],
    }

    console.print(
        f"[green]Creating fork with thread_id: {fork_config['configurable']['thread_id']}[/green]"
    )
    console.print(
        "[blue]Fork parameters: Comprehensive analysis, Aggressive risk tolerance[/blue]"
    )

    # Initialize the forked state
    graph.update_state(fork_config, forked_state)

    return fork_config


def demonstrate_recovery_scenario(graph, config):
    """Demonstrate recovery from a simulated failure."""
    console.print("\n[bold red]Failure Recovery Demonstration[/bold red]")

    # Get current state
    current_state = graph.get_state(config)

    # Simulate a failure by injecting an error
    error_state = {
        **current_state.values,
        "error_count": current_state.values.get("error_count", 0) + 1,
        "last_error": "Simulated network timeout during API call",
        "recovery_attempts": current_state.values.get("recovery_attempts", 0) + 1,
        "messages": current_state.values.get("messages", [])
        + [AIMessage(content="🚨 Simulated failure: Network timeout during API call")],
    }

    console.print("[red]Simulated failure injected: Network timeout[/red]")

    # Update state with error
    graph.update_state(config, error_state)

    # Show recovery process
    console.print("[yellow]Initiating recovery process...[/yellow]")

    # Simulate recovery by resetting error state
    recovery_state = {
        **error_state,
        "last_error": None,
        "messages": error_state.get("messages", [])
        + [AIMessage(content="Recovery successful: Retry mechanism activated")],
    }

    graph.update_state(config, recovery_state)
    console.print("[green]Recovery completed successfully[/green]")


def run_complete_analysis(graph, ticker: str, risk_tolerance: str = "moderate") -> dict:
    """Run a complete investment analysis with full checkpointing."""

    # Create unique thread ID for this analysis
    thread_id = f"investment_analysis_{ticker}_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    # Initialize state
    initial_state = InvestmentResearchState(
        ticker=ticker,
        thread_id=thread_id,
        company_info=None,
        technical_analysis=None,
        fundamental_analysis=None,
        risk_assessment=None,
        recommendation=None,
        messages=[],
        analysis_timestamp=datetime.now().isoformat(),
        current_step="initialized",
        steps_completed=[],
        human_approval_required=False,
        human_feedback=None,
        approved_by_human=False,
        error_count=0,
        last_error=None,
        recovery_attempts=0,
        risk_tolerance=risk_tolerance,
        analysis_depth="standard",
    )

    console.print(
        Panel.fit(
            f"[bold green]Starting Investment Analysis for {ticker}[/bold green]\n"
            f"Risk Tolerance: {risk_tolerance}\n"
            f"Thread ID: {thread_id}",
            title="Investment Research Assistant",
        )
    )

    try:
        # Run the analysis
        result = graph.invoke(initial_state, config)

        console.print("\n[bold green]Analysis completed successfully![/bold green]")
        display_state_summary(result, f"Final Analysis Results for {ticker}")

        return {"config": config, "result": result, "graph": graph}

    except GraphInterrupt as e:
        console.print(
            f"\n[yellow]Analysis paused for human review:[/yellow]\n{e.args[0]}"
        )

        # Simulate human approval
        console.print("\n[blue]Simulating human approval...[/blue]")
        time.sleep(2)

        # Get current state and approve
        current_state = graph.get_state(config)
        approved_state = {
            **current_state.values,
            "approved_by_human": True,
            "human_feedback": "Approved by human reviewer after risk assessment",
            "messages": current_state.values.get("messages", [])
            + [
                HumanMessage(
                    content="Human approval granted - proceeding with analysis"
                )
            ],
        }

        # Update state and resume
        graph.update_state(config, approved_state)

        console.print("[green]Human approval granted - resuming analysis...[/green]")

        # Resume from where we left off
        result = graph.invoke(None, config)

        console.print(
            "\n[bold green]Analysis completed after human review![/bold green]"
        )
        display_state_summary(result, f"Final Analysis Results for {ticker}")

        return {"config": config, "result": result, "graph": graph}


def main():
    """
    Main demonstration function showcasing all LangGraph checkpointing and time-travel features.
    """
    console.print(
        Panel.fit(
            "[bold blue]LangGraph Investment Research Assistant[/bold blue]\n"
            "[white]Advanced Checkpointing & Time-Travel Debugging Demo[/white]",
            title="System Startup",
        )
    )

    # Create the investment research graph with fresh state
    console.print("\n[cyan]Initializing Investment Research Graph with fresh state...[/cyan]")
    graph = create_investment_research_graph()
    console.print("[green]Graph initialized with fresh checkpointing state (no previous state persisted)[/green]")

    # Run initial analysis
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 1: Running Complete Analysis[/bold yellow]")
    console.print("=" * 60)

    analysis_result = run_complete_analysis(graph, "AAPL", "moderate")
    config = analysis_result["config"]

    # Display all checkpoints
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 2: Checkpoint History Analysis[/bold yellow]")
    console.print("=" * 60)

    checkpoints = display_checkpoints(graph, config)

    # Demonstrate time-travel debugging
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 3: Time-Travel Debugging[/bold yellow]")
    console.print("=" * 60)

    modified_config = demonstrate_time_travel(graph, config)

    # Demonstrate state forking
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 4: State Forking[/bold yellow]")
    console.print("=" * 60)

    fork_config = demonstrate_state_forking(graph, config)

    # Demonstrate recovery scenario
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 5: Failure Recovery[/bold yellow]")
    console.print("=" * 60)

    demonstrate_recovery_scenario(graph, config)

    # Run analysis on the fork to show parallel execution
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 6: Fork Execution[/bold yellow]")
    console.print("=" * 60)

    console.print("[cyan]Resuming execution on forked thread...[/cyan]")
    fork_result = graph.invoke(None, fork_config)
    display_state_summary(fork_result, "Fork Execution Results")

    # Compare results between original and fork
    console.print("\n" + "=" * 60)
    console.print("[bold yellow]PHASE 7: Results Comparison[/bold yellow]")
    console.print("=" * 60)

    original_state = graph.get_state(config)
    fork_state = graph.get_state(fork_config)

    console.print("\n[bold blue]Execution Path Comparison:[/bold blue]")

    comparison_table = Table(show_header=True, header_style="bold green")
    comparison_table.add_column("Metric", style="cyan")
    comparison_table.add_column("Original Thread", style="yellow")
    comparison_table.add_column("Forked Thread", style="magenta")

    comparison_table.add_row(
        "Risk Tolerance",
        original_state.values.get("risk_tolerance", "N/A"),
        fork_state.values.get("risk_tolerance", "N/A"),
    )

    comparison_table.add_row(
        "Analysis Depth",
        original_state.values.get("analysis_depth", "N/A"),
        fork_state.values.get("analysis_depth", "N/A"),
    )

    orig_rec = original_state.values.get("recommendation")
    fork_rec = fork_state.values.get("recommendation")

    comparison_table.add_row(
        "Recommendation",
        orig_rec.recommendation_type.value if orig_rec else "N/A",
        fork_rec.recommendation_type.value if fork_rec else "N/A",
    )

    comparison_table.add_row(
        "Confidence",
        f"{orig_rec.confidence:.1%}" if orig_rec else "N/A",
        f"{fork_rec.confidence:.1%}" if fork_rec else "N/A",
    )

    console.print(comparison_table)

    # Final summary
    console.print("\n" + "=" * 60)
    console.print("[bold green]DEMONSTRATION COMPLETE[/bold green]")
    console.print("=" * 60)


if __name__ == "__main__":
    main()
