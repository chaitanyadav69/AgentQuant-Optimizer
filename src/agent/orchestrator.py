"""
StratosAI: System Orchestrator
==============================
Implements a recursive feedback control loop for autonomous system calibration.
Flow: Environmental Analysis -> Hypothesis Synthesis -> Validation -> Recursive Reflection.
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd

from src.agent.context_builder import RegimeContext, build_context
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.utils.config import config

# Professional Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemState(TypedDict, total=False):
    """Encapsulates the state-space of the autonomous controller."""
    raw_telemetry: Dict[str, pd.DataFrame]
    processed_features: pd.DataFrame
    environmental_context: Optional[RegimeContext]
    candidate_proposals: List[Proposal]
    performance_benchmarks: List[Dict[str, Any]]
    optimal_config: Optional[Dict[str, Any]]
    current_cycle: int
    loop_limit: int
    logic_profile: str
    target_sensor: str
    active_loop: bool
    historical_memory: str
    execution_telemetry: List[str]

class AutonomousOrchestrator:
    """
    Main controller implementing the recursive ReAct pattern.
    Designed for adaptive signal processing in non-stationary environments.
    """
    
    def __init__(self, state: SystemState):
        self.state = state

    def environmental_analysis_node(self) -> None:
        """Extracts regime-specific context from raw telemetry data."""
        from src.features.engine import compute_features
        from src.features.regime import detect_regime

        data = self.state["raw_telemetry"]
        sensor = self.state.get("target_sensor", config.reference_asset)

        # Signal Processing & Feature Engineering
        features = compute_features(data, sensor, config.vix_ticker)
        ctx = build_context(features)
        regime_identifier = detect_regime(features)
        ctx.regime_label = regime_identifier

        # Retrieval of historical state-mappings
        memory_engine = StrategyMemory()
        historical_ctx = memory_engine.to_prompt_context(regime_identifier, self.state.get("logic_profile", "momentum"))

        # Update System State
        self.state["processed_features"] = features
        self.state["environmental_context"] = ctx
        self.state["historical_memory"] = historical_ctx
        self.state["execution_telemetry"] = self.state.get("execution_telemetry", [])
        self.state["execution_telemetry"].append(f"Regime Identified: {regime_identifier}")
        
        logger.info(f"Phase 1 Complete: System operating in {regime_identifier} regime.")

    def hypothesis_synthesis_node(self) -> None:
        """Synthesizes new parameter sets based on environmental feedback."""
        cycle = self.state.get("current_cycle", 0) + 1
        self.state["current_cycle"] = cycle
        
        generator = ProposalGenerator()
        profile = self.state.get("logic_profile", "momentum")
        
        # Generation of N-candidate solutions
        proposals = generator.generate(
            context=self.state["environmental_context"],
            n_proposals=5,
            strategy_type=profile,
        )

        self.state["candidate_proposals"] = proposals
        self.state["execution_telemetry"].append(f"Cycle {cycle}: Synthesized {len(proposals)} candidates.")

    def validation_engine_node(self) -> None:
        """Stress-tests candidate proposals against historical non-stationarity."""
        from src.backtest.runner import run_backtest

        data = self.state["raw_telemetry"]
        sensor = self.state.get("target_sensor", config.reference_asset)
        profile = self.state.get("logic_profile", "momentum")

        benchmarks = []
        for i, candidate in enumerate(self.state["candidate_proposals"]):
            try:
                # Execution of the validation runner
                sim_res = run_backtest(data, [sensor], profile, candidate.params)
                if sim_res and "metrics" in sim_res:
                    m = sim_res["metrics"]
                    benchmarks.append({
                        "id": i,
                        "configuration": candidate.params,
                        "sharpe_index": m.get("sharpe_ratio", 0.0),
                        "efficiency_metric": m.get("total_return", 0.0),
                        "risk_profile": m.get("max_drawdown", 0.0),
                        "methodology": candidate.generation_method
                    })
            except Exception as e:
                logger.error(f"Validation Failure on candidate {i}: {str(e)}")

        # Rank solutions by Sharpe Index (Stability/Risk Ratio)
        benchmarks.sort(key=lambda x: x.get("sharpe_index", 0.0), reverse=True)
        self.state["performance_benchmarks"] = benchmarks
        
        if benchmarks:
            self.state["optimal_config"] = benchmarks[0]

    def recursive_reflection_node(self) -> None:
        """Decision-making gate: Accept solution or trigger recursive recalibration."""
        opt = self.state.get("optimal_config")
        cycle = self.state.get("current_cycle", 1)
        limit = self.state.get("loop_limit", config.agent.max_iterations)
        threshold = config.agent.min_acceptable_sharpe

        if opt is None:
            self.state["active_loop"] = cycle < limit
            return

        current_performance = opt.get("sharpe_index", 0.0)

        # Stability Convergence Logic
        if current_performance >= threshold:
            self.state["active_loop"] = False
            self.state["execution_telemetry"].append("Stability threshold reached. Solution accepted.")
        elif cycle >= limit:
            self.state["active_loop"] = False
            self.state["execution_telemetry"].append("Loop limit reached. Deploying best available configuration.")
        else:
            self.state["active_loop"] = True
            logger.info(f"Performance {current_performance:.2f} < {threshold}. Re-initializing synthesis cycle.")

    def persistence_node(self) -> None:
        """Commits the optimal state-mapping to the SQLite Knowledge Store."""
        opt = self.state.get("optimal_config")
        if not opt: return

        ctx = self.state.get("environmental_context")
        regime = ctx.regime_label if ctx else "Stochastic"

        db_engine = StrategyMemory()
        record = PastResult(
            regime=regime,
            strategy_type=self.state.get("logic_profile", "momentum"),
            params=json.dumps(opt["configuration"]),
            sharpe=opt.get("sharpe_index", 0.0),
            total_return=opt.get("efficiency_metric", 0.0),
            max_drawdown=opt.get("risk_profile", 0.0),
            generation_method=opt.get("methodology", "")
        )
        db_engine.store(record)

def execute_orchestration(telemetry_data: Dict[str, pd.DataFrame]) -> SystemState:
    """Entry point for the autonomous controller."""
    
    # Initialize the State-Space
    initial_state: SystemState = {
        "raw_telemetry": telemetry_data,
        "processed_features": pd.DataFrame(),
        "current_cycle": 0,
        "loop_limit": config.agent.max_iterations,
        "active_loop": True,
        "execution_telemetry": [],
    }

    controller = AutonomousOrchestrator(initial_state)

    # Phase 1: Analysis
    controller.environmental_analysis_node()

    # Phases 2-4: The Recursive Feedback Loop
    while controller.state["active_loop"] and controller.state["current_cycle"] < controller.state["loop_limit"]:
        controller.hypothesis_synthesis_node()
        controller.validation_engine_node()
        controller.recursive_reflection_node()

    # Phase 5: Knowledge Persistence
    controller.persistence_node()

    return controller.state
