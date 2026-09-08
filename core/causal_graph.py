"""Structural Causal Model (SCM) and Directed Acyclic Graph (DAG) Engine.

Implements Pearl's Ladder of Causation:
1. Association (Observational state)
2. Intervention (Pearl's do-operator on mutilated graphs G_{/X})
3. Counterfactuals (Abduction -> Action -> Prediction)
"""

from __future__ import annotations

from collections import defaultdict, deque
import copy
from enum import Enum
from typing import Any, Callable
from pydantic import BaseModel, Field


class CausalNodeType(str, Enum):
    EXOGENOUS = "exogenous"      # Unobserved background variable (U)
    ENDOGENOUS = "endogenous"    # System variable determined by internal mechanisms (V)
    INTERVENTION = "intervention" # Action node targeted by do(X)
    OUTCOME = "outcome"          # Final clinical or operational consequence (Y)


class CausalNode(BaseModel):
    """Node in the Structural Causal Model."""

    name: str
    description: str = ""
    node_type: CausalNodeType = CausalNodeType.ENDOGENOUS
    observed_value: Any = None
    baseline_value: Any = None
    possible_values: list[Any] = Field(default_factory=list)


class CausalEdge(BaseModel):
    """Directed causal mechanism from source to target."""

    source: str
    target: str
    mechanism_description: str = ""
    weight: float = 1.0
    inhibitor_nodes: list[str] = Field(default_factory=list)
    is_confounded: bool = False


class StructuralCausalModel:
    """Directed Acyclic Graph representing causal equations and interventions."""

    def __init__(self, name: str = "default_scm") -> None:
        self.name = name
        self.nodes: dict[str, CausalNode] = {}
        self.edges: dict[str, list[CausalEdge]] = defaultdict(list)
        self.incoming_edges: dict[str, list[CausalEdge]] = defaultdict(list)
        self.mechanisms: dict[str, Callable[[dict[str, Any], Any], Any]] = {}

    def add_node(
        self,
        name: str,
        description: str = "",
        node_type: CausalNodeType = CausalNodeType.ENDOGENOUS,
        baseline_value: Any = None,
        possible_values: list[Any] | None = None,
    ) -> CausalNode:
        """Add a variable to the causal model."""
        node = CausalNode(
            name=name,
            description=description,
            node_type=node_type,
            observed_value=baseline_value,
            baseline_value=baseline_value,
            possible_values=possible_values or [],
        )
        self.nodes[name] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        mechanism_description: str = "",
        weight: float = 1.0,
        inhibitor_nodes: list[str] | None = None,
    ) -> CausalEdge:
        """Add a directed causal mechanism X -> Y."""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Both source '{source}' and target '{target}' must exist in the model.")

        edge = CausalEdge(
            source=source,
            target=target,
            mechanism_description=mechanism_description,
            weight=weight,
            inhibitor_nodes=inhibitor_nodes or [],
        )
        self.edges[source].append(edge)
        self.incoming_edges[target].append(edge)

        # Enforce acyclicity
        if self._has_cycle():
            self.edges[source].remove(edge)
            self.incoming_edges[target].remove(edge)
            raise ValueError(f"Adding edge {source} -> {target} creates a cycle. SCM must remain a DAG.")

        return edge

    def register_mechanism(self, target_node: str, func: Callable[[dict[str, Any], Any], Any]) -> None:
        """Define the structural equation f_i(PA_i, U_i) for a variable."""
        if target_node not in self.nodes:
            raise ValueError(f"Node '{target_node}' does not exist.")
        self.mechanisms[target_node] = func

    def _has_cycle(self) -> bool:
        """Check for cycles using Kahn's algorithm."""
        in_degree = {name: len(self.incoming_edges[name]) for name in self.nodes}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node_name = queue.popleft()
            visited_count += 1
            for edge in self.edges[node_name]:
                target = edge.target
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        return visited_count != len(self.nodes)

    def topological_sort(self) -> list[str]:
        """Return nodes in topological causal order."""
        in_degree = {name: len(self.incoming_edges[name]) for name in self.nodes}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for edge in self.edges[curr]:
                target = edge.target
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        return order

    def set_evidence(self, observations: dict[str, Any]) -> None:
        """Record observed factual evidence across variables."""
        for var, val in observations.items():
            if var in self.nodes:
                self.nodes[var].observed_value = val

    def forward_simulate(self) -> dict[str, Any]:
        """Compute state values by evaluating structural equations in topological order."""
        values: dict[str, Any] = {name: node.observed_value for name, node in self.nodes.items()}
        order = self.topological_sort()

        for node_name in order:
            if node_name in self.mechanisms:
                parents_values = {
                    edge.source: values.get(edge.source) for edge in self.incoming_edges[node_name]
                }
                # Check for active inhibitors
                inhibited = False
                for edge in self.incoming_edges[node_name]:
                    for inh in edge.inhibitor_nodes:
                        if values.get(inh):
                            inhibited = True
                            break

                if not inhibited:
                    values[node_name] = self.mechanisms[node_name](parents_values, values.get(node_name))

        return values

    def do_intervention(self, interventions: dict[str, Any]) -> StructuralCausalModel:
        """Pearl's Level 2: Apply the do(X=x) operator.

        Constructs the mutilated graph G_{/X} where all incoming arrows into X
        are severed, setting X directly to value x.
        """
        mutilated = copy.deepcopy(self)

        for target_var, fixed_val in interventions.items():
            if target_var not in mutilated.nodes:
                continue

            # Sever incoming causal edges into target_var
            incoming = list(mutilated.incoming_edges[target_var])
            for edge in incoming:
                mutilated.edges[edge.source] = [e for e in mutilated.edges[edge.source] if e.target != target_var]
            mutilated.incoming_edges[target_var] = []

            # Remove structural equation for target_var (now fixed by external intervention)
            if target_var in mutilated.mechanisms:
                del mutilated.mechanisms[target_var]

            # Set node value and type
            mutilated.nodes[target_var].observed_value = fixed_val
            mutilated.nodes[target_var].node_type = CausalNodeType.INTERVENTION

        return mutilated

    def counterfactual_analysis(
        self,
        factual_evidence: dict[str, Any],
        hypothetical_intervention: dict[str, Any],
        target_outcome: str,
    ) -> dict[str, Any]:
        """Pearl's Level 3: Three-step counterfactual computation.

        1. Abduction: Calibrate exogenous noise given factual evidence E=e.
        2. Action: Apply hypothetical intervention do(X=x') on mutilated graph G_{/X}.
        3. Prediction: Re-evaluate target outcome in the modified world.
        """
        # 1. Abduction: Set observed factual evidence
        self.set_evidence(factual_evidence)
        factual_state = self.forward_simulate()
        factual_outcome = factual_state.get(target_outcome)

        # 2. Action: Mutilate graph according to the hypothetical alternative
        counterfactual_scm = self.do_intervention(hypothetical_intervention)

        # Retain exogenous background variables from the factual world
        for name, node in self.nodes.items():
            if node.node_type == CausalNodeType.EXOGENOUS:
                counterfactual_scm.nodes[name].observed_value = factual_state.get(name)

        # 3. Prediction: Simulate counterfactual world
        counterfactual_state = counterfactual_scm.forward_simulate()
        counterfactual_outcome = counterfactual_state.get(target_outcome)

        # Causal attribution: Was the intervention necessary to change the outcome?
        causal_shift_detected = counterfactual_outcome != factual_outcome

        return {
            "target_outcome": target_outcome,
            "factual_outcome": factual_outcome,
            "counterfactual_outcome": counterfactual_outcome,
            "causal_shift_detected": causal_shift_detected,
            "factual_state": factual_state,
            "counterfactual_state": counterfactual_state,
            "interventions": hypothetical_intervention,
        }

    def find_backdoor_paths(self, treatment: str, outcome: str) -> list[list[str]]:
        """Identify potential backdoor confounding paths between treatment and outcome."""
        backdoor_paths: list[list[str]] = []

        def dfs_backdoor(curr: str, path: list[str], is_first_step: bool) -> None:
            if curr == outcome and len(path) > 2:
                backdoor_paths.append(list(path))
                return

            if is_first_step:
                # First step of backdoor path MUST be an incoming edge into treatment (parent of treatment)
                for in_edge in self.incoming_edges[curr]:
                    parent = in_edge.source
                    if parent not in path:
                        path.append(parent)
                        dfs_backdoor(parent, path, is_first_step=False)
                        path.pop()
            else:
                # Subsequent steps can traverse in any direction across the moralized graph
                for out_edge in self.edges[curr]:
                    nxt = out_edge.target
                    if nxt not in path:
                        path.append(nxt)
                        dfs_backdoor(nxt, path, is_first_step=False)
                        path.pop()
                for in_edge in self.incoming_edges[curr]:
                    nxt = in_edge.source
                    if nxt not in path:
                        path.append(nxt)
                        dfs_backdoor(nxt, path, is_first_step=False)
                        path.pop()

        dfs_backdoor(treatment, [treatment], is_first_step=True)
        return backdoor_paths
