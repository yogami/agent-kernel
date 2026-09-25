"""Structural Causal Model (SCM) and Directed Acyclic Graph (DAG) Engine.

Implements Pearl's Ladder of Causation:
1. Association (Observational state)
2. Intervention (Pearl's do-operator on mutilated graphs G_{/X})
3. Counterfactuals (Abduction -> Action -> Prediction)
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
import copy
from enum import Enum
import itertools
import json
import operator
import os
from typing import Any, Callable
import xml.etree.ElementTree as ET
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
        self.formula_mechanisms: dict[str, str] = {}

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

    def register_formula_mechanism(self, target_node: str, formula_str: str) -> None:
        """Define a mathematical or logical formula string for a variable's mechanism."""
        if target_node not in self.nodes:
            raise ValueError(f"Node '{target_node}' does not exist.")
        self.formula_mechanisms[target_node] = formula_str

        def formula_func(parents: dict[str, Any], curr: Any) -> Any:
            return self._eval_formula(formula_str, parents, curr)

        self.mechanisms[target_node] = formula_func

    @staticmethod
    def _eval_formula(expr: str, parents: dict[str, Any], current: Any = None) -> Any:
        """Evaluate formula expression safely using an AST visitor with variable mappings."""
        allowed_builtins = {
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "bool": bool,
            "float": float,
            "int": int,
            "str": str,
        }
        context = dict(allowed_builtins)
        context.update(parents)
        context["curr"] = current

        safe_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Not: operator.not_,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }

        def _evaluate_node(node: ast.AST) -> Any:
            if isinstance(node, ast.Expression):
                return _evaluate_node(node.body)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Name):
                if node.id in context:
                    return context[node.id]
                raise ValueError(f"Undefined variable in causal formula: {node.id}")
            elif isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type in safe_operators:
                    return safe_operators[op_type](_evaluate_node(node.operand))
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type in safe_operators:
                    left = _evaluate_node(node.left)
                    right = _evaluate_node(node.right)
                    return safe_operators[op_type](left, right)
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            elif isinstance(node, ast.Compare):
                left = _evaluate_node(node.left)
                for op, comp in zip(node.ops, node.comparators):
                    op_type = type(op)
                    if op_type not in safe_operators:
                        raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
                    right = _evaluate_node(comp)
                    if not safe_operators[op_type](left, right):
                        return False
                    left = right
                return True
            elif isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    for val in node.values:
                        if not _evaluate_node(val):
                            return False
                    return True
                elif isinstance(node.op, ast.Or):
                    for val in node.values:
                        if _evaluate_node(val):
                            return True
                    return False
            elif isinstance(node, ast.IfExp):
                test_val = _evaluate_node(node.test)
                return _evaluate_node(node.body) if test_val else _evaluate_node(node.orelse)
            elif isinstance(node, ast.Call):
                func = _evaluate_node(node.func)
                if not callable(func):
                    raise ValueError(f"Target is not callable in formula: {func}")
                args = [_evaluate_node(a) for a in node.args]
                return func(*args)
            raise ValueError(f"Disallowed expression node in causal formula: {type(node).__name__}")

        parsed = ast.parse(expr, mode="eval")
        return _evaluate_node(parsed)

    @property
    def all_edges(self) -> list[CausalEdge]:
        """Return a flat list of all directed edges in the model."""
        return [e for edges in self.edges.values() for e in edges]

    def _has_cycle(self) -> bool:
        """Check for cycles using Kahn's algorithm."""
        in_degree = {name: len(self.incoming_edges.get(name, [])) for name in self.nodes}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            node_name = queue.popleft()
            visited_count += 1
            for edge in self.edges.get(node_name, []):
                target = edge.target
                in_degree[target] -= 1
                if in_degree[target] == 0:
                    queue.append(target)

        return visited_count != len(self.nodes)

    def topological_sort(self) -> list[str]:
        """Return nodes in topological causal order."""
        in_degree = {name: len(self.incoming_edges.get(name, [])) for name in self.nodes}
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for edge in self.edges.get(curr, []):
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

    def ancestors(self, nodes: set[str] | list[str] | str) -> set[str]:
        """Compute the set of all ancestors for given nodes including the nodes themselves."""
        if isinstance(nodes, str):
            nodes_set = {nodes}
        else:
            nodes_set = set(nodes)

        anc = set(nodes_set)
        queue = deque(list(nodes_set))
        while queue:
            curr = queue.popleft()
            for edge in self.incoming_edges.get(curr, []):
                parent = edge.source
                if parent not in anc:
                    anc.add(parent)
                    queue.append(parent)
        return anc

    def descendants(self, nodes: set[str] | list[str] | str) -> set[str]:
        """Compute the set of all descendants for given nodes including the nodes themselves."""
        if isinstance(nodes, str):
            nodes_set = {nodes}
        else:
            nodes_set = set(nodes)

        desc = set(nodes_set)
        queue = deque(list(nodes_set))
        while queue:
            curr = queue.popleft()
            for edge in self.edges.get(curr, []):
                child = edge.target
                if child not in desc:
                    desc.add(child)
                    queue.append(child)
        return desc

    def is_d_separated(
        self,
        x: set[str] | list[str] | str,
        y: set[str] | list[str] | str,
        z: set[str] | list[str] | str | None = None,
    ) -> bool:
        """Determine if sets X and Y are d-separated given conditioning set Z.

        Uses moralized ancestral graph reachability:
        1. Form ancestral subgraph for X union Y union Z.
        2. Moralize by marrying parents sharing a child.
        3. Convert directed edges to undirected connections.
        4. Remove conditioning set Z.
        5. Check if any path exists connecting X and Y.
        """
        set_x = {x} if isinstance(x, str) else set(x)
        set_y = {y} if isinstance(y, str) else set(y)
        set_z = set() if z is None else ({z} if isinstance(z, str) else set(z))

        if not set_x or not set_y:
            return True
        if not set_x.isdisjoint(set_y):
            return False

        # Ancestral graph nodes
        v_anc = self.ancestors(set_x | set_y | set_z)

        # Build moralized undirected adjacency
        adj: dict[str, set[str]] = defaultdict(set)
        for node in v_anc:
            parents = [e.source for e in self.incoming_edges.get(node, []) if e.source in v_anc]
            for p in parents:
                adj[p].add(node)
                adj[node].add(p)
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1, p2 = parents[i], parents[j]
                    adj[p1].add(p2)
                    adj[p2].add(p1)

        # Remove conditioning nodes
        for blocked in set_z:
            if blocked in adj:
                for nbr in adj[blocked]:
                    adj[nbr].discard(blocked)
                del adj[blocked]

        # Check reachability between (set_x - set_z) and (set_y - set_z)
        start_nodes = set_x - set_z
        target_nodes = set_y - set_z
        if not start_nodes or not target_nodes:
            return True

        visited: set[str] = set()
        queue = deque(list(start_nodes))
        visited.update(start_nodes)

        while queue:
            curr = queue.popleft()
            if curr in target_nodes:
                return False
            for nbr in adj.get(curr, set()):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        return True

    def find_implied_independencies(
        self,
        max_conditioning_size: int = 2,
    ) -> list[dict[str, Any]]:
        """Identify minimal testable conditional independencies implied by the DAG.

        Returns a list of dicts with var_x, var_y, and conditioning_set.
        """
        all_nodes = sorted(self.nodes.keys())
        independencies: list[dict[str, Any]] = []

        adjacent_pairs: set[tuple[str, str]] = set()
        for src, edges in self.edges.items():
            for e in edges:
                adjacent_pairs.add((src, e.target))
                adjacent_pairs.add((e.target, src))

        for i in range(len(all_nodes)):
            for j in range(i + 1, len(all_nodes)):
                u, v = all_nodes[i], all_nodes[j]
                if (u, v) in adjacent_pairs:
                    continue

                candidates = [n for n in all_nodes if n != u and n != v]
                found_minimal = False

                for size in range(max_conditioning_size + 1):
                    for combo in itertools.combinations(candidates, size):
                        z_set = set(combo)
                        if self.is_d_separated(u, v, z_set):
                            independencies.append({
                                "var_x": u,
                                "var_y": v,
                                "conditioning_set": sorted(combo),
                            })
                            found_minimal = True
                            break
                    if found_minimal:
                        break

        return independencies

    def to_dict(self) -> dict[str, Any]:
        """Serialize model definition to a structured dictionary."""
        return {
            "name": self.name,
            "nodes": [
                {
                    "name": n.name,
                    "description": n.description,
                    "node_type": n.node_type.value,
                    "baseline_value": n.baseline_value,
                    "possible_values": n.possible_values,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "mechanism_description": e.mechanism_description,
                    "weight": e.weight,
                    "inhibitor_nodes": e.inhibitor_nodes,
                    "is_confounded": e.is_confounded,
                }
                for edge_list in self.edges.values()
                for e in edge_list
            ],
            "formulas": dict(self.formula_mechanisms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuralCausalModel:
        """Construct StructuralCausalModel from dictionary."""
        name = data.get("name", "imported_scm")
        scm = cls(name=name)

        for n_data in data.get("nodes", []):
            nt_str = n_data.get("node_type", "endogenous")
            try:
                nt = CausalNodeType(nt_str)
            except ValueError:
                nt = CausalNodeType.ENDOGENOUS
            scm.add_node(
                name=n_data["name"],
                description=n_data.get("description", ""),
                node_type=nt,
                baseline_value=n_data.get("baseline_value"),
                possible_values=n_data.get("possible_values", []),
            )

        for e_data in data.get("edges", []):
            edge = scm.add_edge(
                source=e_data["source"],
                target=e_data["target"],
                mechanism_description=e_data.get("mechanism_description", ""),
                weight=float(e_data.get("weight", 1.0)),
                inhibitor_nodes=e_data.get("inhibitor_nodes", []),
            )
            if e_data.get("is_confounded"):
                edge.is_confounded = True

        for target_node, formula in data.get("formulas", {}).items():
            if target_node in scm.nodes:
                scm.register_formula_mechanism(target_node, formula)

        return scm

    def to_json(self, filepath: str | None = None, indent: int = 2) -> str:
        """Export SCM to JSON string and optionally write to file."""
        payload = self.to_dict()
        json_str = json.dumps(payload, indent=indent)
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str) -> StructuralCausalModel:
        """Load SCM from JSON string or file path."""
        if os.path.exists(json_str_or_path):
            with open(json_str_or_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = json_str_or_path
        data = json.loads(content)
        return cls.from_dict(data)

    def to_graphml(self, filepath: str | None = None) -> str:
        """Export SCM to standard GraphML XML string and optionally save to file."""
        root = ET.Element(
            "graphml",
            {
                "xmlns": "http://graphml.graphdrawing.org/xmlns",
                "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
                "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            },
        )

        keys = [
            ("description", "node", "string"),
            ("node_type", "node", "string"),
            ("baseline_value", "node", "string"),
            ("formula", "node", "string"),
            ("mechanism_description", "edge", "string"),
            ("weight", "edge", "double"),
            ("inhibitor_nodes", "edge", "string"),
            ("is_confounded", "edge", "boolean"),
        ]
        for k_id, k_for, k_type in keys:
            ET.SubElement(root, "key", {"id": k_id, "for": k_for, "attr.name": k_id, "attr.type": k_type})

        graph = ET.SubElement(root, "graph", {"id": self.name, "edgedefault": "directed"})

        for node in self.nodes.values():
            n_elem = ET.SubElement(graph, "node", {"id": node.name})
            if node.description:
                d = ET.SubElement(n_elem, "data", {"key": "description"})
                d.text = node.description
            d_type = ET.SubElement(n_elem, "data", {"key": "node_type"})
            d_type.text = node.node_type.value
            if node.baseline_value is not None:
                d_base = ET.SubElement(n_elem, "data", {"key": "baseline_value"})
                d_base.text = str(node.baseline_value)
            if node.name in self.formula_mechanisms:
                d_form = ET.SubElement(n_elem, "data", {"key": "formula"})
                d_form.text = self.formula_mechanisms[node.name]

        for edges in self.edges.values():
            for edge in edges:
                e_elem = ET.SubElement(graph, "edge", {"source": edge.source, "target": edge.target})
                if edge.mechanism_description:
                    d_mech = ET.SubElement(e_elem, "data", {"key": "mechanism_description"})
                    d_mech.text = edge.mechanism_description
                d_w = ET.SubElement(e_elem, "data", {"key": "weight"})
                d_w.text = str(edge.weight)
                if edge.inhibitor_nodes:
                    d_inh = ET.SubElement(e_elem, "data", {"key": "inhibitor_nodes"})
                    d_inh.text = ",".join(edge.inhibitor_nodes)
                if edge.is_confounded:
                    d_conf = ET.SubElement(e_elem, "data", {"key": "is_confounded"})
                    d_conf.text = "true"

        ET.indent(root)
        xml_str = ET.tostring(root, encoding="utf-8").decode("utf-8")
        if filepath:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(xml_str)
        return xml_str

    @classmethod
    def from_graphml(cls, xml_str_or_path: str) -> StructuralCausalModel:
        """Load SCM from GraphML XML string or file path."""
        if os.path.exists(xml_str_or_path):
            with open(xml_str_or_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = xml_str_or_path

        root = ET.fromstring(content)
        graph_elem = None
        for elem in root.iter():
            if elem.tag.split("}")[-1] == "graph":
                graph_elem = elem
                break

        name = graph_elem.get("id", "imported_graphml") if graph_elem is not None else "imported_graphml"
        scm = cls(name=name)

        if graph_elem is None:
            return scm

        for elem in graph_elem:
            tag = elem.tag.split("}")[-1]
            if tag == "node":
                node_id = elem.get("id")
                if not node_id:
                    continue
                node_type_str = "endogenous"
                description = ""
                baseline_val = None
                formula_str = None

                for child in elem:
                    if child.tag.split("}")[-1] == "data":
                        k = child.get("key")
                        val = child.text or ""
                        if k == "node_type":
                            node_type_str = val
                        elif k == "description":
                            description = val
                        elif k == "baseline_value":
                            baseline_val = val
                        elif k == "formula":
                            formula_str = val

                try:
                    nt = CausalNodeType(node_type_str)
                except ValueError:
                    nt = CausalNodeType.ENDOGENOUS

                scm.add_node(
                    name=node_id,
                    description=description,
                    node_type=nt,
                    baseline_value=baseline_val,
                )
                if formula_str:
                    scm.register_formula_mechanism(node_id, formula_str)

            elif tag == "edge":
                src = elem.get("source")
                tgt = elem.get("target")
                if not src or not tgt:
                    continue
                mech_desc = ""
                weight = 1.0
                inhibitors: list[str] = []
                is_confounded = False

                for child in elem:
                    if child.tag.split("}")[-1] == "data":
                        k = child.get("key")
                        val = child.text or ""
                        if k == "mechanism_description":
                            mech_desc = val
                        elif k == "weight":
                            try:
                                weight = float(val)
                            except ValueError:
                                weight = 1.0
                        elif k == "inhibitor_nodes":
                            inhibitors = [s.strip() for s in val.split(",") if s.strip()]
                        elif k == "is_confounded":
                            is_confounded = val.lower() == "true"

                edge = scm.add_edge(
                    source=src,
                    target=tgt,
                    mechanism_description=mech_desc,
                    weight=weight,
                    inhibitor_nodes=inhibitors,
                )
                if is_confounded:
                    edge.is_confounded = True

        return scm
