"""Three-path retrieval routing. Importing the package never loads a model."""
from .core import ROUTES, FixedRouter, RuleRouter, RouteDecision, RoutingRequest

__all__ = ["ROUTES", "FixedRouter", "RuleRouter", "RouteDecision", "RoutingRequest"]
