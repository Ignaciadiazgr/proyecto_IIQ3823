"""
SteadyCom for microbial communities -- cobrapy front end, Gurobi back end.

This is a Python port of the COBRA Toolbox implementation of SteadyCom
(``SteadyCom.m`` / ``SteadyComFVA.m``), following the same LP formulation and
the same root-finding strategy.

Reference
---------
Chan SHJ, Simons MN, Maranas CD (2017) "SteadyCom: Predicting microbial
abundances while ensuring community stability." PLoS Comput Biol 13(5):
e1005539.

The model
---------
Community members are merged into a single network that shares one
extracellular compartment.  Every organism-specific flux is expressed in
mmol/h (not mmol/gDW/h), i.e. it is already multiplied by the biomass
``X_k`` (gDW) of the organism that carries it.  The LP at a fixed community
growth rate ``mu`` is::

    max   sum_k X_k                          (total community biomass)
    s.t.  S v = 0                            (mass balance)
          v_j - ub_j * X_k <= 0              (flux bounded by biomass)
          v_j - lb_j * X_k >= 0
          v_bio,k - mu * X_k  = 0            (every member grows at mu)
          lb_ex <= v_ex <= ub_ex             (community uptake/export, absolute)
          0 <= X_k <= biomass_ub

``max sum X_k`` is a strictly decreasing function of ``mu``.  The maximum
community growth rate is the ``mu`` at which it drops to the reference total
biomass ``bm_weight`` (1 gDW by default), and it is located by root finding,
not by testing raw LP feasibility.  That is the key numerical difference from
a naive implementation: LP feasibility flips erratically near the optimum,
whereas ``sum X_k`` is smooth and gives ~1e-6 accuracy on ``mu``.

Public API
----------
``build_steadycom_model``  build a ``CommunityModel`` from cobrapy models
``SteadyCom``              maximum community growth rate + fluxes + abundances
``SteadyComVA``            abundance variability at a fraction of the optimum
``SteadyComFVA``           flux variability at a fraction of the optimum
``detect_biomass_reaction``
"""

from __future__ import annotations

import math
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cobra
from cobra import Metabolite, Model, Reaction

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "steadycom requires gurobipy. Install it with `pip install gurobipy` "
        "(a free size-limited licence covers models of this size)."
    ) from exc


__all__ = [
    "Status",
    "SteadyComParams",
    "SteadyComSolution",
    "Solution",
    "Organism",
    "CommunityModel",
    "Community",
    "SteadyComProblem",
    "build_steadycom_model",
    "build_problem",
    "SteadyCom",
    "SteadyComVA",
    "SteadyComFVA",
    "detect_biomass_reaction",
]

INF = float("inf")

# Fallback names for the extracellular compartment, used only when it cannot be
# inferred from the boundary reactions of a model.
_EXTERNAL_HINTS = {"e", "e0", "ext", "extracellular", "exo", "u", "out"}


# =============================================================================
# Status / parameters
# =============================================================================

class Status(IntEnum):
    """Solver status. Values kept identical to the previous implementation."""

    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    OTHER = 4


@dataclass
class SteadyComParams:
    """Algorithm parameters. Names mirror the MATLAB ``options`` struct.

    Attributes
    ----------
    gr_guess
        Initial guess for the community growth rate (MATLAB ``GRguess``).
    gr_tol
        Absolute precision required on the growth rate (MATLAB ``GRtol``).
        The previous implementation effectively used 1e-3.
    gr_ub_cap
        Safety cap while bracketing. Growth rates above this are treated as a
        modelling error (unbounded community) rather than searched for.
    bm_weight
        Reference total community biomass in gDW (MATLAB ``BMweight``).  With
        the default of 1.0 every reported flux is per gDW of community.
        Set it close to zero to compute the wash-out dilution rate.
    bm_tol_abs
        Below this total biomass the community is considered dead
        (MATLAB ``BMtolAbs``).
    biomass_ub
        Upper bound on each organism's biomass variable (MATLAB uses 1000).
    max_iter
        Maximum number of LP solves in the growth-rate search.
    feas_tol
        Gurobi ``FeasibilityTol``/``OptimalityTol``.
    min_norm
        Minimise the L1 norm of the fluxes at the optimum (MATLAB
        ``minNorm``).  Without it the flux distribution is an arbitrary vertex
        among the alternate optima, so exchange fluxes are not reproducible.
    normalize
        Report fluxes and abundances per gDW of community, i.e. divided by the
        total biomass actually achieved.
    verbose
        Print the iterations of the growth-rate search.
    threads
        Gurobi thread count (0 = automatic).
    """

    gr_guess: float = 0.2
    gr_tol: float = 1e-6
    gr_ub_cap: float = 1e3
    bm_weight: float = 1.0
    bm_tol_abs: float = 1e-5
    biomass_ub: float = 1000.0
    max_iter: int = 1000
    feas_tol: float = 1e-8
    min_norm: bool = True
    normalize: bool = True
    verbose: bool = False
    threads: int = 0


# =============================================================================
# Solution
# =============================================================================

class SteadyComSolution:
    """Result of a SteadyCom simulation.

    Attributes
    ----------
    status
        :class:`Status` of the final LP.
    growth_rate
        Maximum community specific growth rate (1/h).  Also exposed as
        ``fobj`` for backwards compatibility.
    stat
        Human-readable outcome: ``'optimal'``, ``'maintenance'`` (the community
        is viable but cannot grow), or ``'infeasible'``.
    abundance
        ``{organism_id: relative abundance}``, summing to 1.
    biomass
        ``{organism_id: X_k}`` in gDW.
    fluxes
        ``{merged_reaction_id: flux}``.
    values
        Backwards-compatible flat dict: every entry of ``fluxes`` plus
        ``x_<org>`` (relative abundance), ``X_<org>`` (gDW) and
        ``community_growth`` (the growth rate).
    iterations
        ``[(mu, total_biomass), ...]`` for every LP solved during the search.
    """

    def __init__(self, community, status, growth_rate=None, stat="",
                 fluxes=None, biomass=None, total_biomass=None, iterations=None):
        self.community = community
        self.status = status
        self.growth_rate = growth_rate
        self.stat = stat
        self.fluxes = fluxes or {}
        self.biomass = biomass or {}
        self.total_biomass = total_biomass
        self.iterations = iterations or []

        self.abundance = {}
        if self.biomass:
            total = sum(self.biomass.values())
            if total > 0:
                self.abundance = {k: v / total for k, v in self.biomass.items()}
            else:
                self.abundance = {k: 0.0 for k in self.biomass}

        self.values = dict(self.fluxes)
        for org, x in self.abundance.items():
            self.values[f"x_{org}"] = x
        for org, X in self.biomass.items():
            self.values[f"X_{org}"] = X
        if growth_rate is not None:
            self.values[getattr(community, "biomass_reaction_id", "community_growth")] = growth_rate

    # -- backwards compatibility ------------------------------------------
    @property
    def fobj(self):
        """The community growth rate (previous name for it)."""
        return self.growth_rate

    def organism_fluxes(self, organism_id: str) -> Dict[str, float]:
        """Fluxes of one organism keyed by its *original* reaction ids."""
        out = {}
        for merged_id in self.community.organism_reactions[organism_id]:
            original = self.community.original_reaction[merged_id]
            out[original] = self.fluxes.get(merged_id, 0.0)
        return out

    def exchange_fluxes(self, tol: float = 1e-9) -> Dict[str, float]:
        """Non-zero community exchange fluxes (negative = uptake)."""
        return {
            rid: self.fluxes[rid]
            for rid in self.community.exchange_reactions
            if abs(self.fluxes.get(rid, 0.0)) > tol
        }

    def cross_feeding(self, tol: float = 1e-9) -> Dict[str, Dict[str, float]]:
        """Net production (+) / consumption (-) of each shared metabolite,
        per organism.  Only metabolites exchanged by at least two members with
        opposite signs are reported, i.e. actual cross-feeding."""
        merged = self.community.merged_model
        table: Dict[str, Dict[str, float]] = {}
        for met_id in self.community.exchange_reactions.values():
            met = merged.metabolites.get_by_id(met_id)
            per_org: Dict[str, float] = {}
            for rxn in met.reactions:
                org = self.community.reaction_owner.get(rxn.id)
                if org is None:
                    continue
                per_org[org] = per_org.get(org, 0.0) + rxn.metabolites[met] * self.fluxes.get(rxn.id, 0.0)
            per_org = {k: v for k, v in per_org.items() if abs(v) > tol}
            produced = max(per_org.values(), default=0.0) > tol
            consumed = min(per_org.values(), default=0.0) < -tol
            if len(per_org) > 1 and produced and consumed:
                table[met_id] = per_org
        return table

    def summary(self) -> str:
        lines = [f"status          : {self.status.name} ({self.stat})"]
        if self.growth_rate is not None:
            lines.append(f"growth rate     : {self.growth_rate:.6f} 1/h")
        if self.total_biomass is not None:
            lines.append(f"total biomass   : {self.total_biomass:.6f} gDW")
        lines.append("relative abundance:")
        for org, x in self.abundance.items():
            lines.append(f"    {org:<20s} {x:.4f}")
        ex = self.exchange_fluxes(1e-6)
        if ex:
            lines.append("community exchanges (negative = uptake):")
            for rid, val in sorted(ex.items(), key=lambda kv: kv[1]):
                lines.append(f"    {rid:<20s} {val: .4f}")
        return "\n".join(l for l in lines if l)

    def __repr__(self):
        gr = "None" if self.growth_rate is None else f"{self.growth_rate:.6f}"
        return f"<SteadyComSolution {self.status.name} mu={gr}>"


#: Backwards-compatible alias.
Solution = SteadyComSolution


# =============================================================================
# Model inspection helpers (no hard-coded ids)
# =============================================================================

def _external_compartment(model: cobra.Model, hint: Optional[str] = None) -> str:
    """Infer the extracellular compartment of a model.

    The compartment carrying most boundary reactions wins; this works for any
    naming convention ('e', 'e0', 'C_e', ...).  Name hints are only a fallback
    for models without boundary reactions.
    """
    if hint is not None:
        if hint not in (model.compartments or {}):
            raise ValueError(
                f"'{hint}' is not a compartment of model '{model.id}'. "
                f"Available: {sorted(model.compartments or {})}"
            )
        return hint

    counts: Counter = Counter()
    for rxn in model.boundary:
        for met in rxn.metabolites:
            counts[met.compartment] += 1
    if counts:
        return counts.most_common(1)[0][0]

    for cid, name in (model.compartments or {}).items():
        stem = cid.lower()
        if stem.startswith("c_"):  # SBML-style 'C_e'
            stem = stem[2:]
        if stem in _EXTERNAL_HINTS or "extracell" in str(name).lower():
            return cid

    raise ValueError(
        f"Could not infer the extracellular compartment of model '{model.id}'. "
        f"Pass it explicitly via `external_compartments={{'<org_id>': '<compartment>'}}`."
    )


def detect_biomass_reaction(model: cobra.Model) -> str:
    """Detect the biomass reaction of a cobrapy model.

    1. The reaction carrying a positive objective coefficient (the usual case).
    2. Otherwise a non-boundary reaction whose id or name mentions 'biomass'
       or 'growth'.

    Unlike the previous version this reads ``Reaction.objective_coefficient``
    instead of parsing the objective expression as a string, so reaction ids
    that happen to contain ``R_`` are no longer mangled.
    """
    objective_terms = {}
    for rxn in model.reactions:
        try:
            coeff = rxn.objective_coefficient
        except Exception:  # non-linear objective
            coeff = 0.0
        if coeff and coeff > 0:
            objective_terms[rxn.id] = coeff
    if len(objective_terms) == 1:
        return next(iter(objective_terms))
    if objective_terms:
        best = max(objective_terms.items(), key=lambda kv: kv[1])[0]
        warnings.warn(
            f"Model '{model.id}' has {len(objective_terms)} reactions in the "
            f"objective; using '{best}' as the biomass reaction. Pass "
            f"`biomass_reactions` explicitly to override.",
            stacklevel=2,
        )
        return best

    for keyword in ("biomass", "growth"):
        for rxn in model.reactions:
            if rxn.boundary:
                continue
            if keyword in rxn.id.lower() or keyword in (rxn.name or "").lower():
                return rxn.id

    raise ValueError(
        f"Could not detect the biomass reaction of model '{model.id}'. "
        f"Set `model.objective` or pass `biomass_reactions={{'<org_id>': '<rxn_id>'}}`."
    )


def _boundary_bounds(rxn: cobra.Reaction, met: cobra.Metabolite) -> Tuple[float, float]:
    """Bounds of a boundary reaction in the ``met -->`` convention.

    Reactions written as ``--> met`` are flipped so that a negative flux always
    means uptake.  The previous implementation ignored the stoichiometric
    coefficient and therefore inverted the medium for such models.
    """
    lb, ub = float(rxn.lower_bound), float(rxn.upper_bound)
    if rxn.metabolites[met] < 0:
        return lb, ub
    return -ub, -lb


# =============================================================================
# Community model
# =============================================================================

@dataclass
class Organism:
    """One community member."""

    id: str
    model: cobra.Model
    biomass_reaction: str
    external_compartment: str

    @property
    def reactions(self) -> Dict[str, cobra.Reaction]:
        return {rxn.id: rxn for rxn in self.model.reactions}


class CommunityModel:
    """A community of cobrapy models sharing one extracellular compartment.

    Parameters
    ----------
    community_id
        Name of the community.
    organism_models
        ``{organism_id: cobra.Model}``.
    biomass_reactions
        Optional ``{organism_id: reaction_id}`` overriding the automatic
        detection.
    external_compartments
        Optional ``{organism_id: compartment_id}`` overriding the automatic
        detection.
    id_separator
        Separator used when suffixing organism-specific ids.
    exchange_policy
        How the bounds of a community exchange reaction are derived from the
        individual models: ``'union'`` (default, widest bounds -- a nutrient
        available to one member is available to the community),
        ``'intersection'`` (tightest bounds) or ``'closed'`` (everything shut,
        the medium must be given explicitly).  Metabolites that no member
        exchanges with the environment are always closed, so they can only be
        cross-fed.
    mu_init
        Kept for backwards compatibility; used as the initial guess of the
        growth-rate search when no explicit guess is given.
    """

    def __init__(
        self,
        community_id: str,
        organism_models: Mapping[str, cobra.Model],
        biomass_reactions: Optional[Mapping[str, str]] = None,
        external_compartments: Optional[Mapping[str, str]] = None,
        id_separator: str = "_",
        exchange_policy: str = "union",
        mu_init: float = 0.2,
        copy_gene_rules: bool = False,
    ):
        if exchange_policy not in ("union", "intersection", "closed"):
            raise ValueError("exchange_policy must be 'union', 'intersection' or 'closed'")

        self.id = community_id
        self.id_separator = id_separator
        self.exchange_policy = exchange_policy
        self.mu_init = mu_init
        self.copy_gene_rules = copy_gene_rules

        self.organisms: "OrderedDict[str, Organism]" = OrderedDict()
        for org_id, mdl in organism_models.items():
            if not isinstance(mdl, cobra.Model):
                raise TypeError(f"organism '{org_id}' is not a cobra.Model")
            ext = _external_compartment(mdl, (external_compartments or {}).get(org_id))
            bio = (biomass_reactions or {}).get(org_id) or detect_biomass_reaction(mdl)
            if bio not in {r.id for r in mdl.reactions}:
                raise ValueError(f"'{bio}' is not a reaction of model '{mdl.id}'")
            self.organisms[org_id] = Organism(org_id, mdl, bio, ext)

        # -- lookup tables filled in by _merge() ---------------------------
        self.reaction_map: Dict[Tuple[str, str], str] = {}
        self.metabolite_map: Dict[Tuple[str, str], str] = {}
        self.reaction_owner: Dict[str, str] = {}
        self.original_reaction: Dict[str, str] = {}
        self.organism_reactions: Dict[str, List[str]] = {}
        self.biomass_reactions: Dict[str, str] = {}
        self.exchange_reactions: "OrderedDict[str, str]" = OrderedDict()
        self.external_compartment: str = ""

        # Name kept so that `sol.values[community.biomass_reaction_id]` still
        # returns the community growth rate.
        self.biomass_reaction_id = "community_growth"

        self._merged_model: Optional[cobra.Model] = None
        self._gr_cache: Dict[object, float] = {}

    # -- construction ------------------------------------------------------
    @property
    def merged_model(self) -> cobra.Model:
        if self._merged_model is None:
            self._merged_model = self._merge()
        return self._merged_model

    def _rename(self, old_id: str, org_id: str) -> str:
        return f"{old_id}{self.id_separator}{org_id}"

    def _merge(self) -> cobra.Model:
        merged = Model(self.id)

        # ---- 1. collect the shared extracellular pool --------------------
        ext_templates: "OrderedDict[str, cobra.Metabolite]" = OrderedDict()
        ex_ids: Dict[str, List[str]] = {}
        ex_lb: Dict[str, List[float]] = {}
        ex_ub: Dict[str, List[float]] = {}

        for org in self.organisms.values():
            for met in org.model.metabolites:
                if met.compartment == org.external_compartment:
                    ext_templates.setdefault(met.id, met)
            for rxn in org.model.boundary:
                mets = list(rxn.metabolites)
                if len(mets) != 1:
                    continue
                met = mets[0]
                if met.compartment != org.external_compartment:
                    continue  # sink/demand on an internal metabolite: kept below
                lb, ub = _boundary_bounds(rxn, met)
                ex_ids.setdefault(met.id, []).append(rxn.id)
                ex_lb.setdefault(met.id, []).append(lb)
                ex_ub.setdefault(met.id, []).append(ub)

        ext_ids = {org.external_compartment for org in self.organisms.values()}
        self.external_compartment = ext_ids.pop() if len(ext_ids) == 1 else "u"

        merged.compartments = dict(merged.compartments or {})
        merged.compartments[self.external_compartment] = "shared extracellular space"

        merged.add_metabolites([
            _copy_metabolite(met, met.id, self.external_compartment)
            for met in ext_templates.values()
        ])

        # ---- 2. community exchange reactions -----------------------------
        reserved = set()
        for met_id in ext_templates:
            ids = ex_ids.get(met_id, [])
            if ids and len(set(ids)) == 1 and ids[0] not in reserved:
                rxn_id = ids[0]
            else:
                rxn_id = f"EX_{met_id}"
            while rxn_id in reserved:
                rxn_id += "_com"
            reserved.add(rxn_id)

            if not ids or self.exchange_policy == "closed":
                lb = ub = 0.0
            elif self.exchange_policy == "union":
                lb, ub = min(ex_lb[met_id]), max(ex_ub[met_id])
            else:  # intersection
                lb, ub = max(ex_lb[met_id]), min(ex_ub[met_id])
                if lb > ub:
                    lb = ub = 0.0

            ex_rxn = Reaction(rxn_id)
            ex_rxn.name = f"{met_id} community exchange"
            ex_rxn.lower_bound, ex_rxn.upper_bound = lb, ub
            ex_rxn.add_metabolites({merged.metabolites.get_by_id(met_id): -1.0})
            merged.add_reactions([ex_rxn])
            self.exchange_reactions[rxn_id] = met_id

        # ---- 3. organism-specific metabolites and reactions --------------
        for org_id, org in self.organisms.items():
            self.organism_reactions[org_id] = []

            for cid in {m.compartment for m in org.model.metabolites}:
                if cid != org.external_compartment:
                    merged.compartments[self._rename(cid, org_id)] = \
                        f"{(org.model.compartments or {}).get(cid, cid)} ({org_id})"

            new_mets = []
            for met in org.model.metabolites:
                if met.compartment == org.external_compartment:
                    continue
                new_id = self._rename(met.id, org_id)
                if new_id in merged.metabolites:
                    raise ValueError(
                        f"id collision while merging: '{new_id}'. Choose a "
                        f"different `id_separator` or organism id."
                    )
                new_mets.append(_copy_metabolite(
                    met, new_id, self._rename(met.compartment, org_id)))
                self.metabolite_map[(org_id, met.id)] = new_id
            merged.add_metabolites(new_mets)

            new_rxns = []
            for rxn in org.model.reactions:
                # boundary reactions on the shared pool are replaced by the
                # community exchanges built above; sinks/demands on internal
                # metabolites are kept (the previous version dropped them).
                if rxn.boundary:
                    mets = list(rxn.metabolites)
                    if len(mets) == 1 and mets[0].compartment == org.external_compartment:
                        continue

                new_id = self._rename(rxn.id, org_id)
                if new_id in reserved or new_id in merged.reactions:
                    raise ValueError(f"id collision while merging: '{new_id}'")

                new_rxn = Reaction(new_id)
                new_rxn.name = f"{rxn.name} ({org_id})" if rxn.name else new_id
                new_rxn.subsystem = rxn.subsystem
                new_rxn.lower_bound = float(rxn.lower_bound)
                new_rxn.upper_bound = float(rxn.upper_bound)
                new_rxn.add_metabolites({
                    merged.metabolites.get_by_id(
                        met.id if met.compartment == org.external_compartment
                        else self._rename(met.id, org_id)
                    ): float(coeff)
                    for met, coeff in rxn.metabolites.items()
                })
                new_rxns.append(new_rxn)

                self.reaction_map[(org_id, rxn.id)] = new_id
                self.reaction_owner[new_id] = org_id
                self.original_reaction[new_id] = rxn.id
                self.organism_reactions[org_id].append(new_id)
                if rxn.id == org.biomass_reaction:
                    self.biomass_reactions[org_id] = new_id

            merged.add_reactions(new_rxns)

            if self.copy_gene_rules:
                for rxn in org.model.reactions:
                    merged_id = self.reaction_map.get((org_id, rxn.id))
                    if merged_id and rxn.gene_reaction_rule:
                        merged.reactions.get_by_id(merged_id).gene_reaction_rule = \
                            rxn.gene_reaction_rule

            if org_id not in self.biomass_reactions:
                raise ValueError(
                    f"the biomass reaction '{org.biomass_reaction}' of organism "
                    f"'{org_id}' is a boundary reaction on the shared pool and "
                    f"was dropped during the merge"
                )

        return merged

    # -- convenience -------------------------------------------------------
    def exchange_reaction_for(self, metabolite_id: str) -> Optional[str]:
        """Community exchange reaction id handling a shared metabolite."""
        for rxn_id, met_id in self.exchange_reactions.items():
            if met_id == metabolite_id:
                return rxn_id
        return None

    def __repr__(self):
        return (f"<CommunityModel '{self.id}': {len(self.organisms)} organisms, "
                f"{len(self.merged_model.reactions)} reactions, "
                f"{len(self.exchange_reactions)} community exchanges>")


#: Backwards-compatible alias.
Community = CommunityModel


def _copy_metabolite(met: cobra.Metabolite, new_id: str, compartment: str) -> cobra.Metabolite:
    new = Metabolite(id=new_id, name=met.name, compartment=compartment)
    new.formula = met.formula
    new.charge = met.charge
    return new


def build_steadycom_model(
    models: Sequence[cobra.Model],
    organism_ids: Sequence[str],
    mu_init: float = 0.2,
    community_id: str = "community",
    **kwargs,
) -> CommunityModel:
    """Build a :class:`CommunityModel` from a list of cobrapy models.

    Parameters
    ----------
    models, organism_ids
        Parallel sequences; ``organism_ids`` must be unique and are used to
        suffix every organism-specific id in the merged model.
    mu_init
        Initial guess for the growth-rate search.
    community_id
        Name of the merged model.
    **kwargs
        Forwarded to :class:`CommunityModel` (``biomass_reactions``,
        ``external_compartments``, ``exchange_policy``, ``id_separator``...).

    Notes
    -----
    The organism models are *not* modified and *not* copied; the merge reads
    their bounds at call time, so change them before calling this function.
    """
    if len(models) != len(organism_ids):
        raise ValueError("`models` and `organism_ids` must have the same length")
    if len(set(organism_ids)) < len(organism_ids):
        raise ValueError("`organism_ids` must be unique")

    community = CommunityModel(
        community_id,
        OrderedDict(zip(organism_ids, models)),
        mu_init=mu_init,
        **kwargs,
    )
    _ = community.merged_model  # build eagerly so errors surface here
    return community


# =============================================================================
# The LP
# =============================================================================

class SteadyComProblem:
    """The SteadyCom LP, held in a single reusable Gurobi model.

    Only the coefficients of the ``v_bio - mu * X = 0`` rows change between
    growth rates, so the whole search reuses one factorised basis.
    """

    def __init__(
        self,
        community: CommunityModel,
        params: Optional[SteadyComParams] = None,
        growth_fixed: Optional[Mapping[str, float]] = None,
        biomass_gdw: Optional[Mapping[str, float]] = None,
        biomass_objective: Optional[Mapping[str, float]] = None,
        biomass_constraints: Optional[Sequence[Tuple[Mapping[str, float], str, float]]] = None,
    ):
        self.community = community
        self.params = params or SteadyComParams()
        self.growth_fixed = dict(growth_fixed or {})
        self.biomass_gdw = {org: float((biomass_gdw or {}).get(org, 1.0))
                            for org in community.organisms}
        self.biomass_objective = {org: float((biomass_objective or {}).get(org, 1.0))
                                  for org in community.organisms}

        self.growth_rate: Optional[float] = None
        self._l1_ready = False
        self._abs_vars: Dict[str, gp.Var] = {}

        self._build(biomass_constraints)

    # -- construction ------------------------------------------------------
    def _build(self, biomass_constraints):
        merged = self.community.merged_model
        p = self.params

        self.gp_model = gp.Model(f"steadycom_{self.community.id}")
        self.gp_model.Params.OutputFlag = 0
        self.gp_model.Params.FeasibilityTol = p.feas_tol
        self.gp_model.Params.OptimalityTol = p.feas_tol
        if p.threads:
            self.gp_model.Params.Threads = p.threads

        # --- flux variables ------------------------------------------------
        # Organism-specific fluxes carry no numeric bound of their own: their
        # magnitude is set by the coupling rows below, so that a member with
        # zero biomass carries zero flux.  Only the sign is fixed here, exactly
        # as in constructLPcom.m.  Community exchanges keep real bounds, since
        # they are absolute rates for the whole community.
        self.v: Dict[str, gp.Var] = {}
        self._exchange_default_bounds: Dict[str, Tuple[float, float]] = {}
        for rxn in merged.reactions:
            if rxn.id in self.community.exchange_reactions:
                lb, ub = float(rxn.lower_bound), float(rxn.upper_bound)
                self._exchange_default_bounds[rxn.id] = (lb, ub)
            else:
                lb = -GRB.INFINITY if rxn.lower_bound < 0 else 0.0
                ub = GRB.INFINITY if rxn.upper_bound > 0 else 0.0
            self.v[rxn.id] = self.gp_model.addVar(
                lb=_g(lb), ub=_g(ub), name=rxn.id)

        # --- biomass variables ---------------------------------------------
        self.X: Dict[str, gp.Var] = {
            org: self.gp_model.addVar(lb=0.0, ub=p.biomass_ub, name=f"X_{org}")
            for org in self.community.organisms
        }
        self.gp_model.update()

        # --- mass balance ---------------------------------------------------
        for met in merged.metabolites:
            expr = gp.LinExpr()
            for rxn in met.reactions:
                expr.addTerms(float(rxn.metabolites[met]), self.v[rxn.id])
            self.gp_model.addLConstr(expr, GRB.EQUAL, 0.0, name=f"mb_{met.id}")

        # --- flux coupled to biomass ---------------------------------------
        for org_id, org in self.community.organisms.items():
            X = self.X[org_id]
            for merged_id in self.community.organism_reactions[org_id]:
                original = self.community.original_reaction[merged_id]
                rxn0 = org.model.reactions.get_by_id(original)
                v = self.v[merged_id]
                lb, ub = float(rxn0.lower_bound), float(rxn0.upper_bound)
                if not math.isinf(lb):
                    self.gp_model.addLConstr(
                        v - lb * X, GRB.GREATER_EQUAL, 0.0, name=f"lb_{merged_id}")
                if not math.isinf(ub):
                    self.gp_model.addLConstr(
                        v - ub * X, GRB.LESS_EQUAL, 0.0, name=f"ub_{merged_id}")

        # --- growth rows: v_bio - mu * gdw * X = 0 ---------------------------
        self._gr_constraints: Dict[str, gp.Constr] = {}
        for org_id in self.community.organisms:
            bio = self.v[self.community.biomass_reactions[org_id]]
            self._gr_constraints[org_id] = self.gp_model.addLConstr(
                bio - self.params.gr_guess * self.biomass_gdw[org_id] * self.X[org_id],
                GRB.EQUAL, 0.0, name=f"gr_{org_id}")

        # --- optional user constraints on the biomass variables --------------
        for i, (coeffs, sense, rhs) in enumerate(biomass_constraints or []):
            expr = gp.LinExpr()
            for org_id, coeff in coeffs.items():
                expr.addTerms(float(coeff), self.X[org_id])
            op = {"<": GRB.LESS_EQUAL, "L": GRB.LESS_EQUAL,
                  ">": GRB.GREATER_EQUAL, "G": GRB.GREATER_EQUAL,
                  "=": GRB.EQUAL, "E": GRB.EQUAL}[sense]
            self.gp_model.addLConstr(expr, op, float(rhs), name=f"bmcon_{i}")

        # --- total-biomass window, deactivated until needed ------------------
        total = gp.quicksum(self.X.values())
        self._bm_lo = self.gp_model.addLConstr(
            total, GRB.GREATER_EQUAL, -GRB.INFINITY, name="total_biomass_lb")
        self._bm_hi = self.gp_model.addLConstr(
            total, GRB.LESS_EQUAL, GRB.INFINITY, name="total_biomass_ub")

        self._total_biomass_expr = gp.quicksum(
            self.biomass_objective[org] * X for org, X in self.X.items())
        self.gp_model.setObjective(self._total_biomass_expr, GRB.MAXIMIZE)
        self.gp_model.update()

    # -- medium ------------------------------------------------------------
    def set_medium(self, medium=None, close_others: bool = False):
        """Set the bounds of the community exchange reactions.

        ``medium`` maps a community exchange reaction id -- or the id of the
        shared metabolite -- to ``(lower_bound, upper_bound)`` or to a single
        number read as the lower bound (a negative value is an uptake rate).

        Every exchange not named in ``medium`` is reset to the bound inherited
        from the organism models, or closed for uptake when ``close_others``
        is True.  Calls are therefore idempotent: a second call does not
        accumulate on top of the first.
        """
        for rid, (lb, ub) in self._exchange_default_bounds.items():
            var = self.v[rid]
            var.LB = _g(0.0 if close_others else lb)
            var.UB = _g(ub)

        for key, bounds in (medium or {}).items():
            rid = key if key in self._exchange_default_bounds else \
                self.community.exchange_reaction_for(key)
            if rid is None:
                raise KeyError(
                    f"'{key}' is neither a community exchange reaction nor a "
                    f"shared metabolite of community '{self.community.id}'"
                )
            var = self.v[rid]
            if isinstance(bounds, (tuple, list)):
                lb, ub = bounds
                var.LB = _g(lb if lb is not None else -INF)
                var.UB = _g(ub if ub is not None else INF)
            else:
                var.LB = _g(float(bounds))
        self.gp_model.update()

    # -- growth rate --------------------------------------------------------
    def set_growth_rate(self, mu: float):
        """Rewrite the ``v_bio = mu * X`` rows in place."""
        for org_id, constr in self._gr_constraints.items():
            rate = self.growth_fixed.get(org_id, mu)
            self.gp_model.chgCoeff(
                constr, self.X[org_id], -float(rate) * self.biomass_gdw[org_id])
        self.gp_model.update()

    def set_total_biomass_window(self, lo: Optional[float] = None,
                                 hi: Optional[float] = None):
        """Constrain ``sum(X)``. ``None`` deactivates a side."""
        self._bm_lo.RHS = -GRB.INFINITY if lo is None else float(lo)
        self._bm_hi.RHS = GRB.INFINITY if hi is None else float(hi)
        self.gp_model.update()

    # -- solving -------------------------------------------------------------
    def _optimize(self) -> Status:
        self.gp_model.optimize()
        st = self.gp_model.Status
        if st == GRB.INF_OR_UNBD:
            # ask Gurobi to tell the two apart
            self.gp_model.Params.DualReductions = 0
            self.gp_model.optimize()
            st = self.gp_model.Status
            self.gp_model.Params.DualReductions = 1
        return {
            GRB.OPTIMAL: Status.OPTIMAL,
            GRB.INFEASIBLE: Status.INFEASIBLE,
            GRB.UNBOUNDED: Status.UNBOUNDED,
        }.get(st, Status.OTHER)

    def max_total_biomass(self, mu: Optional[float] = None) -> Tuple[Status, float]:
        """Maximise ``sum(X)`` at a growth rate. Returns ``(status, value)``;
        the value is 0.0 whenever the LP has no solution."""
        if mu is not None:
            self.set_growth_rate(mu)
        self.gp_model.setObjective(self._total_biomass_expr, GRB.MAXIMIZE)
        status = self._optimize()
        return status, (self.gp_model.ObjVal if status == Status.OPTIMAL else 0.0)

    def optimize_variable(self, var: gp.Var, minimize: bool = False) -> Tuple[Status, Optional[float]]:
        self.gp_model.setObjective(
            gp.LinExpr(1.0, var), GRB.MINIMIZE if minimize else GRB.MAXIMIZE)
        status = self._optimize()
        return status, (self.gp_model.ObjVal if status == Status.OPTIMAL else None)

    # -- L1 norm --------------------------------------------------------------
    def _ensure_l1(self):
        """Add ``|v_j| <= a_j`` once, lazily."""
        if self._l1_ready:
            return
        for rid, var in self.v.items():
            a = self.gp_model.addVar(lb=0.0, ub=GRB.INFINITY, name=f"abs_{rid}")
            self._abs_vars[rid] = a
        self.gp_model.update()
        for rid, var in self.v.items():
            a = self._abs_vars[rid]
            self.gp_model.addLConstr(var - a, GRB.LESS_EQUAL, 0.0, name=f"absp_{rid}")
            self.gp_model.addLConstr(-var - a, GRB.LESS_EQUAL, 0.0, name=f"absn_{rid}")
        self.gp_model.update()
        self._l1_ready = True

    def minimize_flux_norm(self) -> Status:
        """Minimise ``sum |v|`` over the current feasible set.

        Without this the reported fluxes are an arbitrary vertex of the optimal
        face, so futile cycles appear and exchange rates are not reproducible
        between runs or solver versions.
        """
        self._ensure_l1()
        self.gp_model.setObjective(
            gp.quicksum(self._abs_vars.values()), GRB.MINIMIZE)
        return self._optimize()

    # -- reading the solution --------------------------------------------------
    def read_fluxes(self) -> Dict[str, float]:
        return {rid: var.X for rid, var in self.v.items()}

    def read_biomass(self) -> Dict[str, float]:
        return {org: var.X for org, var in self.X.items()}

    # -- the growth-rate search -------------------------------------------------
    def max_growth_rate(self) -> Tuple[float, str, List[Tuple[float, float]]]:
        """Find the largest ``mu`` at which the community sustains
        ``bm_weight`` gDW of total biomass.

        Returns ``(growth_rate, stat, iterations)``.  ``growth_rate`` is always
        on the feasible side of the root, so the LP can be re-solved there.
        """
        p = self.params
        iterations: List[Tuple[float, float]] = []

        def total_biomass(mu: float) -> float:
            _, value = self.max_total_biomass(mu)
            iterations.append((mu, value))
            if p.verbose:
                print(f"    mu = {mu:.6f}   sum(X) = {value:.6f}")
            return value

        # --- can the community exist at all? ---------------------------------
        bm0 = total_biomass(0.0)
        if bm0 < p.bm_tol_abs:
            return 0.0, "infeasible", iterations
        if bm0 < p.bm_weight:
            # viable, but it cannot even carry the reference biomass at zero
            # growth: the medium is too poor for bm_weight gDW of cells.
            return 0.0, "maintenance", iterations

        # --- bracket the root with the SteadyCom guess mu' = mu * X / X_ref ---
        gr_lb, bm_lb = 0.0, bm0
        gr_ub, bm_ub = INF, 0.0
        mu = max(p.gr_guess, p.gr_tol)

        for _ in range(p.max_iter):
            bm = total_biomass(mu)
            if bm >= p.bm_weight:
                gr_lb, bm_lb = mu, bm
            else:
                gr_ub, bm_ub = mu, bm
            if gr_ub < INF:
                break
            if gr_lb > p.gr_ub_cap:
                raise RuntimeError(
                    f"the community still grows at mu = {gr_lb:g} 1/h; the "
                    f"model is probably unbounded (check the medium and the "
                    f"bounds of the community exchange reactions)"
                )
            # sum(X) * mu is roughly constant, so this lands close to the root
            nxt = mu * bm / p.bm_weight if p.bm_weight > 0 else mu * 2.0
            mu = max(nxt, mu * 1.01, mu + p.gr_tol)

        if gr_ub == INF:
            raise RuntimeError("could not bracket the maximum growth rate")

        # --- Illinois: safeguarded regula falsi on f(mu) = bm_weight - sum(X) -
        a, fa = gr_lb, p.bm_weight - bm_lb          # <= 0
        b, fb = gr_ub, p.bm_weight - bm_ub          # >  0
        for _ in range(p.max_iter):
            if b - a <= p.gr_tol:
                break
            c = b - fb * (b - a) / (fb - fa) if fb != fa else 0.5 * (a + b)
            span = b - a
            c = min(max(c, a + 0.05 * span), b - 0.05 * span)
            fc = p.bm_weight - total_biomass(c)
            if fc <= 0.0:
                a, fa = c, fc
                fb *= 0.5
            else:
                b, fb = c, fc
                fa *= 0.5

        return a, ("optimal" if a > p.gr_tol else "maintenance"), iterations


def _g(value: float) -> float:
    """Map +/-inf to Gurobi's infinity."""
    if value is None:
        return GRB.INFINITY
    if math.isinf(value):
        return GRB.INFINITY if value > 0 else -GRB.INFINITY
    return float(value)


def build_problem(community: CommunityModel, growth: Optional[float] = None,
                  params: Optional[SteadyComParams] = None, **kwargs) -> SteadyComProblem:
    """Build the SteadyCom LP. Kept for backwards compatibility."""
    problem = SteadyComProblem(community, params=params, **kwargs)
    problem.set_growth_rate(community.mu_init if growth is None else growth)
    return problem


# =============================================================================
# Public API
# =============================================================================

def _medium_key(medium) -> object:
    if not medium:
        return ()
    items = []
    for key, value in sorted(medium.items()):
        items.append((key, tuple(value) if isinstance(value, (tuple, list)) else value))
    return tuple(items)


def SteadyCom(
    community: CommunityModel,
    medium: Optional[Mapping] = None,
    solver: Optional[SteadyComProblem] = None,
    params: Optional[SteadyComParams] = None,
    close_other_exchanges: bool = False,
    **param_overrides,
) -> SteadyComSolution:
    """Maximum community growth rate at community steady state.

    Parameters
    ----------
    community
        A :class:`CommunityModel`, e.g. from :func:`build_steadycom_model`.
    medium
        ``{exchange reaction id or shared metabolite id: (lb, ub) or lb}``
        applied to the community exchange reactions.  Exchanges left out keep
        the bounds inherited from the organism models.
    solver
        An existing :class:`SteadyComProblem` to reuse across calls, which
        skips rebuilding the LP.  Note that the structural parameters
        (``biomass_ub``, ``feas_tol``) are fixed when the problem is built, so
        changing them here has no effect on a reused solver.
    params
        A :class:`SteadyComParams`; individual fields can also be passed as
        keywords (``gr_tol=1e-8``, ``min_norm=False``, ``verbose=True``, ...).
    close_other_exchanges
        Close uptake for every exchange not named in ``medium``, so the medium
        is defined exclusively by that dict.

    Returns
    -------
    SteadyComSolution
        ``.growth_rate`` (also ``.fobj``), ``.abundance``, ``.biomass``,
        ``.fluxes`` and the flat ``.values`` dict of the previous API.

    Notes
    -----
    A member is free to go extinct: if the community grows fastest without it,
    its abundance comes out at 0.  That is the correct SteadyCom answer, not a
    failure -- use :func:`SteadyComVA` below the optimum to see the range of
    compositions that support coexistence.
    """
    params = _resolve_params(params, param_overrides)
    problem = solver or SteadyComProblem(community, params=params)
    problem.params = params
    problem.set_medium(medium, close_others=close_other_exchanges)

    growth_rate, stat, iterations = problem.max_growth_rate()
    community._gr_cache[_medium_key(medium)] = growth_rate

    if stat == "infeasible":
        return SteadyComSolution(community, Status.INFEASIBLE, stat=stat,
                                 iterations=iterations)

    # Re-solve at the optimum and pin the total biomass so that the flux
    # distribution belongs to the optimal face, then make it reproducible.
    status, bm_max = problem.max_total_biomass(growth_rate)
    if status != Status.OPTIMAL:
        return SteadyComSolution(community, status, stat=stat, iterations=iterations)

    if params.min_norm:
        problem.set_total_biomass_window(lo=bm_max * (1.0 - 1e-6), hi=None)
        norm_status = problem.minimize_flux_norm()
        if norm_status != Status.OPTIMAL:  # numerical edge case: fall back
            problem.set_total_biomass_window(None, None)
            problem.max_total_biomass(growth_rate)

    # Read the solution out before touching the model again: any structural
    # change discards Gurobi's solution vector.
    fluxes = problem.read_fluxes()
    biomass = problem.read_biomass()
    problem.set_total_biomass_window(None, None)
    total = sum(biomass.values())

    if params.normalize and total > 0:
        fluxes = {k: v / total for k, v in fluxes.items()}
        biomass = {k: v / total for k, v in biomass.items()}
        total_reported = 1.0
    else:
        total_reported = total

    problem.growth_rate = growth_rate
    return SteadyComSolution(
        community, Status.OPTIMAL, growth_rate=growth_rate, stat=stat,
        fluxes=fluxes, biomass=biomass, total_biomass=total_reported,
        iterations=iterations,
    )


def SteadyComVA(
    community: CommunityModel,
    obj_frac: float = 1.0,
    medium: Optional[Mapping] = None,
    solver: Optional[SteadyComProblem] = None,
    organisms: Optional[Iterable[str]] = None,
    gr_max: Optional[float] = None,
    params: Optional[SteadyComParams] = None,
    close_other_exchanges: bool = False,
    **param_overrides,
) -> Dict[str, List[float]]:
    """Range of relative abundance of every member at a fixed growth rate.

    The growth rate is fixed at ``obj_frac * mu_max`` and the total biomass at
    ``bm_weight``, then each ``X_k`` is minimised and maximised.  Because the
    total is pinned, the returned numbers are directly relative abundances.

    Parameters
    ----------
    obj_frac
        Fraction of the maximum community growth rate (0 < obj_frac <= 1).
    gr_max
        Skip the growth-rate search and use this value.  ``SteadyComVA``
        already caches ``mu_max`` per community and medium, so sweeping
        ``obj_frac`` costs one search in total, not one per point.
    organisms
        Restrict the analysis to these members.

    Returns
    -------
    dict
        ``{organism_id: [minimum, maximum]}``.  A bound is ``None`` only if the
        LP is infeasible at that growth rate.
    """
    params = _resolve_params(params, param_overrides)
    problem = solver or SteadyComProblem(community, params=params)
    problem.params = params
    problem.set_medium(medium, close_others=close_other_exchanges)

    key = _medium_key(medium)
    if gr_max is None:
        gr_max = community._gr_cache.get(key)
    if gr_max is None:
        gr_max, stat, _ = problem.max_growth_rate()
        community._gr_cache[key] = gr_max

    growth = obj_frac * gr_max
    problem.set_growth_rate(growth)
    problem.set_total_biomass_window(lo=params.bm_weight, hi=params.bm_weight)

    targets = list(organisms) if organisms is not None else list(community.organisms)
    variability: Dict[str, List[Optional[float]]] = {org: [None, None] for org in targets}
    try:
        for org in targets:
            for i, minimize in enumerate((True, False)):
                status, value = problem.optimize_variable(problem.X[org], minimize=minimize)
                if status == Status.OPTIMAL:
                    variability[org][i] = value / params.bm_weight
    finally:
        problem.set_total_biomass_window(None, None)

    return variability


def SteadyComFVA(
    community: CommunityModel,
    reactions: Optional[Iterable[str]] = None,
    obj_frac: float = 1.0,
    medium: Optional[Mapping] = None,
    solver: Optional[SteadyComProblem] = None,
    gr_max: Optional[float] = None,
    params: Optional[SteadyComParams] = None,
    close_other_exchanges: bool = False,
    **param_overrides,
) -> Dict[str, List[float]]:
    """Flux variability of the community at ``obj_frac * mu_max``.

    ``reactions`` are merged-model reaction ids; it defaults to the community
    exchange reactions.  Same conventions as :func:`SteadyComVA`.
    """
    params = _resolve_params(params, param_overrides)
    problem = solver or SteadyComProblem(community, params=params)
    problem.params = params
    problem.set_medium(medium, close_others=close_other_exchanges)

    key = _medium_key(medium)
    if gr_max is None:
        gr_max = community._gr_cache.get(key)
    if gr_max is None:
        gr_max, _, _ = problem.max_growth_rate()
        community._gr_cache[key] = gr_max

    problem.set_growth_rate(obj_frac * gr_max)
    problem.set_total_biomass_window(lo=params.bm_weight, hi=params.bm_weight)

    targets = list(reactions) if reactions is not None else list(community.exchange_reactions)
    result: Dict[str, List[Optional[float]]] = {rid: [None, None] for rid in targets}
    try:
        for rid in targets:
            for i, minimize in enumerate((True, False)):
                status, value = problem.optimize_variable(problem.v[rid], minimize=minimize)
                if status == Status.OPTIMAL:
                    result[rid][i] = value / params.bm_weight
    finally:
        problem.set_total_biomass_window(None, None)

    return result


def _resolve_params(params: Optional[SteadyComParams],
                    overrides: Mapping) -> SteadyComParams:
    base = params or SteadyComParams()
    if not overrides:
        return base
    unknown = set(overrides) - set(vars(base))
    if unknown:
        raise TypeError(f"unknown SteadyCom parameter(s): {sorted(unknown)}")
    return SteadyComParams(**{**vars(base), **overrides})
