"""Concept-level severity direction + deterministic 5-state temporal readout for FTCB (Phase 4).

Two faithful, non-learned building blocks used by the Faithful Temporal Concept Bottleneck:

1. SEVERITY_SIGN — per-concept clinical direction s_c in {+1, 0}:
     +1  a rise in the concept's activation means the finding got WORSE
          (pathology: effusion, edema, consolidation, pneumothorax, pneumonia, ...).
      0  no reliable severity direction -> EXCLUDED from the direction head
          (chronic/benign findings + ALL tubes/lines/devices; spec Part A section 4.2).
   There is no -1: the M3 concept space has no "clearing/resolution" concept, so the *improved*
   direction is expressed by a NEGATIVE delta of a +1 concept (e = s_c * delta_c), not by a -1 sign.

2. concept_state(...) — a DETERMINISTIC 5-state readout per (region, concept) from the prior and
   current concept activations, no training involved (so it is faithful by construction):
     new       concept crossed absent->present   (only current present)
     resolved  concept crossed present->absent    (only prior present)
     worsened  present in both, severity rose      (e = s*delta > +deadband)
     improved  present in both, severity fell       (e = s*delta < -deadband)
     stable    present in both, |delta| within deadband (or s==0, presence unchanged)
     absent    concept absent in both              (masked; not surfaced)

Disease-level M4 output stays 3-class {stable, improved, worsened} for MS-CXR-T comparability; the
concept-level new/resolved states are surfaced in the M5 evidence ledger, not forced into the disease
head. This mirrors CheXTemporal's {New, Worse, Stable, Improved, Resolved} at the honest concept level.

REVIEW: the +1/0 assignment below is a first clinical draft (per user request). Borderline concepts are
tagged `# REVIEW`; a clinician should confirm before the FTCB direction claim is finalized.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Concept severity direction by concept NAME (robust to concept-index reordering; the [69] vector is
# built by looking names up against the shipped concept space). +1 = rise-is-worse, 0 = excluded.
SEVERITY_SIGN: dict[str, int] = {
    # --- anatomical findings: acute pathology, rise-is-worse (+1) ---
    "lung opacity": 1,
    "pleural effusion": 1,
    "pulmonary edema/hazy opacity": 1,
    "pneumothorax": 1,
    "atelectasis": 1,
    "consolidation": 1,
    "mediastinal displacement": 1,
    "vascular congestion": 1,
    "mediastinal widening": 1,
    "superior mediastinal mass/enlargement": 1,
    "enlarged cardiac silhouette": 1,
    "enlarged hilum": 1,
    "lung lesion": 1,
    "linear/patchy atelectasis": 1,
    "airspace opacity": 1,
    "lobar/segmental collapse": 1,
    "sub-diaphragmatic air": 1,
    "mass/nodule (not otherwise specified)": 1,
    "vascular redistribution": 1,
    "infiltration": 1,
    "costophrenic angle blunting": 1,
    "rib fracture": 1,
    "multiple masses/nodules": 1,
    "pneumomediastinum": 1,
    "subcutaneous air": 1,
    "increased reticular markings/ild pattern": 1,
    "hydropneumothorax": 1,
    "spinal fracture": 1,
    "clavicle fracture": 1,
    # --- diseases: acute, rise-is-worse (+1) ---
    "pneumonia": 1,
    "fluid overload/heart failure": 1,
    "aspiration": 1,
    "interstitial lung disease": 1,
    "lung cancer": 1,
    "alveolar hemorrhage": 1,
    "pericardial effusion": 1,
    # --- chronic / benign / structural findings: no reliable acute severity direction (0) ---
    "pleural/parenchymal scarring": 0,
    "tortuous aorta": 0,
    "bone lesion": 0,          # REVIEW: metastasis vs benign — ambiguous acuity
    "vascular calcification": 0,
    "hyperaeration": 0,        # REVIEW: chronic air-trapping; increase could be worse
    "hernia": 0,
    "calcified nodule": 0,     # benign (dropped from Lung Lesion in xwalk v2)
    "elevated hemidiaphragm": 0,   # REVIEW: often positional/chronic; can mean volume loss
    "spinal degenerative changes": 0,
    "shoulder osteoarthritis": 0,
    "scoliosis": 0,
    "bronchiectasis": 0,       # REVIEW: chronic structural
    "cyst/bullae": 0,          # benign (dropped from Lung Lesion in xwalk v2)
    "diaphragmatic eventration (benign)": 0,
    # --- chronic diseases: no acute-change direction (0) ---
    "copd/emphysema": 0,
    "granulomatous disease": 0,
    "goiter": 0,
    # --- tubes/lines + devices: treatment hardware, NEVER a severity direction (0) ---
    "enteric tube": 0, "endotracheal tube": 0, "picc": 0, "ij line": 0, "chest port": 0,
    "chest tube": 0, "swan-ganz catheter": 0, "subclavian line": 0, "tracheostomy tube": 0,
    "pigtail catheter": 0, "intra-aortic balloon pump": 0, "mediastinal drain": 0,
    "cardiac pacer and wires": 0, "cabg grafts": 0, "prosthetic valve": 0, "aortic graft/repair": 0,
}

# 5-state (+ absent) concept-level temporal readout. Disease-level PROG (stable/improved/worsened)
# is unchanged; these are the honest concept-level states surfaced in the evidence ledger.
CONCEPT_STATE_NAMES = ["absent", "stable", "improved", "worsened", "new", "resolved"]
CS_ABSENT, CS_STABLE, CS_IMPROVED, CS_WORSENED, CS_NEW, CS_RESOLVED = range(6)

DEFAULT_PRESENCE_TAU = 0.5   # sigmoid(concept_logit) >= tau  => concept present
DEFAULT_DEADBAND = 0.10      # |delta| below this in an overlap => stable (noise guard, spec L_stable-margin)

_DEFAULT_SPACE = Path(__file__).with_name("m3_concept_space.json")


def concept_names(space_path: Path | str = _DEFAULT_SPACE) -> list[str]:
    """Ordered concept names (idx 0..68) from the shipped concept space."""
    space = json.loads(Path(space_path).read_text())
    return [c["name"] for c in sorted(space["concepts"], key=lambda c: c["idx"])]


def severity_vector(names: list[str] | None = None,
                    space_path: Path | str = _DEFAULT_SPACE) -> np.ndarray:
    """Return the [n_concepts] severity-sign vector aligned to concept index order.

    Raises if a concept has no SEVERITY_SIGN entry, so the table can never silently drift out of
    sync with the concept space.
    """
    names = names if names is not None else concept_names(space_path)
    missing = [n for n in names if n not in SEVERITY_SIGN]
    if missing:
        raise KeyError(f"SEVERITY_SIGN missing {len(missing)} concept(s): {missing}")
    return np.array([SEVERITY_SIGN[n] for n in names], dtype=np.float32)


def directed_evidence(c_prior: np.ndarray, c_current: np.ndarray,
                      sign: np.ndarray) -> np.ndarray:
    """e = s_c * (c_current - c_prior). e>0 supports worsened, e<0 supports improved (spec 4.1)."""
    return sign * (c_current - c_prior)


def concept_state(c_prior: np.ndarray, c_current: np.ndarray, sign: np.ndarray,
                  tau: float = DEFAULT_PRESENCE_TAU,
                  deadband: float = DEFAULT_DEADBAND) -> np.ndarray:
    """Deterministic 5-state (+absent) readout per (…, concept). Fully vectorized; no training.

    Shapes broadcast: c_prior/c_current [..., n_concepts], sign [n_concepts] (or broadcastable).
    Returns int array of CS_* codes with the same leading shape.
    """
    c_prior = np.asarray(c_prior, dtype=np.float32)
    c_current = np.asarray(c_current, dtype=np.float32)
    pres_p = c_prior >= tau
    pres_c = c_current >= tau
    e = sign * (c_current - c_prior)

    state = np.full(c_prior.shape, CS_ABSENT, dtype=np.int64)
    both = pres_p & pres_c
    # overlap: direction from severity-signed delta, with a deadband -> stable
    state[both & (np.abs(e) <= deadband)] = CS_STABLE
    state[both & (e < -deadband)] = CS_IMPROVED
    state[both & (e > deadband)] = CS_WORSENED
    # presence crossings dominate (a concept that appeared/disappeared is new/resolved, not a delta)
    state[pres_c & ~pres_p] = CS_NEW
    state[pres_p & ~pres_c] = CS_RESOLVED
    return state
