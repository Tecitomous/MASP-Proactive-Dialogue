"""Public surface of `masp.data`.

New (preferred) names are exported first; legacy P4G-flavoured aliases
are kept for compatibility.
"""
# --- generic dataset abstraction (preferred) ---
from .dataset_adapter import (
    ASSISTANT_SPEAKERS,
    USER_SPEAKERS,
    CraigslistBargainAdapter,
    DatasetAdapter,
    DialogueSession,
    DialogueTurn,
    EmpatheticDialoguesAdapter,
    EpisodeSeed,
    ESConvAdapter,
    P4GAdapter,
    build_episode_seeds,
    get_adapter,
    list_adapters,
    load_dialogue_sessions,
    register_adapter,
)

# --- legacy P4G-flavoured aliases ---
from .p4g_loader import (
    P4GSession,
    P4GTurn,
    load_p4g_sessions,
)

# --- BDI-derived datasets ---
from .bdi_dataset import (
    BCDataset,
    BDILabelCache,
    BDITurnDataset,
    build_bc_dataset,
    build_bdi_turn_dataset,
)
