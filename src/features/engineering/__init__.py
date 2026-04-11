"""
Feature engineering para Spaceship Titanic.

Re-exporta todo desde los submódulos para mantener compatibilidad con imports
existentes (src.features.engineering.X sigue funcionando).
"""
from src.features.engineering.base import (
    _SPENDING_COLS,
    _CATEGORICAL_FILL,
    apply_domain_rules,
    create_age_features,
    create_spending_features,
    extract_cabin_features,
    extract_group_features,
    handle_missing_values_spaceship,
    impute_age_by_group,
    impute_spending_group_aware,
)
from src.features.engineering.derived import (
    _add_cabin_percentile,
    create_child_route_features,
    create_cryo_spending_interaction_features,
    create_group_context_features,
    create_group_spending_features,
    create_solo_interaction_features,
    create_structural_context_features,
)
from src.features.engineering.encoders import (
    _cryo_to_int,
    encode_cryosleep,
    encode_side,
)

# Aliases de backward compatibility (nombres usados en scripts y notebooks existentes)
create_exp007_features = create_group_spending_features
create_fs003_features = create_solo_interaction_features
create_fs005_features = create_structural_context_features
create_fs010_features = create_cryo_spending_interaction_features
create_fs011_features = create_child_route_features
create_fs013_features = create_group_context_features

__all__ = [
    # base
    "apply_domain_rules",
    "create_age_features",
    "create_spending_features",
    "extract_cabin_features",
    "extract_group_features",
    "handle_missing_values_spaceship",
    "impute_age_by_group",
    "impute_spending_group_aware",
    # derived
    "create_child_route_features",
    "create_cryo_spending_interaction_features",
    "create_group_context_features",
    "create_group_spending_features",
    "create_solo_interaction_features",
    "create_structural_context_features",
    # encoders
    "encode_cryosleep",
    "encode_side",
    # backward compat
    "create_exp007_features",
    "create_fs003_features",
    "create_fs005_features",
    "create_fs010_features",
    "create_fs011_features",
    "create_fs013_features",
]
