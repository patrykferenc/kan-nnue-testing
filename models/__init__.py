from .sfnnv9 import model as SFNNv9
from .sfnnvkan_g2_k2_3072_15_32_mlp import model as SFNNvkan_g2_k2_3072_15_32_mlp
from .sfnnvkan_g3_k3_3072_15_32_mlp import model as SFNNvkan_g3_k3_3072_15_32_mlp
from .sfnnvkan_g2_k2_3072_2_4_mlp import model as SFNNvkan_g2_k2_3072_2_4_mlp
from .sfnnvkan_early_3072_15_32 import model as SFNNvkan_early_3072_15_32
from .sfnnvkan_refactored_early_g2_k2_3072_15_32 import model as SFNNvkan_refactored_early_3072_15_32

nets = {
    'sfnnv9': lambda feature_set: SFNNv9.NNUE(feature_set),
    'sfnnvkan_g2_k2_3072_15_32_mlp': lambda feature_set: SFNNvkan_g2_k2_3072_15_32_mlp.NNUE(feature_set),
    'sfnnvkan_g3_k3_3072_15_32_mlp': lambda feature_set: SFNNvkan_g3_k3_3072_15_32_mlp.NNUE(feature_set),
    'sfnnvkan_g2_k2_3072_2_4_mlp': lambda feature_set: SFNNvkan_g2_k2_3072_2_4_mlp.NNUE(feature_set),
    'sfnnvkan_early_3072_15_32': lambda feature_set: SFNNvkan_early_3072_15_32.NNUE(feature_set),
    'sfnnvkan_refactored_early_g2_k2_3072_15_32': lambda feature_set: SFNNvkan_refactored_early_3072_15_32.NNUE(feature_set),
}
"""nets contains models to be run in scripts"""
