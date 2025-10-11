import sfnnv9.model as SFNNv9
import sfnnvkan_g2_k2_3072_15_32_mlp.model as SFNNvkan_g2_k2_3072_15_32_mlp
import sfnnvkan_g3_k3_3072_15_32_mlp.model as SFNNvkan_g3_k3_3072_15_32_mlp

nets = {
    'sfnnv9': lambda feature_set: SFNNv9.NNUE(feature_set),
    'sfnnvkan_g2_k2_3072_15_32_mlp': lambda feature_set: SFNNvkan_g2_k2_3072_15_32_mlp.NNUE(feature_set),
    'sfnnvkan_g3_k3_3072_15_32_mlp': lambda feature_set: SFNNvkan_g3_k3_3072_15_32_mlp.NNUE(feature_set),
}
"""nets contains models to be run in scripts"""
