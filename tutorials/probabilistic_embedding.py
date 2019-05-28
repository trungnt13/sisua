from __future__ import print_function, division, absolute_import

from odin.stats import describe

from sisua.data import get_dataset
from sisua.label_threshold import ProbabilisticEmbedding

# ===========================================================================
# Load dataset
# ===========================================================================
FIGURE_PATH = '/tmp/tmp.pdf'

ds, _, _ = get_dataset('pbmc_citeseq')
protein = ds['y']
protein_name = ds['y_col']

print(protein.shape)
print(protein_name)

# ===========================================================================
# Probabilistic Embedding
# ===========================================================================
pb = ProbabilisticEmbedding(
    n_components_per_class=2, positive_component=1,
    log_norm=True, clip_quartile=0., remove_zeros=True,
    ci_threshold=-0.68, random_state=5218,
    verbose=True)
pb.fit(protein)

# binarize the protein matrix
y_bin = pb.predict(protein)
print(describe(y_bin))

# probabilize the protein matrix
y_prob = pb.predict_proba(protein)
print(describe(y_prob))

# ====== save the analysis and diagnosis ====== #
pb.boxplot(protein, protein_name
).plot_diagnosis(protein, protein_name
).plot_distribution(protein, protein_name
).save_figures(FIGURE_PATH, verbose=True)
