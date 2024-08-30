"""Wrapper functions to use SpatialDE directly on AnnData objects."""

import logging
from collections.abc import Collection
from typing import Any

import anndata as ad
import NaiveDE
import pandas as pd

from .aeh import spatial_patterns
from .base import run
from .util import qvalue


def spatialde_test(
    adata: ad.AnnData,
    coord_columns: Collection[str] = ("x", "y"),
    regress_formula: str = "np.log(total_counts)",
) -> pd.DataFrame:
    """Run the SpatialDE test on an AnnData object.

    Parameters
    ----------

    adata: An AnnData object with counts in the .X field.

    coord_columns: A list with the columns of adata.obs which represent spatial
    coordinates. Default ['x', 'y'].

    regress_formula: A patsy formula for linearly regressing out fixed effects from
    columns in adata.obs before fitting the SpatialDE models. Default is
    'np.log(total_counts)'.

    Returns:
    -------
    results: A table of spatial statistics for each gene.
    """
    coord_columns = list(coord_columns)
    logging.info("Performing VST for NB counts")
    adata.layers["stabilized"] = NaiveDE.stabilize(adata.X.T).T

    logging.info("Regressing out fixed effects")
    adata.layers["residual"] = NaiveDE.regress_out(
        adata.obs, adata.layers["stabilized"].T, regress_formula
    ).T

    x = adata.obs[coord_columns].to_numpy()
    expr_mat = pd.DataFrame.from_records(
        adata.layers["residual"], columns=adata.var.index, index=adata.obs.index
    )

    results = run(x, expr_mat)

    # Clip 0 p-values.
    min_pval = results.query("pval > 0")["pval"].min() / 2
    results["pval"] = results["pval"].clip_lower(min_pval)

    # Correct for multiple testing.
    results["qval"] = qvalue(results["pval"], pi0=1.0)

    return results


def automatic_expression_histology(
    adata: ad.AnnData,
    filtered_results: pd.DataFrame,
    C: int,  # noqa: N803
    l: float,  # noqa: E741
    coord_columns: Collection[str] = ("x", "y"),
    layer: str = "residual",
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the Automatic Expression Histology (AEH) model for an AnnData object.

    Args:
        adata (AnnData): An AnnData object with a layer of stabilized expression values
        filtered_results (DataFrame): A DataFrame with the significant subset of
        results from the SpatialDE significance test.
        C (int): The number of hidden spatial patterns.
        l (float): The common length-scale for the hidden spatial patterns.
        coord_columns (Collection[str], optional): A list with the columns of
        `adata.obs` which represent spatial coordinates. Defaults to ("x", "y").
        layer (str, optional): A string indicating the layer of adata to fit the AEH
        model to. Defaults to "residual".
        kwargs: (Any, optional): Remaining arguments are passed to
        `SpatialDE.aeh.spatial_patterns()`.

    Returns:
        tuple[DataFrame, DataFrame]: A DataFrame with pattern membership information for
        each gene, and a DataFrame with the inferred hidden spatial functions the genes
        belong to evaluated at all points in the data.
    """
    x = adata.obs[[*coord_columns]].to_numpy()

    expr_mat = pd.DataFrame.from_records(
        adata.layers[layer], columns=adata.var.index, index=adata.obs.index
    )

    logging.info("Performing Automatic Expression Histology")
    histology_results, patterns = spatial_patterns(
        x, expr_mat, filtered_results, C, l, **kwargs
    )

    return histology_results, patterns
