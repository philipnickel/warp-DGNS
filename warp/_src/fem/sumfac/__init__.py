# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sum-factorized high-order DG operators for :mod:`warp.fem`.

This subpackage holds the building blocks for the sum-factorization path:
1D interpolation/derivative matrices, tensor-product contraction kernels, and
the Q-function extraction machinery. See ``design/`` and the design spec for the
overall ``B^T D B`` factorization.
"""

from warp._src.fem.sumfac.kernels import (
    SumfacNotApplicableError,
    make_sumfac_linear_operator,
)
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)
from warp._src.fem.sumfac.qfunction import (
    extract_qfunction,
    reference_geometry_factors,
)

__all__ = [
    "SumfacNotApplicableError",
    "build_derivative_matrix",
    "build_interpolation_matrix",
    "default_basis_nodes",
    "default_quadrature_points",
    "extract_qfunction",
    "make_sumfac_linear_operator",
    "reference_geometry_factors",
]
