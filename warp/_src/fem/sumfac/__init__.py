# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sum-factorized high-order DG operators for :mod:`warp.fem`.

This subpackage holds the building blocks for the sum-factorization path:
1D interpolation/derivative matrices, tensor-product contraction kernels, and
the Q-function extraction machinery. See ``design/`` and the design spec for the
overall ``B^T D B`` factorization.
"""

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
from warp._src.fem.sumfac.tensor_contract import (
    build_operator_arrays,
    interpolate_2d,
    interpolate_3d,
    make_interpolation_kernel_2d,
    make_interpolation_kernel_3d,
    pack_dofs_2d,
    pack_dofs_3d,
)

__all__ = [
    "build_derivative_matrix",
    "build_interpolation_matrix",
    "build_operator_arrays",
    "default_basis_nodes",
    "default_quadrature_points",
    "extract_qfunction",
    "interpolate_2d",
    "interpolate_3d",
    "make_interpolation_kernel_2d",
    "make_interpolation_kernel_3d",
    "pack_dofs_2d",
    "pack_dofs_3d",
    "reference_geometry_factors",
]
