# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Phase 3 spike (spec risk 9.1): staged ``B^T D B`` kernel through the existing machinery.

One fused tile kernel, generated through ``cache.get_integrand_kernel`` +
``PassFieldArgsToIntegrand`` (the exact machinery ``_generate_integrate_kernel``
uses), for the 2D mass form ``u * v`` on a discontinuous ``Grid2D`` space at
``P = 3``:

1. ``B`` stage: per-element nodal DOFs interpolated to quadrature points with
   two ``wp.tile_matmul`` contractions against the 1D interpolation matrix.
2. ``D`` stage: per quadrature point, the *transformed user integrand* is
   invoked ``d + 1`` times between the tile ops, with the test field replaced
   by a ``SeedField`` (value/gradient seeds via ``Sample.test_dof``) and the
   trial value injected through a minimal ``ValueInjectedField`` whose
   ``EvalArg`` carries the ``B``-stage interpolated value.
3. ``B^T`` stage: ``(f0, f1)`` tiles contracted back to nodal residuals with
   transposed tile matmuls; result stored per element and scattered to the
   global DOF vector by node index.

The result is compared against the naive ``fem.integrate`` oracle on both CPU
and CUDA.
"""

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem import cache
from warp._src.fem.field.virtual import AdjointField, SeedField
from warp._src.fem.integrate import (
    IntegrandTransformer,
    PassFieldArgsToIntegrand,
    _check_field_compat,
    _find_integrand_operators,
    _gen_field_struct,
    _notify_operator_usage,
    _parse_integrand_arguments,
)
from warp._src.fem.sumfac.operators_1d import (
    build_derivative_matrix,
    build_interpolation_matrix,
    default_basis_nodes,
    default_quadrature_points,
)
from warp._src.fem.sumfac.tensor_contract import _default_block_dim
from warp._src.fem.types import NULL_DOF_INDEX, DofIndex, make_free_sample

DEGREE = 3
N = DEGREE + 1  # 1D nodes per element
Q = DEGREE + 1  # 1D quadrature points (GL, order 2*DEGREE)


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


class ValueInjectedField(AdjointField):
    """Trial-value stand-in whose value is injected by the staged kernel.

    Minimal spike version of the ``SeedField`` sibling sketched in
    ``warp._src.fem.sumfac.qfunction`` ("Trial-value injection (Phase 3)"):
    the ``EvalArg`` holds a plain ``value`` member that the fused kernel fills
    per quadrature point from the ``B``-stage interpolation tile before
    calling the transformed integrand.
    """

    @classmethod
    def from_field(cls, field, domain):
        return cls(field.space, field.space_partition, domain)

    def _make_eval_arg(self):
        @cache.dynamic_struct(suffix=self.name)
        class EvalArg:
            value: self.dtype

        return EvalArg

    def fill_eval_arg(self, arg, device):
        # The value member is written in-kernel; nothing to fill host-side.
        pass

    def _make_eval_inner(self):
        @cache.dynamic_func(suffix=self.name)
        def eval_value_inner(args: self.ElementEvalArg, s: self.SampleType):
            return args.eval_arg.value

        return eval_value_inner

    def _make_eval_grad_inner(self):
        # The mass form does not read grad(u); a full Phase 3 version adds a
        # `gradient` member filled from the B-stage derivative tiles.
        return None

    def _make_eval_div_inner(self):
        return None

    def _make_eval_outer(self):
        return self.eval_inner

    def _make_eval_grad_outer(self):
        return None

    def _make_eval_div_outer(self):
        return None


def build_sumfac_kernel(integrand, domain, quadrature, arguments, FieldStruct, ValueStruct, integrand_func, u_field):
    """Generate the fused B^T D B kernel through ``cache.get_integrand_kernel``."""

    SampleType = domain.geometry.sample_type
    scalar_type = domain.geometry.scalar_type
    UEvalArg = u_field.EvalArg

    n_c = wp.constant(N)
    q_c = wp.constant(Q)
    nn_c = wp.constant(N * N)

    def sumfac_mass_kernel_fn(
        qp_arg: quadrature.Arg,
        domain_arg: domain.ElementArg,
        domain_index_arg: domain.ElementIndexArg,
        fields: FieldStruct,
        values: ValueStruct,
        interp: wp.array2d(dtype=scalar_type),
        deriv: wp.array2d(dtype=scalar_type),
        u_dofs: wp.array2d(dtype=scalar_type),
        result_elem: wp.array2d(dtype=scalar_type),
    ):
        domain_element_index = wp.tid()
        element_index = domain.element_index(domain_index_arg, domain_element_index)

        # --- B stage: interpolate nodal DOFs to quadrature points -----------
        a_tile = wp.tile_load(interp, shape=(q_c, n_c))
        a_t = wp.tile_transpose(a_tile)
        d_tile = wp.tile_load(deriv, shape=(q_c, n_c))
        d_t = wp.tile_transpose(d_tile)

        dofs_row = wp.tile_load(u_dofs, shape=(1, nn_c), offset=(element_index, 0))
        u_mat = wp.tile_reshape(dofs_row, shape=(n_c, n_c))  # [i, j], i slow

        stage1 = wp.tile_matmul(a_tile, u_mat)  # (q, n) [qx, j]
        uq = wp.tile_zeros(shape=(q_c, q_c), dtype=scalar_type)
        wp.tile_matmul(stage1, a_t, uq)  # (q, q) [qx, qy]

        # --- D stage: seeded integrand evaluation per quadrature point ------
        f0 = wp.tile_zeros(shape=(q_c, q_c), dtype=scalar_type)
        f1x = wp.tile_zeros(shape=(q_c, q_c), dtype=scalar_type)
        f1y = wp.tile_zeros(shape=(q_c, q_c), dtype=scalar_type)

        qp_fields = FieldStruct()
        qp_fields.v = fields.v

        for qx in range(q_c):
            for qy in range(q_c):
                qp = qx * q_c + qy
                qp_index = quadrature.point_index(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_coords = quadrature.point_coords(domain_arg, qp_arg, domain_element_index, element_index, qp)
                qp_weight = quadrature.point_weight(domain_arg, qp_arg, domain_element_index, element_index, qp)

                free_sample = make_free_sample(element_index, qp_coords)
                vol = domain.element_measure(domain_arg, free_sample)
                scale = qp_weight * vol

                # Inject the B-stage interpolated trial value
                u_arg = UEvalArg()
                u_arg.value = uq[qx, qy]
                qp_fields.u = u_arg

                # Value seed (v = 1, grad v = 0) -> f0
                sample = SampleType(element_index, qp_coords, qp_index, qp_weight, DofIndex(0, 0), NULL_DOF_INDEX)
                f0[qx, qy] = scale * scalar_type(integrand_func(sample, qp_fields, values))

                # Gradient seeds (v = 0, grad v = e_i) -> f1 (zero for mass,
                # but exercises repeated in-kernel integrand calls). The seeds
                # are physical, so map back to reference with J^{-1}.
                jac = domain.element_deformation_gradient(domain_arg, free_sample)
                jac_inv = wp.inverse(jac)
                f1_phys = wp.vec2d()
                for seed in range(2):
                    sample = SampleType(
                        element_index, qp_coords, qp_index, qp_weight, DofIndex(seed + 1, 0), NULL_DOF_INDEX
                    )
                    f1_phys[seed] = scale * scalar_type(integrand_func(sample, qp_fields, values))
                f1_ref = jac_inv * f1_phys
                f1x[qx, qy] = f1_ref[0]
                f1y[qx, qy] = f1_ref[1]

        # --- B^T stage: contract (f0, f1) back to nodal residuals -----------
        # r = Kron(I, I)^T f0 + Kron(D, I)^T f1x + Kron(I, D)^T f1y
        tmp0 = wp.tile_matmul(a_t, f0)  # (n, q)
        r = wp.tile_zeros(shape=(n_c, n_c), dtype=scalar_type)
        wp.tile_matmul(tmp0, a_tile, r)

        tmp1 = wp.tile_matmul(d_t, f1x)  # (n, q)
        wp.tile_matmul(tmp1, a_tile, r)
        tmp2 = wp.tile_matmul(a_t, f1y)  # (n, q)
        wp.tile_matmul(tmp2, d_tile, r)

        r_flat = wp.tile_reshape(r, shape=(1, nn_c))
        wp.tile_store(result_elem, r_flat, offset=(domain_element_index, 0))

    field_names = tuple((k, f.name) for k, f in arguments.field_args.items())
    kernel, _fs, _vs = cache.get_integrand_kernel(
        integrand=integrand,
        kernel_fn=sumfac_mass_kernel_fn,
        suffix=("sumfac-spike", quadrature.name, field_names, N, Q, 2, 1),  # (marker, quad, fields, n, q, dim, E_b)
        code_transformers=[
            PassFieldArgsToIntegrand(
                arg_names=integrand.argspec.args,
                parsed_args=arguments,
                integrand_func=integrand_func,
                fields_var_name="qp_fields",
            )
        ],
        FieldStruct=FieldStruct,
        ValueStruct=ValueStruct,
    )
    return kernel


def run(device):
    device = wp.get_device(device)
    rng = np.random.default_rng(123)

    geo = fem.Grid2D(
        res=wp.vec2i(4, 2),
        bounds_lo=wp.vec2d(0.0, 0.0),
        bounds_hi=wp.vec2d(1.0, 1.5),
        scalar_type=wp.float64,
    )
    space = fem.make_polynomial_space(geo, degree=DEGREE, discontinuous=True, dtype=wp.float64)
    domain = fem.Cells(geometry=geo)
    test_field = fem.make_test(space=space, domain=domain)
    quadrature = fem.RegularQuadrature(domain, order=2 * DEGREE)

    u = space.make_field()
    u.dof_values = wp.array(rng.uniform(-1.0, 1.0, size=space.node_count()), dtype=wp.float64, device=device)

    # --- Oracle: naive integrate() --------------------------------------
    result_naive = fem.integrate(
        mass_form,
        fields={"u": u, "v": test_field},
        quadrature=quadrature,
        output_dtype=wp.float64,
        device=device,
    )
    wp.synchronize_device(device)
    result_naive_np = result_naive.numpy()

    # --- Sanity: quadrature point ordering matches the 1D Kronecker order
    qpoints_1d = default_quadrature_points(DEGREE)
    expected = np.stack([np.repeat(qpoints_1d, Q), np.tile(qpoints_1d, Q)], axis=-1)  # lexicographic, x slowest
    actual = np.array([[p[0], p[1]] for p in quadrature.points])
    np.testing.assert_allclose(actual, expected, atol=1e-14)

    # --- 1D operators -----------------------------------------------------
    nodes_1d = default_basis_nodes(DEGREE)
    interp_np = build_interpolation_matrix(nodes_1d, qpoints_1d)
    # Derivative matrix only feeds the (identically zero) f1 contraction here
    deriv_np = build_derivative_matrix(nodes_1d, qpoints_1d)

    # --- Per-element DOF packing (host-side gather via topology map) -----
    with wp.ScopedDevice(device):
        node_indices = space.topology.element_node_indices().numpy()  # (E, n*n)
    u_dofs_elem = u.dof_values.numpy()[node_indices]  # (E, n*n) lexicographic (i slow, j fast)

    # --- Build the fused kernel through the integrand machinery ----------
    fields = {"u": u, "v": test_field}
    arguments = _parse_integrand_arguments(mass_form, fields)
    _check_field_compat(mass_form, arguments, domain)

    u_injected = ValueInjectedField.from_field(u, domain)
    field_args = dict(arguments.field_args)
    field_args[arguments.test_name] = SeedField.from_field(test_field)
    field_args["u"] = u_injected
    arguments = arguments._replace(field_args=field_args)

    _find_integrand_operators(mass_form, field_args)
    _notify_operator_usage(mass_form, field_args)

    FieldStruct = _gen_field_struct(arguments.field_args)
    ValueStruct = cache.get_argument_struct(arguments.value_args)
    integrand_func = IntegrandTransformer.apply(mass_form, field_args, sample_type=domain.geometry.sample_type)

    kernel = build_sumfac_kernel(
        mass_form, domain, quadrature, arguments, FieldStruct, ValueStruct, integrand_func, u_injected
    )

    # --- Launch ------------------------------------------------------------
    field_struct = FieldStruct()
    for name, f in field_args.items():
        f.fill_eval_arg(getattr(field_struct, name), device=device)
    value_struct = ValueStruct()

    num_elements = domain.element_count()
    interp_wp = wp.array(interp_np, dtype=wp.float64, device=device)
    deriv_wp = wp.array(deriv_np, dtype=wp.float64, device=device)
    u_dofs_wp = wp.array(u_dofs_elem, dtype=wp.float64, device=device)
    result_elem = wp.zeros((num_elements, N * N), dtype=wp.float64, device=device)

    wp.launch_tiled(
        kernel,
        dim=[num_elements],
        inputs=[
            quadrature.arg_value(device),
            domain.element_arg_value(device),
            domain.element_index_arg_value(device),
            field_struct,
            value_struct,
            interp_wp,
            deriv_wp,
            u_dofs_wp,
            result_elem,
        ],
        block_dim=_default_block_dim(device),
        device=device,
    )
    wp.synchronize_device(device)

    # --- Scatter per-element residuals to node order and compare ----------
    result_sumfac_np = np.zeros(space.node_count())
    np.add.at(result_sumfac_np, node_indices.ravel(), result_elem.numpy().ravel())

    abs_err = np.max(np.abs(result_sumfac_np - result_naive_np))
    rel_err = abs_err / np.max(np.abs(result_naive_np))
    print(f"[{device}] max |sumfac - naive| = {abs_err:.3e}  (rel {rel_err:.3e})")
    print(f"[{device}] naive    range: [{result_naive_np.min():+.6e}, {result_naive_np.max():+.6e}]")
    print(f"[{device}] sumfac   range: [{result_sumfac_np.min():+.6e}, {result_sumfac_np.max():+.6e}]")
    np.testing.assert_allclose(result_sumfac_np, result_naive_np, rtol=1e-12, atol=1e-13)
    print(f"[{device}] PASS: fused B^T D B kernel matches naive integrate()")


if __name__ == "__main__":
    wp.init()
    run("cpu")
    if wp.is_cuda_available():
        run("cuda:0")
