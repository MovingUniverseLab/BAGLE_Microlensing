"""JAX linear-sum assignment for image-track matching.

Adapted from microlux (Ren and Zhu 2025; https://github.com/CoastEgo/microlux),
itself based on https://github.com/google/jax/issues/10403. Used by FSBL d2
for Jonker–Volgenant matching across limb samples (parity-weighted cost).

Microlux is MIT-licensed:

Copyright (c) 2024 Haibin Ren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp


def find_nearest(array1, parity1, array2, parity2, parity_weight=5.0):
    """
    Match images in ``array1`` onto ``array2`` by linear-sum assignment.

    Parameters
    ----------
    array1 : jnp.ndarray, complex, shape = (N_images,)
        Previous limb image positions.
    parity1 : jnp.ndarray, float, shape = (N_images,)
        Parities for ``array1``.
    array2 : jnp.ndarray, complex, shape = (N_images,)
        Current limb image positions.
    parity2 : jnp.ndarray, float, shape = (N_images,)
        Parities for ``array2``.
    parity_weight : float, optional
        Cost weight on parity mismatch (microlux default ``K = 5``).

    Returns
    -------
    col_idx : jnp.ndarray, int, shape = (N_images,)
        Indices into ``array2`` aligned with ``array1``.
    """
    # Cost: geometric distance plus a parity penalty.
    cost = (
        jnp.abs(array2 - array1[:, None])
        + jnp.abs(parity2 - parity1[:, None]) * parity_weight
    )
    cost = jnp.where(jnp.isnan(cost), 100.0, cost)
    _row_ind, col_idx = solve(cost)

    return col_idx


@jax.jit
def augmenting_path(cost, u, v, path, row4col, i):
    """
    One augmenting-path search for the Jonker–Volgenant solver.

    Parameters
    ----------
    cost : jnp.ndarray, float, shape = (N, M)
        Assignment cost matrix.
    u, v : jnp.ndarray, float
        Dual variables for rows and columns.
    path, row4col : jnp.ndarray, int
        Path and column-to-row maps.
    i : int
        Current row.

    Returns
    -------
    sink, minVal, remaining, SR, SC, shortestPathCosts, path
        Updated path-search state (same layout as microlux).
    """
    minVal = 0
    remaining = jnp.arange(cost.shape[1])[::-1]
    num_remaining = cost.shape[1]
    SR = jnp.full(cost.shape[0], False)
    SC = jnp.full(cost.shape[1], False)
    shortestPathCosts = jnp.full(cost.shape[1], jnp.inf)

    sink = -1
    break_cond = False

    def cond_fun(carry):
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ) = carry
        return (sink == -1) & (~break_cond)

    def while_loop_body(carry):
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ) = carry
        index = -1
        lowest = jnp.inf
        SR = SR.at[i].set(True)

        def body_fun(carry):
            (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                it,
            ) = carry
            j = remaining[it]
            r = minVal + cost[i, j] - u[i] - v[j]
            path = lax.cond(
                r < shortestPathCosts[j], lambda: path.at[j].set(i), lambda: path
            )
            shortestPathCosts = shortestPathCosts.at[j].min(r)

            index = lax.cond(
                (shortestPathCosts[j] < lowest)
                | ((shortestPathCosts[j] == lowest) & (row4col[j] == -1)),
                lambda: it,
                lambda: index,
            )
            lowest = jnp.minimum(lowest, shortestPathCosts[j])
            it += 1
            return (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                it,
            )

        carry = lax.while_loop(
            lambda x: x[-1] < num_remaining,
            body_fun,
            (
                cost,
                u,
                v,
                path,
                row4col,
                i,
                remaining,
                minVal,
                shortestPathCosts,
                lowest,
                index,
                0,
            ),
        )
        (
            cost,
            u,
            v,
            path,
            row4col,
            i,
            remaining,
            minVal,
            shortestPathCosts,
            lowest,
            index,
            _,
        ) = carry
        minVal = lowest

        def True_fun(carry):
            remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
            sink = -1
            break_cond = True
            return (remaining, index, row4col, sink, i, SC, num_remaining, break_cond)

        def False_fun(carry):
            remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
            j = remaining[index]

            pred = row4col[j] == -1
            sink = lax.cond(pred, lambda: j, lambda: sink)
            i = lax.cond(~pred, lambda: row4col[j], lambda: i)

            SC = SC.at[j].set(True)
            num_remaining -= 1
            remaining = remaining.at[index].set(remaining[num_remaining])
            return (remaining, index, row4col, sink, i, SC, num_remaining, break_cond)

        carry = lax.cond(
            minVal == jnp.inf,
            True_fun,
            False_fun,
            (remaining, index, row4col, sink, i, SC, num_remaining, break_cond),
        )
        remaining, index, row4col, sink, i, SC, num_remaining, break_cond = carry
        return (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        )

    carry = lax.while_loop(
        cond_fun,
        while_loop_body,
        (
            sink,
            minVal,
            remaining,
            SR,
            SC,
            shortestPathCosts,
            path,
            break_cond,
            i,
            u,
            v,
            row4col,
            cost,
            num_remaining,
        ),
    )
    (
        sink,
        minVal,
        remaining,
        SR,
        SC,
        shortestPathCosts,
        path,
        break_cond,
        i,
        u,
        v,
        row4col,
        cost,
        num_remaining,
    ) = carry
    return sink, minVal, remaining, SR, SC, shortestPathCosts, path


@jax.jit
def solve(cost):
    """
    Solve the linear sum assignment problem (square cost matrix).

    Parameters
    ----------
    cost : jnp.ndarray, float, shape = (N, N)
        Assignment cost matrix.

    Returns
    -------
    row_ind : jnp.ndarray, int, shape = (N,)
        Row indices ``0 .. N-1``.
    col_ind : jnp.ndarray, int, shape = (N,)
        Column assignment for each row.
    """
    u = jnp.full(cost.shape[0], 0.0)
    v = jnp.full(cost.shape[1], 0.0)
    path = jnp.full(cost.shape[1], -1)
    col4row = jnp.full(cost.shape[0], -1)
    row4col = jnp.full(cost.shape[1], -1)

    def loop_body(carry, curRow):
        u, v, path, col4row, row4col, cost = carry
        j, minVal, remaining, SR, SC, shortestPathCosts, path = augmenting_path(
            cost, u, v, path, row4col, curRow
        )

        u = u.at[curRow].add(minVal)

        mask = SR & (jnp.arange(cost.shape[0]) != curRow)
        u = jnp.where(mask, u + minVal - shortestPathCosts[col4row], u)
        v = jnp.where(SC, v + shortestPathCosts - minVal, v)

        def while_loop_body(carry):
            path, j, row4col, col4row, break_cond = carry

            i = path[j]

            row4col = row4col.at[j].set(i)

            col4row, j = col4row.at[i].set(j), col4row[i]

            break_cond = ~(i == curRow)
            return (path, j, row4col, col4row, break_cond)

        carry = lax.while_loop(
            lambda x: x[-1], while_loop_body, (path, j, row4col, col4row, True)
        )
        path, j, row4col, col4row, break_cond = carry
        return (u, v, path, col4row, row4col, cost), curRow

    carry, _ = lax.scan(
        loop_body, (u, v, path, col4row, row4col, cost), jnp.arange(cost.shape[0])
    )
    u, v, path, col4row, row4col, cost = carry
    return jnp.arange(cost.shape[0]), col4row
