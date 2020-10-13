import torch
from torch import nn
from torch.nn import functional as F


EPS = 1e-10


def regular_control(n_control):
    control = torch.linspace(-1, 1, n_control)
    c1 = control.view(1, n_control).expand(n_control, -1).reshape(-1)
    c2 = control.view(n_control, 1).expand(-1, n_control).reshape(-1)
    control = torch.stack((c1, c2), -1)

    return control


def regular_grid(height, width):
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))

    return torch.stack([yy, xx], 2).view(-1, 2)


def cross_sq_dist_mat(x, y):
    x_norm_sq = x.pow(2).sum(2)
    y_norm_sq = y.pow(2).sum(2)

    x_norm_sq_tile = x_norm_sq.unsqueeze(2)
    y_norm_sq_tile = y_norm_sq.unsqueeze(1)

    x_y_t = x @ y.transpose(1, 2)

    sq_dist = x_norm_sq_tile - 2 * x_y_t + y_norm_sq_tile

    return sq_dist


def pairwise_sq_dist_mat(x):
    x_x_t = x @ x.transpose(1, 2)
    x_norm_sq = torch.diagonal(x_x_t, dim1=1, dim2=2)
    x_norm_sq_tile = x_norm_sq.unsqueeze(2)

    sq_dist = x_norm_sq_tile - 2 * x_x_t + x_norm_sq_tile.transpose(1, 2)

    return sq_dist


def phi(r, order):
    r = r.clamp(min=EPS)

    if order == 1:
        r = torch.sqrt(r)

        return r

    elif order == 2:
        return 0.5 * r * torch.log(r)

    elif order == 4:
        return 0.5 * r.pow(2) * torch.log(r)

    elif order % 2 == 0:
        return 0.5 * r.pow(0.5 * order) * torch.log(r)

    else:
        return r.pow(0.5 * order)


def solve_interpolation(train_points, train_values, order, regularization_weight):
    b, n, d = train_points.shape
    k = train_values.shape[2]

    c = train_points
    f = train_values

    mat_a = phi(pairwise_sq_dist_mat(c), order)

    if regularization_weight > 0:
        batch_eye_mat = torch.eye(
            n, dtype=train_points.dtype, device=train_points.device
        ).unsqueeze(0)
        mat_a += regularization_weight * batch_eye_mat

    ones = c.new_ones(b, n, 1)
    mat_b = torch.cat([c, ones], 2)

    left = torch.cat([mat_a, mat_b.transpose(1, 2)], 1)

    n_b_cols = mat_b.shape[2]
    lhs_zeros = train_points.new_zeros(b, n_b_cols, n_b_cols)
    right = torch.cat([mat_b, lhs_zeros], 1)
    lhs = torch.cat([left, right], 2)

    rhs_zeros = train_points.new_zeros(b, d + 1, k)
    rhs = torch.cat([f, rhs_zeros], 1)

    w_v = torch.solve(rhs, lhs).solution
    w = w_v[:, :n, :]
    v = w_v[:, n:, :]

    return w, v


def apply_interpolation(query_points, train_points, w, v, order):
    pairwise_dist = cross_sq_dist_mat(query_points, train_points)
    phi_pairwise_dist = phi(pairwise_dist, order)

    rbf = phi_pairwise_dist @ w

    b, m, d = query_points.shape

    query_points_pad = torch.cat([query_points, query_points.new_ones(b, m, 1)], 2)
    linear = query_points_pad @ v

    print(phi_pairwise_dist.shape, query_points_pad.shape, w.shape, v.shape)

    alt1 = torch.cat([phi_pairwise_dist, query_points_pad], 2)
    alt2 = torch.cat([w, v], 1)
    alt = alt1 @ alt2

    print((rbf + linear - alt).abs().max())

    return rbf + linear


def interpolate_spline(
    train_points, train_values, query_points, order=2, regularization_weight=0
):
    w, v = solve_interpolation(train_points, train_values, order, regularization_weight)
    query_values = apply_interpolation(query_points, train_points, w, v, order)

    return query_values


def solve_interpolation_precomputed(train_points, order, regularization_weight):
    b, n, d = train_points.shape

    c = train_points

    mat_a = phi(pairwise_sq_dist_mat(c), order)

    if regularization_weight > 0:
        batch_eye_mat = torch.eye(
            n, dtype=train_points.dtype, device=train_points.device
        ).unsqueeze(0)
        mat_a += regularization_weight * batch_eye_mat

    ones = c.new_ones(b, n, 1)
    mat_b = torch.cat([c, ones], 2)

    left = torch.cat([mat_a, mat_b.transpose(1, 2)], 1)

    n_b_cols = mat_b.shape[2]
    lhs_zeros = train_points.new_zeros(b, n_b_cols, n_b_cols)
    right = torch.cat([mat_b, lhs_zeros], 1)
    lhs = torch.cat([left, right], 2)

    # w_inv = torch.inverse(lhs)

    return lhs


def apply_interpolation_precomputed(phi_pairwise_query_points, train_values_pad, w_inv):
    # w = w_inv @ rhs
    w = torch.solve(train_values_pad, w_inv).solution
    rbf_linear = phi_pairwise_query_points @ w

    return rbf_linear


class InterpolateSpline(nn.Module):
    def __init__(self, train_points, query_points, order, regularization_weight=0):
        super().__init__()

        train_points = train_points.unsqueeze(0)
        query_points = query_points.unsqueeze(0)

        kernel = solve_interpolation_precomputed(
            train_points, order, regularization_weight
        )

        self.train_values_pad = train_points.shape[2] + 1

        pairwise_dist = cross_sq_dist_mat(query_points, train_points)
        phi_pairwise_dist = phi(pairwise_dist, order)
        query_points_pad = F.pad(query_points, (0, 1), value=1)
        phi_pairwise_query_points = torch.cat([phi_pairwise_dist, query_points_pad], 2)

        self.register_buffer("kernel", kernel)
        self.register_buffer("phi_pairwise_query_points", phi_pairwise_query_points)

    def forward(self, train_values):
        train_values_pad = F.pad(train_values, (0, 0, 0, self.train_values_pad))
        query_values = apply_interpolation_precomputed(
            self.phi_pairwise_query_points, train_values_pad, self.kernel
        )

        return query_values
