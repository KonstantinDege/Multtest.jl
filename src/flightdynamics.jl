# Flight dynamics utilities: trim, linearisation and modal analysis.
#
# A steady glide looks like it is going somewhere, but it is a genuine
# equilibrium of every state except position: the world-frame velocity, the
# attitude and the body rates are all constant, and only `r_0` integrates. So
# trim is an ordinary root find on the model's own right-hand side rather than
# something that has to be approached by simulating and hoping it settles.
#
# That distinction matters. An aircraft with a divergent phugoid or no fin will
# never settle, so "simulate for 250 s and linearise" silently linearises about
# an arbitrary point on a wandering trajectory. Everything here works from the
# root find instead.
#
# These operate on the compiled `ODEProblem`, so they are independent of the
# Dyad model's internals; they only assume the free-flying base is a
# `MultibodyComponents.Body` carrying `r_0`, `v_0`, `w_a` and a quaternion
# `Q_hat`, and that the frame convention is x forward, y up, z right.
#
# AUXILIARY STATES. An airframe is not always the whole state vector. A
# propeller carries its shaft speed, and that shaft has to be in equilibrium
# too or the "trim" is nothing of the sort --- leave it out and the solver
# happily returns an answer with the propeller stopped, which silently removes
# both the thrust and the windmilling drag. Pass such states as `aux` and they
# are solved for alongside the flight variables. `trim_glide` returns the full
# trimmed state vector `u`, and every function downstream takes that rather
# than rebuilding it, so the two can never drift apart.

using LinearAlgebra
using ModelingToolkit: unknowns
import SymbolicIndexingInterface as SII

"""
    StateLayout(sys, body)

Index of each base-body state inside the solver's state vector. Resolved by
symbol, so it survives any reordering the compiler chooses.
"""
struct StateLayout
    n::Int
    v0::Vector{Int}
    w::Vector{Int}
    q::Vector{Int}
    r::Vector{Int}
end

function StateLayout(sys, body)
    # Resolve through SymbolicIndexingInterface rather than matching against
    # `unknowns`: `body.v_0[i]` is a wrapped `Num` and the unknowns are raw
    # symbolics, and any attempt to unwrap the list with a broadcast hits the
    # broadcasting that symbolic arrays overload rather than mapping elementwise.
    # An explicit check, not `something(idx, error(...))`: Julia evaluates both
    # arguments of `something` before calling it, so the error would fire
    # unconditionally.
    function ix(v)
        i = SII.variable_index(sys, v)
        isnothing(i) && error("state $v is not an unknown of the simplified system; " *
                              "is `body` the free-flying base with a quaternion orientation state?")
        i
    end
    StateLayout(length(unknowns(sys)),
                [ix(body.v_0[i]) for i in 1:3],
                [ix(body.w_a[i]) for i in 1:3],
                [ix(body.Q_hat[i]) for i in 1:4],
                [ix(body.r_0[i]) for i in 1:3])
end

"""
    state_index(sys, v)

Index of any other state, for passing to `trim_glide` as an auxiliary unknown.
Use it for a propeller shaft speed: `state_index(sys, sys.glider.propeller.n)`.
"""
function state_index(sys, v)
    i = SII.variable_index(sys, v)
    isnothing(i) && error("state $v is not an unknown of the simplified system")
    i
end

"""
    glide_state(L, V, gamma, theta; altitude, aux, aux_vals)

State vector for a symmetric, wings-level glide: airspeed `V` along a flight
path `gamma`, pitch attitude `theta`, zero body rates and no sideslip. The
incidence that results is `theta - gamma`.

Pitch is a rotation about the body z axis, so the attitude quaternion
`[w,i,j,k]` is `[cos(theta/2), 0, 0, sin(theta/2)]`. Any auxiliary states are
written in at the indices given by `aux`.
"""
function glide_state(L::StateLayout, V, gamma, theta; altitude = 500.0,
                     aux = Int[], aux_vals = Float64[])
    u = zeros(L.n)
    u[L.v0[1]] = V * cos(gamma)
    u[L.v0[2]] = V * sin(gamma)
    u[L.r[2]]  = altitude
    u[L.q[1]]  = cos(theta / 2)
    u[L.q[4]]  = sin(theta / 2)
    for (k, i) in pairs(aux)
        u[i] = aux_vals[k]
    end
    u
end

"""
    trim_glide(prob, L, V; control, aux, aux0, x0, altitude, tol, maxiter)

Find the steady flight condition at airspeed `V`: solve for the flight path
angle, the pitch attitude, the trim `control` (a symbolic parameter, normally
the elevator) and any auxiliary states, such that the two velocity residuals,
the pitch acceleration and the auxiliary derivatives all vanish.

Three unknowns plus one per auxiliary state, against the same number of
equations, so the trim is unique for a given `V`; sweep `V` to walk the polar.

Returns a named tuple carrying the solution, the full trimmed state vector `u`,
the residual norm and a `converged` flag --- **check it**. Below the stall the
required lift coefficient exceeds `CL_max`, no trim exists, and Newton will
happily wander off to a nonsensical answer; a continuation sweep that feeds the
previous solution forward will then poison every point after it.
"""
function trim_glide(prob, L::StateLayout, V;
                    control, x0 = [-0.02, 0.03, 0.0], altitude = 500.0,
                    aux = Int[], aux0 = zeros(length(aux)),
                    tol = 1e-11, maxiter = 80,
                    gamma_max = 0.5, control_max = 0.5)
    f! = prob.f
    du = zeros(L.n)
    na = length(aux)
    nx = 3 + na

    function state_of(x)
        glide_state(L, V, x[1], x[2]; altitude, aux, aux_vals = x[4:end])
    end
    function residual(x)
        prob.ps[control] = x[3]
        f!(du, state_of(x), prob.p, 0.0)
        vcat([du[L.v0[1]], du[L.v0[2]], du[L.w[3]]], [du[i] for i in aux])
    end

    x = vcat(collect(float.(x0[1:3])), collect(float.(aux0)))
    r = residual(x)
    iters = 0
    singular = false
    for it in 1:maxiter
        iters = it
        norm(r) < tol && break
        J = zeros(nx, nx)
        for j in 1:nx
            dx = copy(x); h = 1e-7 * max(1.0, abs(x[j])); dx[j] += h
            J[:, j] = (residual(dx) .- r) ./ h
        end
        # A singular Jacobian is a legitimate outcome, not a bug: it is what a
        # control with no authority over the residuals looks like --- asking for
        # an elevator trim on a tailless aircraft, for instance. Report it as a
        # failure to converge rather than letting the factorisation throw.
        step = try
            J \ r
        catch err
            err isa LinearAlgebra.SingularException || rethrow()
            singular = true
            break
        end
        x -= step
        r = residual(x)
    end

    converged = !singular && norm(r) < 1e-9 && abs(x[1]) < gamma_max &&
                abs(x[3]) < control_max && all(isfinite, x)
    prob.ps[control] = x[3]
    (gamma = x[1], theta = x[2], control = x[3], aux = x[4:end], x = x,
     u = state_of(x), aux_idx = aux,
     alpha = x[2] - x[1], LD = -1 / tan(x[1]), sink = -V * sin(x[1]),
     residual = norm(r), iterations = iters, converged = converged)
end

"""
    jacobian_at(prob, u, t = 0.0)

Central-difference Jacobian of the right-hand side. Finite differences rather
than AD because the multibody right-hand side is cheap here and this avoids the
AD issues the library warns about.
"""
function jacobian_at(prob, u, t = 0.0)
    n = length(u); f! = prob.f
    J = zeros(n, n); fp = zeros(n); fm = zeros(n)
    for j in 1:n
        h = 1e-7 * max(1.0, abs(u[j]))
        up = copy(u); up[j] += h; f!(fp, up, prob.p, t)
        um = copy(u); um[j] -= h; f!(fm, um, prob.p, t)
        J[:, j] = (fp .- fm) ./ (2h)
    end
    J
end

"""
    longitudinal_indices(L; aux) / lateral_indices(L)

At a symmetric trim the Jacobian block-diagonalises. Splitting it before taking
eigenvalues is what lets the phugoid be identified unambiguously: an aircraft
with a weak fin has lateral modes at a similar frequency, and picking "the
slowest oscillation" out of the full spectrum will sooner or later pick the
wrong one.

Auxiliary states belong to the longitudinal block --- a propeller shaft couples
to airspeed through thrust, and symmetrically, so it has no lateral content.

Position states are excluded --- they are pure integrators that feed nothing
back while the density is uniform, and they would only contribute zero
eigenvalues.
"""
longitudinal_indices(L::StateLayout; aux = Int[]) =
    vcat([L.w[3], L.v0[2], L.v0[1], L.q[4], L.q[1]], collect(aux))
lateral_indices(L::StateLayout) = [L.w[2], L.w[1], L.v0[3], L.q[3], L.q[2]]

"""
    modes(J, idx)

Eigenvalues of a sub-block, as `(lambda, period, zeta)` sorted slowest
oscillation first, with the non-oscillatory roots after them.
"""
function modes(J, idx)
    ev = eigvals(J[idx, idx])
    osc = [e for e in ev if imag(e) > 1e-9]
    real_roots = [e for e in ev if abs(imag(e)) <= 1e-9]
    sort!(osc, by = e -> abs(imag(e)))
    out = [(lambda = e, period = 2pi / imag(e), zeta = -real(e) / abs(e)) for e in osc]
    append!(out, [(lambda = e, period = Inf, zeta = sign(-real(e))) for e in real_roots])
    out
end

"""
    trim_and_modes(prob, L, V; control, aux, aux0, x0, altitude)

Trim at `V`, then report the longitudinal and lateral modes there. The phugoid
is the first longitudinal entry and the short period the last oscillatory one.
"""
function trim_and_modes(prob, L::StateLayout, V; control, x0 = [-0.02, 0.03, 0.0],
                        aux = Int[], aux0 = zeros(length(aux)), altitude = 500.0)
    tr = trim_glide(prob, L, V; control, x0, altitude, aux, aux0)
    tr.converged || return (trim = tr, lon = nothing, lat = nothing, J = nothing)
    J = jacobian_at(prob, tr.u)
    (trim = tr, lon = modes(J, longitudinal_indices(L; aux)),
     lat = modes(J, lateral_indices(L)), J = J)
end

"""
    stability_derivatives(prob, L, tr, V; control)

Dimensional longitudinal derivatives in body axes at a trim point, by
perturbing the body-axis velocity components and the pitch rate about the
trimmed state `tr.u` --- auxiliary states included, so a propeller stays at its
trimmed shaft speed rather than being reset.

`w` is the *downward* body velocity, so `w = -v_body_y` in this frame's
x-forward, y-up convention, and `Z` is positive down. These are the quantities
to compare against handbook expressions: `Z_w` against `-rho*V*S*a/(2m)` and
`M_q` against `-rho*V*S_t*l_t^2*a_t/(2*I_pitch)`.
"""
function stability_derivatives(prob, L::StateLayout, tr, V; control)
    prob.ps[control] = tr.control
    th = tr.theta
    R  = [cos(th) sin(th) 0.0; -sin(th) cos(th) 0.0; 0.0 0.0 1.0]  # world -> body
    ub = V * cos(tr.alpha); wb = V * sin(tr.alpha)
    du = zeros(L.n)
    function forces(d_u, d_w, d_q)
        v0 = R' * [ub + d_u, -(wb + d_w), 0.0]
        u = copy(tr.u)
        u[L.v0[1]] = v0[1]; u[L.v0[2]] = v0[2]; u[L.v0[3]] = v0[3]
        u[L.w[3]]  = d_q
        prob.f(du, u, prob.p, 0.0)
        ab = R * [du[L.v0[1]], du[L.v0[2]], du[L.v0[3]]]
        (X = ab[1], Z = -ab[2], M = du[L.w[3]])
    end
    h = 1e-4
    d(sel, a, b) = (getfield(a, sel) - getfield(b, sel)) / (2h)
    pu, mu = forces(h, 0, 0), forces(-h, 0, 0)
    pw, mw = forces(0, h, 0), forces(0, -h, 0)
    pq, mq = forces(0, 0, h), forces(0, 0, -h)
    (Xu = d(:X, pu, mu), Zu = d(:Z, pu, mu), Mu = d(:M, pu, mu),
     Xw = d(:X, pw, mw), Zw = d(:Z, pw, mw), Mw = d(:M, pw, mw),
     Mq = d(:M, pq, mq))
end

"""
    lateral_derivatives(prob, L, tr, V; control)

Dimensional lateral derivatives at a trim point, by perturbing the sideslip
velocity and the roll and yaw rates about the trimmed state.

Sign conventions in this frame (x forward, y up, z right) are not the textbook
ones and are worth stating: `v` is the body z velocity, positive to the right;
a positive roll rate `w_a[1]` is right wing **down**; a positive yaw rate
`w_a[2]` is nose **left**. So stability reads as

* `Nv < 0` --- directional stiffness, the aircraft weathercocks into the slip,
* `Lv < 0` --- dihedral effect, sideslip to the right rolls it left,
* `Lp < 0` --- roll damping,
* `Nr < 0` --- yaw damping.

`Lp` and `Lv` are produced entirely by where the wing panels sit; nothing
supplies them as coefficients.
"""
function lateral_derivatives(prob, L::StateLayout, tr, V; control)
    prob.ps[control] = tr.control
    th = tr.theta
    R  = [cos(th) sin(th) 0.0; -sin(th) cos(th) 0.0; 0.0 0.0 1.0]
    ub = V * cos(tr.alpha); wb = V * sin(tr.alpha)
    du = zeros(L.n)
    function acc(dv, dp, dr)
        v0 = R' * [ub, -wb, dv]
        u = copy(tr.u)
        u[L.v0[1]] = v0[1]; u[L.v0[2]] = v0[2]; u[L.v0[3]] = v0[3]
        u[L.w[1]] = dp; u[L.w[2]] = dr
        prob.f(du, u, prob.p, 0.0)
        (roll = du[L.w[1]], yaw = du[L.w[2]],
         side = (R * [du[L.v0[1]], du[L.v0[2]], du[L.v0[3]]])[3])
    end
    h = 1e-4
    d(s, a, b) = (getfield(a, s) - getfield(b, s)) / (2h)
    pv, mv = acc(h, 0, 0), acc(-h, 0, 0)
    pp, mp = acc(0, h, 0), acc(0, -h, 0)
    pr, mr = acc(0, 0, h), acc(0, 0, -h)
    (Yv = d(:side, pv, mv), Lv = d(:roll, pv, mv), Nv = d(:yaw, pv, mv),
     Lp = d(:roll, pp, mp), Np = d(:yaw, pp, mp),
     Lr = d(:roll, pr, mr), Nr = d(:yaw, pr, mr))
end

"""
    glide_polar(prob, L, Vs; control, aux, aux0, altitude)

Trim across a range of airspeeds by continuation, feeding each solution forward
as the next guess. Points that fail to converge (below the stall, typically)
come back with `converged = false` and do **not** seed the next point.
"""
function glide_polar(prob, L::StateLayout, Vs; control, x0 = [-0.02, 0.03, 0.0],
                     aux = Int[], aux0 = zeros(length(aux)), altitude = 500.0)
    seed = collect(float.(x0[1:3]))
    seed_aux = collect(float.(aux0))
    map(Vs) do V
        tr = trim_glide(prob, L, V; control, x0 = seed, aux, aux0 = seed_aux, altitude)
        if tr.converged
            seed = tr.x[1:3]
            seed_aux = tr.aux
        end
        (V = V, trim = tr)
    end
end
