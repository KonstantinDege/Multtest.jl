# Regression tests for the aerodynamic components and the glider.
#
# Two kinds of assertion appear here and they are deliberately kept apart:
#
#   * PHYSICS   --- relations that must hold whatever the parameters are, such
#                   as a derivative scaling linearly with airspeed, or a force
#                   resolving to exactly -D along the relative wind. These have
#                   tight tolerances and should never need touching.
#   * ANCHOR    --- specific numbers for the glider as currently configured.
#                   These are here to catch unintended change. If a parameter is
#                   deliberately altered, expect these to move and update them.
#
# Several checks are written to be independent of the assembly's moments of
# inertia, which are not directly available from the simplified system: ratios
# such as `Lp/Lv` cancel the inertia and test the panel placement on its own.

using Multtest
using Test
using ModelingToolkit
using MultibodyComponents
using MultibodyComponents: multibody
using OrdinaryDiffEqDefault
using LinearAlgebra

const FD = Multtest

build(nm::Symbol) = multibody(getfield(Multtest, nm)(; name = nm))
sim(sys, T; kw...) = solve(ODEProblem(sys, [], (0.0, T)), reltol = 1e-10, abstol = 1e-10; kw...)

# Reference geometry, mirroring the defaults in dyad/glider.dyad.
const M_AC   = 350.0                 # total mass  [kg]
const G      = 9.80665
const RHO    = 1.225
const S_W    = 10.5                  # wing area   [m^2]
const B_W    = 15.0                  # wing span   [m]
const AR_W   = B_W^2 / S_W
const DIH    = 0.052                 # wing dihedral [rad]
const Y_PAN  = B_W / (2 * sqrt(3.0)) # spanwise station of the wing panels
const A_W    = 5.7                   # wing lift curve slope [1/rad]
const A_TOT  = A_W + 4.4 / S_W + 2.1 / S_W   # aircraft lift slope, wing + tail + body

# ---------------------------------------------------------------------------
@testset "structure: every harness reduces to a pure ODE" begin
    expected = Dict(:TestAeroSweep => 2, :TestAeroSpeedRamp => 2,
                    :TestDragTerminal => 13, :TestDragAnisotropy => 2,
                    :TestAtmosphereWind => 3, :TestGliderWingOnly => 13,
                    :TestGliderNoFin => 13, :TestGlider => 13,
                    :TestGliderAileronStep => 13)
    for (nm, n) in expected
        sys = build(nm)
        @test length(unknowns(sys)) == n
        @test all(ModelingToolkit.is_diff_equation, equations(sys))   # PHYSICS: no algebraic loop
        @test sim(sys, 2.0).retcode == ReturnCode.Success
    end
end

# ---------------------------------------------------------------------------
@testset "AeroSurface coefficients" begin
    sys = build(:TestAeroSweep)
    sol = sim(sys, 1.0)
    # The harness sweeps the velocity direction through a full turn over 1 s,
    # so incidence = -pi + 2*pi*t. Recover the time for a given incidence.
    tof(adeg) = 0.5 + deg2rad(adeg) / (2pi)
    at(adeg, v) = sol(tof(adeg), idxs = v)

    # harness panel: CL0 = 0.2, CL_alpha = 5.5, CD0 = 0.012, e = 0.9, AR = 21.43
    CL0, CLa, CD0, e_os = 0.2, 5.5, 0.012, 0.9
    AR = 15.0^2 / 10.5

    # PHYSICS: well inside the attached regime the coefficients are exactly
    # CL0 + CL_alpha*sin(a) on the parabolic polar. Kept below 6 degrees, where
    # the separation blend is still negligible.
    attachedCL(adeg) = CL0 + CLa * sind(adeg)
    attachedCD(adeg) = CD0 + attachedCL(adeg)^2 / (pi * e_os * AR)
    for adeg in (-6.0, -3.0, 0.0, 3.0, 6.0)
        @test at(adeg, sys.surf.CL) ≈ attachedCL(adeg) rtol = 5e-3
        @test at(adeg, sys.surf.CD) ≈ attachedCD(adeg) rtol = 5e-3
    end

    # PHYSICS: approaching the stall the blend starts to bite, and it can only
    # add drag and remove lift relative to the attached formula
    for adeg in (8.0, 10.0, 12.0)
        @test at(adeg, sys.surf.sigma) > 0
        @test at(adeg, sys.surf.CD) > attachedCD(adeg)
        @test at(adeg, sys.surf.CL) < attachedCL(adeg)
    end
    @test at(8.0, sys.surf.CD) ≈ attachedCD(8.0) rtol = 2e-2   # still small at 8 deg
    # PHYSICS: lift peaks and falls back, i.e. the surface actually stalls
    CLpeak = maximum(at(a, sys.surf.CL) for a in 6.0:0.5:20.0)
    @test CLpeak > at(6.0, sys.surf.CL)
    @test at(20.0, sys.surf.CL) < CLpeak

    # PHYSICS: flat-plate asymptotes well past the stall
    @test at(90.0, sys.surf.CL) ≈ 0.0 atol = 1e-6
    @test at(90.0, sys.surf.CD) ≈ CD0 + 2.0 rtol = 1e-3
    @test at(-90.0, sys.surf.CD) ≈ CD0 + 2.0 rtol = 1e-3
    @test at(45.0, sys.surf.CL) ≈ 2 * sind(45)^2 * cosd(45) rtol = 1e-3
    @test at(-45.0, sys.surf.CL) ≈ -2 * sind(45)^2 * cosd(45) rtol = 1e-3

    # PHYSICS: reversed flow is finite and sheds its lift
    @test at(180.0, sys.surf.CL) ≈ 0.0 atol = 1e-6
    @test at(180.0, sys.surf.CD) ≈ CD0 rtol = 1e-2

    # PHYSICS: the force resolves to exactly -D along the relative wind and +L
    # across it, at every incidence including reversed
    for adeg in (-120.0, -30.0, 0.0, 12.0, 60.0, 150.0)
        t = tof(adeg)
        sa = sol(t, idxs = sys.surf.sa); ca = sol(t, idxs = sys.surf.ca)
        n2 = sa^2 + ca^2
        F = [sol(t, idxs = sys.surf.F_s[i]) for i in 1:3]
        @test (F[1] * ca - F[2] * sa) / n2 ≈ -sol(t, idxs = sys.surf.D) rtol = 1e-8
        @test (F[1] * sa + F[2] * ca) / n2 ≈  sol(t, idxs = sys.surf.L) rtol = 1e-8
        @test F[3] == 0                      # strip theory: no spanwise force
    end
end

@testset "AeroSurface is well conditioned at zero airspeed" begin
    sys = build(:TestAeroSpeedRamp)
    sol = sim(sys, 1.0)
    # PHYSICS: no force and no NaN at rest
    @test sol(0.0, idxs = sys.surf.L) == 0.0
    @test sol(0.0, idxs = sys.surf.D) == 0.0
    @test isfinite(sol(0.0, idxs = sys.surf.alpha))
    @test !any(isnan, sol[sys.surf.L])
    @test !any(isnan, sol[sys.surf.CL])
    # PHYSICS: lift recovers the V^2 law once well clear of the v_eps fade
    L1 = sol(0.5, idxs = sys.surf.L); L2 = sol(1.0, idxs = sys.surf.L)
    @test L2 / L1 ≈ 4.0 rtol = 1e-3
end

# ---------------------------------------------------------------------------
@testset "DragBody" begin
    sys = build(:TestDragTerminal)
    sol = sim(sys, 8.0)
    # PHYSICS: terminal velocity solves m*g = 0.5*rho*CdA*sqrt(v^2+veps^2)*v
    f(v) = 0.5 * RHO * 0.5 * sqrt(v^2 + 0.25) * v - 1.0 * G
    lo, hi = 1.0, 20.0
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    @test -sol(8.0, idxs = sys.body.v_0[2]) ≈ (lo + hi) / 2 rtol = 1e-6
    # PHYSICS: a symmetric drop neither drifts nor rotates
    @test sol(8.0, idxs = sys.body.r_0[1]) ≈ 0.0 atol = 1e-9
    @test sol(8.0, idxs = sys.body.r_0[3]) ≈ 0.0 atol = 1e-9
    @test all(abs(sol(8.0, idxs = sys.body.w_a[i])) < 1e-9 for i in 1:3)

    sysa = build(:TestDragAnisotropy)
    sola = sim(sysa, 1.0)
    CdA = (0.08, 1.2, 0.9); V0 = 30.0
    for pd in 0:45:315
        t = pd / 360
        v = [V0 * cosd(pd), V0 * sind(pd), 0.0]
        Vr = sqrt(v[1]^2 + v[2]^2 + 0.25)
        for i in 1:3     # PHYSICS: force is -0.5*rho*|v|*CdA_i*v_i per axis
            @test sola(t, idxs = sysa.drag.F_s[i]) ≈ -0.5 * RHO * Vr * CdA[i] * v[i] atol = 1e-9
        end
        @test all(sola(t, idxs = sysa.drag.frame_a.tau[i]) == 0 for i in 1:3)
    end
end

# ---------------------------------------------------------------------------
@testset "Atmosphere" begin
    sysi = multibody(Multtest.TestAtmosphereISA(; name = :isa))
    soli = solve(ODEProblem(sysi, [], (0.0, 1.0)), reltol = 1e-10, abstol = 1e-10)
    # ANCHOR: ISA troposphere table
    for (h, rho) in ((0, 1.2250), (1000, 1.1117), (2500, 0.9570), (5000, 0.7364))
        @test soli(h / 5000, idxs = sysi.atm.air.rho) ≈ rho atol = 5e-4
    end

    sysw = build(:TestAtmosphereWind)
    solw = sim(sysw, 1.0)
    t = 0.5
    # PHYSICS: the flow-free bus broadcasts identically to every consumer
    for c in (sysw.surf2.air, sysw.drag.air, sysw.atm.air)
        @test solw(t, idxs = c.rho)      == solw(t, idxs = sysw.surf1.air.rho)
        @test solw(t, idxs = c.v_wind_x) == solw(t, idxs = sysw.surf1.air.v_wind_x)
        @test solw(t, idxs = c.v_wind_y) == solw(t, idxs = sysw.surf1.air.v_wind_y)
    end
    # PHYSICS: wind enters the panel's own flow. Ground speed 25 along x, wind
    # [-10, 3, 0], so the panel sees [35, -3, 0]: a headwind raises the dynamic
    # pressure and an updraught raises the incidence.
    @test solw(t, idxs = sysw.surf1.u) ≈ 35.0 rtol = 1e-9
    @test solw(t, idxs = sysw.surf1.alpha) ≈ atan(3.0, 35.0) rtol = 1e-6
    @test solw(t, idxs = sysw.surf1.q) ≈ 0.5 * RHO * (35.0^2 + 3.0^2) rtol = 1e-9
end

# ---------------------------------------------------------------------------
@testset "Downwash" begin
    sys = build(:TestGlider)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)
    tr = FD.trim_glide(prob, L, 25.0; control = sys.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    prob.ps[sys.elev.k] = tr.control
    u = FD.glide_state(L, 25.0, tr.gamma, tr.theta)
    o(x) = prob.f.observed(x)(u, prob.p, 0.0)
    # PHYSICS: far-field lifting line, eps = 2*CL/(pi*AR)
    CLw = o(sys.glider.wing.L_total) / (0.5 * RHO * 25.0^2 * S_W)
    @test o(sys.glider.downwash.w_dw) / 25.0 ≈ 2 * CLw / (pi * AR_W) rtol = 5e-3
    # PHYSICS: the tailplane sees a lower incidence than the wing because of it
    @test o(sys.glider.htail.panel_l.alpha) < o(sys.glider.wing.panel_l.alpha)
end

# ---------------------------------------------------------------------------
@testset "glider geometry" begin
    sys = build(:TestGlider)
    sol = sim(sys, 1e-3)
    g0(x) = sol(0.0, idxs = x)
    # PHYSICS: composite centre of gravity sits on the datum by construction
    rcm = [-(100.0 * 0.145 + 12.0 * (-4.0) + 8.0 * (-3.8)) / 230.0,
           -(100.0 * (Y_PAN * sin(DIH)) + 12.0 * 0.25 + 8.0 * (0.3 + 1.2 / (2 * sqrt(3.0)))) / 230.0,
           0.0]
    num = 230.0 .* rcm .+ 50.0 .* [0.145, Y_PAN * sin(DIH), Y_PAN * cos(DIH)] .+
                          50.0 .* [0.145, Y_PAN * sin(DIH), -Y_PAN * cos(DIH)]
    num .+= 12.0 .* [-4.0, 0.25, 0.0]
    num .+= 8.0 .* [-3.8, 0.3 + 1.2 / (2 * sqrt(3.0)), 0.0]
    @test norm(num ./ M_AC) < 1e-12

    # PHYSICS: wing panels are mirror images with their tips raised by dihedral
    zl = g0(sys.glider.wing.panel_l.frame_a.r_0[3])
    zr = g0(sys.glider.wing.panel_r.frame_a.r_0[3])
    @test zl ≈ -zr rtol = 1e-9
    @test abs(zr) ≈ Y_PAN * cos(DIH) rtol = 1e-6
    @test g0(sys.glider.wing.panel_l.frame_a.r_0[2]) ≈
          g0(sys.glider.wing.panel_r.frame_a.r_0[2]) rtol = 1e-9

    # PHYSICS: the fin's lift acts sideways and its span points down, which is
    # what turns its incidence into sideslip
    @test g0(sys.glider.vtail.panel_l.frame_a.R[2, 3]) ≈ 1.0 atol = 1e-6
    @test g0(sys.glider.vtail.panel_l.frame_a.R[3, 2]) ≈ -1.0 atol = 2e-2
end

# ---------------------------------------------------------------------------
@testset "trim and glide polar" begin
    sys = build(:TestGlider)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)

    tr = FD.trim_glide(prob, L, 25.0; control = sys.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    @test tr.converged
    @test tr.residual < 1e-9
    @test tr.iterations <= 10
    # ANCHOR: the rig is set so the elevator is near neutral at best glide
    @test abs(rad2deg(tr.control)) < 0.5
    @test tr.LD ≈ 32.8 rtol = 0.02

    # PHYSICS: below the stall no trim exists, and the solver must say so
    # rather than returning a nonsense root
    @test !FD.trim_glide(prob, L, 18.0; control = sys.elev.k).converged

    pol = FD.glide_polar(prob, L, 20.5:0.5:40.0; control = sys.elev.k,
                         x0 = [-0.0305, 0.0196, 0.0])
    ok = [p for p in pol if p.trim.converged]
    @test length(ok) > 30
    V   = [p.V for p in ok]
    LD  = [p.trim.LD for p in ok]
    sink = [p.trim.sink for p in ok]
    # ANCHOR: best glide and minimum sink
    @test maximum(LD) ≈ 32.8 rtol = 0.03
    @test 23.0 <= V[argmax(LD)] <= 27.0
    @test minimum(sink) ≈ 0.70 rtol = 0.05
    # PHYSICS: minimum sink is a genuine interior minimum and lies slower than
    # best glide
    @test V[argmin(sink)] < V[argmax(LD)]
    @test argmin(sink) > 1
    # PHYSICS: sink grows monotonically above best glide
    hi = V .> V[argmax(LD)]
    @test issorted(sink[hi])
end

# ---------------------------------------------------------------------------
@testset "longitudinal derivatives and modes" begin
    sys = build(:TestGlider)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)
    seed = [-0.0305, 0.0196, 0.0]
    d = Dict{Float64, Any}()
    for V in (25.0, 35.0)
        tr = FD.trim_glide(prob, L, V; control = sys.elev.k, x0 = seed)
        @test tr.converged
        d[V] = FD.stability_derivatives(prob, L, tr, V; control = sys.elev.k)
    end

    # PHYSICS: Zw = -rho*V*S*a/(2m), which needs no inertia to check
    for V in (25.0, 35.0)
        @test d[V].Zw ≈ -RHO * V * S_W * A_TOT / (2 * M_AC) rtol = 0.03
    end
    # PHYSICS: Mq and Zw both scale linearly with airspeed
    @test d[35.0].Mq / d[25.0].Mq ≈ 35 / 25 rtol = 0.05
    @test d[35.0].Zw / d[25.0].Zw ≈ 35 / 25 rtol = 0.02
    # PHYSICS: pitch stiffness and pitch damping both restoring
    @test d[25.0].Mw < 0
    @test d[25.0].Mq < 0

    tm = FD.trim_and_modes(prob, L, 25.0; control = sys.elev.k, x0 = seed)
    osc = [m for m in tm.lon if isfinite(m.period)]
    @test length(osc) == 2
    phugoid, shortperiod = osc[1], osc[end]
    # ANCHOR: short period is fast and well damped
    @test 3.0 <= shortperiod.period <= 5.5
    @test 0.6 <= shortperiod.zeta <= 0.95
    # PHYSICS: the phugoid is much slower than the short period and only
    # weakly damped either way
    @test phugoid.period > 3 * shortperiod.period
    @test abs(phugoid.zeta) < 0.1

    # PHYSICS: the phugoid period scales with airspeed (Lanchester)
    tm2 = FD.trim_and_modes(prob, L, 35.0; control = sys.elev.k, x0 = seed)
    p2 = [m for m in tm2.lon if isfinite(m.period)][1]
    @test p2.period / phugoid.period ≈ 35 / 25 rtol = 0.10
end

# ---------------------------------------------------------------------------
@testset "lateral derivatives and modes" begin
    sys = build(:TestGlider)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)
    tr = FD.trim_glide(prob, L, 25.0; control = sys.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    ld = FD.lateral_derivatives(prob, L, tr, 25.0; control = sys.elev.k)

    # PHYSICS: all four lateral stability signs
    @test ld.Nv < 0     # weathercock stability
    @test ld.Lv < 0     # dihedral effect
    @test ld.Lp < 0     # roll damping
    @test ld.Nr < 0     # yaw damping

    # PHYSICS: roll damping and dihedral effect both come from the same panel
    # placement, so their ratio is the panel station over the dihedral angle
    # and is independent of the roll inertia. Measured on the finless aircraft
    # so the fin's side force does not enter Lv.
    sysn = build(:TestGliderNoFin)
    probn = ODEProblem(sysn, [], (0.0, 1.0))
    Ln = FD.StateLayout(sysn, sysn.glider.fuselage)
    trn = FD.trim_glide(probn, Ln, 25.0; control = sysn.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    ldn = FD.lateral_derivatives(probn, Ln, trn, 25.0; control = sysn.elev.k)
    @test ldn.Lp / ldn.Lv ≈ Y_PAN / DIH rtol = 0.02

    # PHYSICS: the fin is what provides directional stability and yaw damping
    @test ldn.Nv > 0            # finless aircraft is directionally unstable
    @test abs(ldn.Nr) < 0.05    # and has essentially no yaw damping
    @test ld.Nv < ldn.Nv
    @test ld.Nr < ldn.Nr

    tm = FD.trim_and_modes(prob, L, 25.0; control = sys.elev.k, x0 = tr.x)
    osc = [m for m in tm.lat if isfinite(m.period)]
    reals = [real(m.lambda) for m in tm.lat if !isfinite(m.period)]
    @test length(osc) == 1
    # ANCHOR: dutch roll present and damped
    @test 3.5 <= osc[1].period <= 6.5
    @test osc[1].zeta > 0.15
    # PHYSICS: fast roll subsidence, a neutral heading mode, and a spiral
    @test minimum(reals) < -3.0                       # roll subsidence
    @test count(r -> abs(r) < 1e-3, reals) >= 1       # heading is cyclic
    @test any(r -> 0 < r < 0.5, reals)                # spiral divergence

    # PHYSICS: without a fin the lateral oscillation diverges instead
    tmn = FD.trim_and_modes(probn, Ln, 25.0; control = sysn.elev.k, x0 = trn.x)
    oscn = [m for m in tmn.lat if isfinite(m.period)]
    @test oscn[1].zeta < 0
end

# ---------------------------------------------------------------------------
@testset "configuration comparisons" begin
    # PHYSICS: without a tailplane the pitch stiffness has the wrong sign.
    # Stated as the derivative rather than as a divergence rate, so it does not
    # depend on how the aircraft happens to be released. Evaluated at the launch
    # state directly, because a tailless aircraft has no pitch control and so no
    # trim in the three-unknown sense.
    sysw = build(:TestGliderWingOnly)
    prw = ODEProblem(sysw, [], (0.0, 1.0))
    Lw = FD.StateLayout(sysw, sysw.glider.fuselage)
    th, gam, V = 0.0196, -0.0305, 25.0
    al = th - gam
    Rb = [cos(th) sin(th) 0.0; -sin(th) cos(th) 0.0; 0.0 0.0 1.0]
    ub, wb = V * cos(al), V * sin(al)
    duw = zeros(Lw.n)
    function pitchacc(dw)
        v0 = Rb' * [ub, -(wb + dw), 0.0]
        u = FD.glide_state(Lw, V, gam, th)
        u[Lw.v0[1]] = v0[1]; u[Lw.v0[2]] = v0[2]; u[Lw.v0[3]] = v0[3]
        prw.f(duw, u, prw.p, 0.0)
        duw[Lw.w[3]]
    end
    hw = 1e-4
    @test (pitchacc(hw) - pitchacc(-hw)) / (2hw) > 0      # unstable in pitch

    # PHYSICS: and it therefore departs, leaving the normal incidence range
    # entirely rather than settling into a glide
    solw = sim(sysw, 12.0)
    alw = [rad2deg(solw(t, idxs = sysw.glider.wing.panel_l.alpha)) for t in 0:0.25:12]
    qw  = [rad2deg(solw(t, idxs = sysw.glider.fuselage.w_a[3])) for t in 0:0.25:12]
    @test maximum(abs, alw) > 45.0
    @test maximum(abs, qw) > 60.0

    # PHYSICS: the full aircraft, by contrast, has restoring pitch stiffness
    sysf = build(:TestGlider)
    prf = ODEProblem(sysf, [], (0.0, 1.0))
    Lf = FD.StateLayout(sysf, sysf.glider.fuselage)
    trf = FD.trim_glide(prf, Lf, 25.0; control = sysf.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    @test FD.stability_derivatives(prf, Lf, trf, 25.0; control = sysf.elev.k).Mw < 0

    # ROBUSTNESS: a control with no authority gives a singular Jacobian, which
    # must be reported as non-convergence rather than thrown
    bad = FD.trim_glide(prw, Lw, 25.0; control = sysw.glider.k_downwash)
    @test !bad.converged

    # PHYSICS: downwash reduces the net pitch stiffness, and by more than it
    # reduces the tail term alone, because Mw is a difference of two larger
    # numbers
    sys = build(:TestGlider)
    seed = [-0.0305, 0.0196, 0.0]
    function mw(k)
        pr = ODEProblem(sys, Dict(sys.glider.k_downwash => k), (0.0, 1.0))
        LL = FD.StateLayout(sys, sys.glider.fuselage)
        t = FD.trim_glide(pr, LL, 25.0; control = sys.elev.k, x0 = seed)
        FD.stability_derivatives(pr, LL, t, 25.0; control = sys.elev.k).Mw
    end
    mw_off, mw_on = mw(0.0), mw(1.0)
    @test mw_on < 0 && mw_off < 0
    @test abs(mw_on) < abs(mw_off)
    @test abs(mw_on) / abs(mw_off) < 0.83          # bigger than the 17% tail cut
end

# ---------------------------------------------------------------------------
@testset "aileron" begin
    sys = build(:TestGlider)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)
    tr = FD.trim_glide(prob, L, 25.0; control = sys.elev.k, x0 = [-0.0305, 0.0196, 0.0])
    prob.ps[sys.elev.k] = tr.control
    u = FD.glide_state(L, 25.0, tr.gamma, tr.theta)
    du = zeros(L.n)
    function roll(delta)
        prob.ps[sys.ail.k] = delta
        prob.f(du, u, prob.p, 0.0)
        du[L.w[1]]
    end
    h = 1e-5
    Ld = (roll(h) - roll(-h)) / (2h)
    prob.ps[sys.ail.k] = 0.0
    # PHYSICS: positive aileron rolls right, and the control power is linear
    @test Ld > 0
    @test roll(0.05) ≈ 0.05 * Ld rtol = 1e-3

    # PHYSICS: steady roll rate is the aileron moment against roll damping
    ld = FD.lateral_derivatives(prob, L, tr, 25.0; control = sys.elev.k)
    p_ss = -Ld / ld.Lp
    @test 1.2 <= p_ss <= 2.0                        # rad/s per rad of aileron
    helix = p_ss * 0.15 * (B_W / 2) / 25.0
    @test 0.05 <= helix <= 0.10                     # ANCHOR: sailplane band

    # PHYSICS: an aileron step produces adverse sideslip, not proverse
    sysa = build(:TestGliderAileronStep)
    sola = sim(sysa, 3.0)
    Q = [sola(3.0, idxs = sysa.glider.fuselage.Q[i]) for i in 1:4]
    w, x, y, z = Q
    R = [1-2*(y^2+z^2) 2*(x*y+z*w) 2*(x*z-y*w);
         2*(x*y-z*w) 1-2*(x^2+z^2) 2*(y*z+x*w);
         2*(x*z+y*w) 2*(y*z-x*w) 1-2*(x^2+y^2)]
    v0 = [sola(3.0, idxs = sysa.glider.fuselage.v_0[i]) for i in 1:3]
    beta = asin(clamp((R * v0)[3] / norm(v0), -1, 1))
    bank = 2 * atan(x, w)
    @test bank > 0                                  # rolled right, as commanded
    @test beta > 0                                  # slipping right: adverse
end
