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
using SynchToolkit

const FD = Multtest

# The DigitalAutopilot's clocked equations need SynchToolkit's synchronous
# pass. `simplify_model` adds it on its own, but `multibody` does not, and
# without it a clocked harness dies with HybridSystemNotSupportedException.
# It is inert on the continuous harnesses -- same unknowns, same equations,
# same trajectories -- so it is simplest to pass it everywhere.
build(nm::Symbol) = multibody(getfield(Multtest, nm)(; name = nm);
                              additional_passes = [SynchToolkit.compile_lustre])
sim(sys, T; tol = 1e-10, kw...) =
    solve(ODEProblem(sys, [], (0.0, T)), reltol = tol, abstol = tol; kw...)

# Attitude out of a frame's rotation matrix. Column 2 of `R` is the world
# vertical resolved in body axes, which in this frame convention is
# [sin(theta), cos(theta)*cos(phi), -cos(theta)*sin(phi)]; the heading follows
# from the nose axis. Exact for any attitude short of nose-vertical.
function attitude(sol, frame, t)
    R = [sol(t, idxs = frame.R[i, j]) for i in 1:3, j in 1:3]
    (theta = atan(R[1, 2], hypot(R[2, 2], R[3, 2])),
     phi   = atan(-R[3, 2], R[2, 2]),
     psi   = atan(-R[1, 3], R[1, 1]))
end

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

# Motor glider, mirroring dyad/motorglider.dyad. The mass is the weight the
# reference polar was computed at, not the 650 kg maximum take-off weight.
const M_MG   = 468.0
const S_MG   = 18.20
const B_MG   = 15.30
const V_S_MG = 16.75                 # stall speed, engine off [m/s]

# ---------------------------------------------------------------------------
@testset "structure: every harness reduces to a pure ODE" begin
    expected = Dict(:TestAeroSweep => 2, :TestAeroSpeedRamp => 2,
                    :TestDragTerminal => 13, :TestDragAnisotropy => 2,
                    :TestAtmosphereWind => 3, :TestGliderWingOnly => 13,
                    :TestGliderNoFin => 13, :TestGlider => 13,
                    :TestGliderAileronStep => 13,
                    :TestPropellerStatic => 2, :TestGroundContactDrop => 13,
                    # the propeller shaft speed is the fourteenth state
                    :TestMotorGlider => 14, :TestMotorGliderFeathered => 13,
                    :TestMotorGliderParked => 14,
                    # and the pilot's two integrators the fifteenth and sixteenth
                    :TestMotorGliderClimb => 16, :TestMotorGliderTakeoff => 16,
                    # the digital autopilot keeps its state on the clock, not in
                    # the continuous state vector, so these are back to fourteen
                    :TestMotorGliderClimbDigital => 14,
                    :TestMotorGliderClimbDigitalSlow => 14,
                    :TestMotorGliderTakeoffDigital => 14)
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


# ---------------------------------------------------------------------------
@testset "Propeller" begin
    sys = build(:TestPropellerStatic)
    settle(ps; T = 25.0) = solve(ODEProblem(sys, ps, (0.0, T)),
                                 reltol = 1e-10, abstol = 1e-10)
    T = 25.0

    # PHYSICS: the shaft speed is a state, so the operating point is where the
    # engine torque and the propeller torque balance -- nothing imposes it
    for V in (0.0, 25.0, 45.0)
        s = settle([sys.V_test => V])
        @test s(T, idxs = sys.prop.Q_eng) ≈ s(T, idxs = sys.prop.Q_prop) rtol = 1e-6
    end

    s0 = settle([sys.V_test => 0.0])
    # ANCHOR: static thrust and the shaft speed it comes at
    @test s0(T, idxs = sys.prop.T) ≈ 1392.0 rtol = 0.01
    @test 60 * s0(T, idxs = sys.prop.n) ≈ 2447.0 rtol = 0.01
    # PHYSICS: static thrust cannot beat momentum theory. The ideal thrust for
    # the shaft power actually delivered is (2*rho*A*P^2)^(1/3), and a real
    # fixed-pitch propeller lands at a figure of merit well under one.
    A_disc = pi * (1.5 / 2)^2
    FoM = s0(T, idxs = sys.prop.T) / cbrt(2 * RHO * A_disc * s0(T, idxs = sys.prop.P_shaft)^2)
    @test 0.45 < FoM < 0.75
    # PHYSICS: standing still it does no useful work
    @test s0(T, idxs = sys.prop.eta) == 0.0

    # PHYSICS: thrust falls with airspeed while the shaft speeds up, because
    # the propeller unloads; efficiency rises but stays under one
    sp = [settle([sys.V_test => V]) for V in (0.0, 15.0, 25.0, 35.0, 45.0)]
    @test issorted([s(T, idxs = sys.prop.T) for s in sp], rev = true)
    @test issorted([s(T, idxs = sys.prop.n) for s in sp])
    @test issorted([s(T, idxs = sys.prop.eta) for s in sp])
    @test all(s(T, idxs = sys.prop.eta) < 1 for s in sp)

    # PHYSICS: with the throttle shut the airflow drags the shaft round until
    # the propeller's own torque vanishes, which is exactly J = CP0/kP, and it
    # then produces drag rather than thrust
    sw = settle([sys.V_test => 25.0, sys.thr.k => 0.0, sys.prop.n_start => 20.0]; T = 60.0)
    @test sw(60.0, idxs = sys.prop.J_adv) ≈ 0.088 / 0.048 rtol = 1e-5
    @test sw(60.0, idxs = sys.prop.Q_prop) ≈ 0.0 atol = 1e-6
    @test sw(60.0, idxs = sys.prop.T) < 0

    # PHYSICS: stopped is the other engine-off equilibrium, and it is stable
    # enough to stay there -- a real stopped propeller does not restart itself
    ss = settle([sys.V_test => 25.0, sys.thr.k => 0.0, sys.prop.n_start => 0.0]; T = 20.0)
    @test ss(20.0, idxs = sys.prop.n) ≈ 0.0 atol = 1e-9
    @test ss(20.0, idxs = sys.prop.T) ≈ 0.0 atol = 1e-9
end

# ---------------------------------------------------------------------------
@testset "GroundContact" begin
    sys = build(:TestGroundContactDrop)
    sol = sim(sys, 12.0)
    m, k = 400.0, 1.0e5

    # PHYSICS: released 100 mm clear, the tanh gate leaves nothing behind but
    # the f_eps floor of the zero clamp -- no phantom lift in flight
    @test sol(0.0, idxs = sys.contact.pen) ≈ 0.0 atol = 1e-12
    @test sol(0.0, idxs = sys.contact.Fn) ≈ 0.025 atol = 1e-6

    # PHYSICS: it settles at the static deflection m*g/k and stops there,
    # rather than bouncing forever or sticking
    @test sol(12.0, idxs = sys.contact.pen) ≈ m * G / k rtol = 2e-3
    @test sol(12.0, idxs = sys.contact.Fn) ≈ m * G rtol = 2e-3
    @test abs(sol(12.0, idxs = sys.body.v_0[2])) < 1e-3

    # PHYSICS: the ground pushes and never pulls, at every instant
    @test all(sol(t, idxs = sys.contact.Fn) > 0 for t in 0:0.01:12)
    # PHYSICS: a vertical drop onto a flat plane does not wander off it
    @test abs(sol(12.0, idxs = sys.body.r_0[1])) < 1e-9
    @test abs(sol(12.0, idxs = sys.body.r_0[3])) < 1e-9
end

# ---------------------------------------------------------------------------
@testset "motor glider engine-off polar" begin
    sys = build(:TestMotorGliderFeathered)
    prob = ODEProblem(sys, [], (0.0, 1.0))
    L = FD.StateLayout(sys, sys.glider.fuselage)
    seed = [-0.045, 0.03, 0.0]

    # ANCHOR: the three points of the reference SF 25 C engine-off polar, which
    # is the part of that data that is self-consistent. The drag coefficients
    # are quoted to two figures, hence the looser tolerance on them.
    for (V, CLd, CDd, LDd) in ((20.833, 0.95, 0.043, 22.1),
                               (25.000, 0.66, 0.028, 23.6),
                               (33.333, 0.37, 0.021, 17.6))
        tr = FD.trim_glide(prob, L, V; control = sys.elev.k, x0 = seed)
        @test tr.converged
        CL = M_MG * G * cos(tr.gamma) / (0.5 * RHO * V^2 * S_MG)
        @test CL ≈ CLd rtol = 0.01
        @test CL / tr.LD ≈ CDd rtol = 0.04
        @test tr.LD ≈ LDd rtol = 0.04
    end

    pol = FD.glide_polar(prob, L, 16.0:0.25:45.0; control = sys.elev.k, x0 = seed)
    ok = [p for p in pol if p.trim.converged]
    @test length(ok) > 100
    # ANCHOR: the quoted 60 km/h stall is reproduced, which is the check that
    # CL_max and the polar weight are consistent with each other
    @test 3.6 * minimum(p.V for p in ok) ≈ 60.0 rtol = 0.02
    @test maximum(p.trim.LD for p in ok) ≈ 22.93 rtol = 0.02
    @test minimum(p.trim.sink for p in ok) ≈ 0.912 rtol = 0.03
end

# ---------------------------------------------------------------------------
@testset "motor glider: windmilling propeller costs glide" begin
    # PHYSICS: a windmilling propeller is drag, so the engine-off glide with
    # the propeller turning must be clearly worse than with none fitted. The
    # shaft speed has to be trimmed alongside the flight variables or the
    # solver returns a stopped propeller and the penalty vanishes.
    sysf = build(:TestMotorGliderFeathered)
    probf = ODEProblem(sysf, [], (0.0, 1.0))
    Lf = FD.StateLayout(sysf, sysf.glider.fuselage)
    trf = FD.trim_glide(probf, Lf, 25.0; control = sysf.elev.k, x0 = [-0.045, 0.03, 0.0])

    sysw = build(:TestMotorGlider)
    probw = ODEProblem(sysw, [], (0.0, 1.0))
    Lw = FD.StateLayout(sysw, sysw.glider.fuselage)
    nidx = FD.state_index(sysw, sysw.glider.propeller.n)
    trw = FD.trim_glide(probw, Lw, 25.0; control = sysw.elev.k,
                        x0 = [-0.05, 0.03, 0.0], aux = [nidx], aux0 = [10.0])
    @test trf.converged && trw.converged
    @test trw.aux[1] > 1.0                       # genuinely windmilling, not stopped
    @test trw.LD < trf.LD
    @test 0.05 < 1 - trw.LD / trf.LD < 0.30      # ANCHOR: a 5-30% penalty
end

# ---------------------------------------------------------------------------
@testset "motor glider on the ground" begin
    sys = build(:TestMotorGliderParked)
    sol = sim(sys, 30.0; tol = 1e-9)
    f = sys.glider.fuselage
    att = attitude(sol, f.frame_a, 30.0)
    Fl = sol(30.0, idxs = sys.glider.mainwheel_l.Fn)
    Fr = sol(30.0, idxs = sys.glider.mainwheel_r.Fn)
    Ft = sol(30.0, idxs = sys.glider.tailwheel.Fn)
    Sl = sol(30.0, idxs = sys.glider.tipskid_l.Fn)
    Sr = sol(30.0, idxs = sys.glider.tipskid_r.Fn)

    # PHYSICS: the gear carries the weight, all of it and nothing more
    @test Fl + Fr + Ft + Sl + Sr ≈ M_MG * G rtol = 1e-4
    # PHYSICS: a pair of main wheels on a track makes upright a *stable*
    # equilibrium, so it sits dead level and the tip skids never load. A single
    # centreline wheel would fall onto a tip from rounding error alone.
    @test Fl ≈ Fr rtol = 1e-8
    @test abs(att.phi) < 1e-9
    @test Sl < 0.03 && Sr < 0.03                 # the f_eps floor and no more
    # PHYSICS: a taildragger settles nose up on all three wheels
    @test Ft > 0.02 * M_MG * G
    @test rad2deg(att.theta) ≈ 6.49 rtol = 0.02
    @test sol(30.0, idxs = f.r_0[2]) ≈ 0.7805 rtol = 1e-3
    # PHYSICS: with the throttle shut the propeller stays stopped
    @test sol(30.0, idxs = sys.glider.propeller.n) ≈ 0.0 atol = 1e-9
end

# ---------------------------------------------------------------------------
@testset "Pilot holds the commanded attitude" begin
    sys = build(:TestMotorGliderClimb)
    sol = solve(ODEProblem(sys, [], (0.0, 150.0)), reltol = 1e-9, abstol = 1e-9)
    @test sol.retcode == ReturnCode.Success
    f = sys.glider.fuselage
    att = attitude(sol, f.frame_a, 150.0)
    v = [sol(150.0, idxs = f.v_0[i]) for i in 1:3]

    # PHYSICS: the integral terms mean the commanded attitude is the attitude
    # that results, not merely the one a proportional loop droops away from
    @test att.theta ≈ 0.12 rtol = 2e-3
    @test abs(att.phi) < 1e-4
    # PHYSICS: a held attitude at a fixed throttle is an equilibrium, so the
    # climb settles rather than phugoiding -- speed and rate both constant
    for t in (120.0, 135.0, 150.0)
        @test norm([sol(t, idxs = f.v_0[i]) for i in 1:3]) ≈ norm(v) rtol = 5e-3
        @test sol(t, idxs = f.v_0[2]) ≈ v[2] rtol = 5e-3
    end
    # ANCHOR: full throttle at this attitude lands on the best rate of climb
    @test norm(v) ≈ 35.4 rtol = 0.02
    @test v[2] ≈ 5.75 rtol = 0.03
    # PHYSICS: the aileron holds a standing deflection against the propeller
    # torque reaction, and it has to be a right-roll input to oppose a
    # clockwise propeller
    @test sol(150.0, idxs = sys.pilot.aileron) > 0
    # PHYSICS: wings level is wings level, so the aircraft does not turn.
    # A bank of phi turns at g*tan(phi)/V; with phi driven to zero the only
    # heading drift left is the aileron's adverse yaw.
    dpsi = (attitude(sol, f.frame_a, 150.0).psi - attitude(sol, f.frame_a, 100.0).psi) / 50
    @test abs(dpsi) < 1e-3                       # rad/s
end

# ---------------------------------------------------------------------------
@testset "motor glider take-off" begin
    sys = build(:TestMotorGliderTakeoff)
    sol = solve(ODEProblem(sys, [], (0.0, 60.0)), reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    f = sys.glider.fuselage
    W = M_MG * G
    ts = range(0, 60, length = 6001)
    Fn(t) = sol(t, idxs = sys.glider.mainwheel_l.Fn) +
            sol(t, idxs = sys.glider.mainwheel_r.Fn) +
            sol(t, idxs = sys.glider.tailwheel.Fn)
    V(t) = norm([sol(t, idxs = f.v_0[i]) for i in 1:3])

    t_tail = ts[findfirst(t -> sol(t, idxs = sys.glider.tailwheel.Fn) < 0.01W, ts)]
    t_lift = ts[findfirst(t -> Fn(t) < 0.01W, ts)]

    # PHYSICS: a taildragger raises the tail well before it flies, and it does
    # so because elevator authority builds with airspeed
    @test t_tail < t_lift
    @test V(t_tail) < 0.5 * V(t_lift)
    # ANCHOR: ground roll and lift-off speed
    @test t_lift ≈ 12.0 rtol = 0.05
    @test sol(t_lift, idxs = f.r_0[1]) ≈ 165.0 rtol = 0.05
    @test V(t_lift) ≈ 27.0 rtol = 0.03
    # PHYSICS: it flies off above the stall, with the margin a rotation at
    # 22 m/s plus a second or two of pitch-up implies
    @test 1.4 < V(t_lift) / V_S_MG < 1.8

    # PHYSICS: the contacts release cleanly and stay released -- nothing left
    # behind but the f_eps floor of the three contacts
    @test all(Fn(t) < 0.1 for t in ts[ts .> t_lift + 1])
    # PHYSICS: the wing tips never touch, and the propeller never does either
    @test maximum(sol(t, idxs = sys.glider.tipskid_l.Fn) for t in ts) < 0.03
    @test minimum(sol(t, idxs = sys.glider.propeller.frame_a.r_0[2]) - 0.75
                  for t in ts) > 0.05

    # PHYSICS: the pilot keeps it straight. With the propeller torque rolling
    # it left and the spiral mode divergent, a fixed stick departs into a
    # climbing left turn; the wings-level loop is what prevents that.
    @test maximum(abs(attitude(sol, f.frame_a, t).phi) for t in ts) < deg2rad(1.0)
    @test abs(attitude(sol, f.frame_a, 60.0).psi) < deg2rad(3.0)
    # PHYSICS: controls stay well inside their travel throughout
    @test maximum(abs(sol(t, idxs = sys.pilot.elevator)) for t in ts) < deg2rad(10.0)

    # PHYSICS: the climb-out converges on the same attitude-held equilibrium
    # the free-flight harness finds, which is the check that the ground run
    # leaves the aircraft in a state the air alone determines from there
    @test attitude(sol, f.frame_a, 60.0).theta ≈ 0.12 rtol = 0.05
    @test sol(60.0, idxs = f.v_0[2]) > 5.0
end

# ---------------------------------------------------------------------------
@testset "AttitudeSensor" begin
    # PHYSICS: the sensor must report exactly the attitude and airspeed that
    # the frame and the air bus carry, so it is checked against the same
    # quantities computed independently from the rotation matrix.
    sys = build(:TestMotorGliderClimbDigital)
    sol = sim(sys, 20.0; tol = 1e-9)
    f = sys.glider.fuselage
    for t in (0.0, 3.0, 9.0, 20.0)
        a = attitude(sol, f.frame_a, t)
        @test sol(t, idxs = sys.sensor.theta) ≈ a.theta atol = 1e-12
        @test sol(t, idxs = sys.sensor.phi) ≈ a.phi atol = 1e-12
        v = [sol(t, idxs = f.v_0[i]) for i in 1:3]
        @test sol(t, idxs = sys.sensor.V) ≈ sqrt(v[1]^2 + v[2]^2 + v[3]^2 + 0.5^2) rtol = 1e-10
    end
    # PHYSICS: a sensor applies nothing to the frame it reads
    @test all(abs(sol(5.0, idxs = sys.sensor.frame_a.f[i])) < 1e-12 for i in 1:3)
    @test all(abs(sol(5.0, idxs = sys.sensor.frame_a.tau[i])) < 1e-12 for i in 1:3)
end

# ---------------------------------------------------------------------------
@testset "DigitalAutopilot: sampled and held" begin
    sys = build(:TestMotorGliderClimbDigitalSlow)      # 1 Hz, so ticks are visible
    sol = sim(sys, 12.0; tol = 1e-9)
    Ts = 1.0

    # PHYSICS: the outputs are held between ticks, so they are piecewise
    # constant -- sampling the same interval twice must give the same number
    for t0 in (2.0, 5.0, 8.0)
        a = sol(t0 + 0.2, idxs = sys.ap.elevator)
        b = sol(t0 + 0.7, idxs = sys.ap.elevator)
        @test a ≈ b rtol = 1e-9
    end
    # PHYSICS: and it does change across a tick boundary
    @test sol(4.7, idxs = sys.ap.elevator) != sol(5.3, idxs = sys.ap.elevator)

    # PHYSICS: the held value is the control law evaluated at the tick
    @test sol(5.4, idxs = sys.ap.elevator) ≈ sol(5.0, idxs = sys.ap.de) rtol = 1e-7
    @test sol(5.4, idxs = sys.ap.aileron) ≈ sol(5.0, idxs = sys.ap.da) rtol = 1e-7

    # PHYSICS: the rate term is a backward difference of the samples, not a
    # derivative -- this is the defining difference from the continuous Pilot
    @test sol(5.0, idxs = sys.ap.dtheta) ≈
          (sol(5.0, idxs = sys.ap.theta_k) - sol(4.0, idxs = sys.ap.theta_k)) / Ts rtol = 1e-6

    # PHYSICS: nothing in the controller enters the continuous state vector
    @test !any(occursin("ap", string(u)) for u in unknowns(sys))
end

# ---------------------------------------------------------------------------
@testset "digital autopilot reproduces the continuous pilot" begin
    # The two harnesses are identical but for the controller, so any
    # difference between them is the sampling and nothing else.
    function climb(nm)
        sys = build(nm)
        sol = solve(ODEProblem(sys, [], (0.0, 200.0)), reltol = 1e-9, abstol = 1e-9)
        f = sys.glider.fuselage
        th = [rad2deg(attitude(sol, f.frame_a, t).theta) for t in 0:0.02:60]
        v = [sol(200.0, idxs = f.v_0[i]) for i in 1:3]
        (retcode = sol.retcode, peak = maximum(th),
         theta = rad2deg(attitude(sol, f.frame_a, 200.0).theta),
         V = norm(v), RoC = v[2], phi = attitude(sol, f.frame_a, 200.0).phi)
    end
    cont = climb(:TestMotorGliderClimb)
    fast = climb(:TestMotorGliderClimbDigital)
    slow = climb(:TestMotorGliderClimbDigitalSlow)
    for r in (cont, fast, slow)
        @test r.retcode == ReturnCode.Success
        # PHYSICS: the integral term does not care how slowly it is ticked, so
        # every version ends in the same commanded attitude and the same climb
        @test r.theta ≈ rad2deg(0.12) rtol = 1e-3
        @test abs(r.phi) < 1e-4
        @test r.V ≈ 35.385 rtol = 2e-3
        @test r.RoC ≈ 5.746 rtol = 3e-3
    end
    # ANCHOR: at 50 Hz against a ~1.3 rad/s loop the discretisation is
    # invisible -- the transient matches the continuous one to a millidegree
    @test fast.peak ≈ cont.peak atol = 0.01
    # PHYSICS: slowing the clock costs phase, so the overshoot grows. The
    # zero-order hold is worth about Ts/2 of lag and the rate term becomes a
    # difference over a whole second.
    @test slow.peak > cont.peak + 0.05
end

# ---------------------------------------------------------------------------
@testset "digital take-off" begin
    sys = build(:TestMotorGliderTakeoffDigital)
    sol = solve(ODEProblem(sys, [], (0.0, 60.0)), reltol = 1e-8, abstol = 1e-8)
    @test sol.retcode == ReturnCode.Success
    f = sys.glider.fuselage
    W = M_MG * G
    ts = range(0, 60, length = 6001)
    Fn(t) = sol(t, idxs = sys.glider.mainwheel_l.Fn) +
            sol(t, idxs = sys.glider.mainwheel_r.Fn) +
            sol(t, idxs = sys.glider.tailwheel.Fn)
    V(t) = norm([sol(t, idxs = f.v_0[i]) for i in 1:3])
    t_lift = ts[findfirst(t -> Fn(t) < 0.01W, ts)]

    # ANCHOR: the same take-off the continuous pilot flies, to within the
    # resolution of a 50 Hz loop
    @test t_lift ≈ 12.0 rtol = 0.05
    @test sol(t_lift, idxs = f.r_0[1]) ≈ 165.0 rtol = 0.05
    @test V(t_lift) ≈ 27.0 rtol = 0.03
    # PHYSICS: still flown straight, and still inside the control travel
    @test maximum(abs(attitude(sol, f.frame_a, t).phi) for t in ts) < deg2rad(1.0)
    @test maximum(abs(sol(t, idxs = sys.ap.elevator)) for t in ts) < deg2rad(10.0)
    @test attitude(sol, f.frame_a, 60.0).theta ≈ 0.12 rtol = 0.05
end
