# Implementation of the `DigitalAutopilot` external component.
#
# The controller is a synchronous clocked system: every equation below that
# mentions a shift `(k-1)` or a `Sample` lives on the clock `Clock(Ts)`, and
# `Hold` is the single boundary back to continuous time. SynchToolkit resolves
# the partition; `simplify_model` picks its pass up automatically, and
# `multibody` needs it passed explicitly:
#
#     multibody(model; additional_passes = [SynchToolkit.compile_lustre])
#
# Why this is Julia rather than Dyad: Dyad has the syntax for both halves of
# this -- clock parameters (`component C@[input c]`) and shifted references
# (`x@(c - 1)`) both parse and are semantically checked -- but kernel 3.4.0
# drops the clock *binding* at a use site and ships no periodic clock source,
# so the partition reaches the compiler as an unresolved `InferredDiscrete`
# and fails. Constructing `Clock(Ts)` here is what pins the rate.

using ModelingToolkit
using ModelingToolkit: t_nounits as t, Sample, Hold, Clock, ShiftIndex

function DigitalAutopilot(; name, Ts = 0.02, theta_ground = 0.0, theta_climb = 0.12,
        V_rotate = 22.0, V_blend = 1.5, phi_cmd = 0.0, k_theta = 0.25,
        kd_theta = 0.06, k_phi = 0.35, kd_phi = 0.05, T_i = 4.0, delta_max = 0.35)

    c = Clock(Ts)
    k = ShiftIndex(c)

    ins = @variables begin
        (theta(t)::Real), [input = true, description = "Measured pitch attitude"]
        (phi(t)::Real), [input = true, description = "Measured bank angle"]
        (V(t)::Real), [input = true, description = "Measured airspeed"]
    end
    outs = @variables begin
        (elevator(t)::Real), [output = true, description = "Elevator command, held"]
        (aileron(t)::Real), [output = true, description = "Aileron command, held"]
    end
    # Everything below is on the clock.
    disc = @variables begin
        (theta_k(t)::Real), [description = "Sampled pitch attitude"]
        (phi_k(t)::Real), [description = "Sampled bank angle"]
        (V_k(t)::Real), [description = "Sampled airspeed"]
        (theta_cmd(t)::Real), [description = "Commanded pitch attitude"]
        (e_theta(t)::Real), [description = "Pitch error at the tick"]
        (e_phi(t)::Real), [description = "Bank error at the tick"]
        (dtheta(t)::Real), [description = "Pitch rate, backward difference of samples"]
        (dphi(t)::Real), [description = "Roll rate, backward difference of samples"]
        (xi_theta(t)::Real), [description = "Accumulated pitch error"]
        (xi_phi(t)::Real), [description = "Accumulated bank error"]
        (de(t)::Real), [description = "Elevator command at the tick"]
        (da(t)::Real), [description = "Aileron command at the tick"]
    end
    pars = @parameters begin
        (theta_ground::Real = theta_ground), [description = "Attitude held below V_rotate"]
        (theta_climb::Real = theta_climb), [description = "Attitude held above V_rotate"]
        (V_rotate::Real = V_rotate), [description = "Rotation airspeed"]
        (V_blend::Real = V_blend), [description = "Width of the attitude blend"]
        (phi_cmd::Real = phi_cmd), [description = "Commanded bank angle"]
        (k_theta::Real = k_theta), [description = "Elevator per radian of pitch error"]
        (kd_theta::Real = kd_theta), [description = "Elevator per rad/s of pitch rate"]
        (k_phi::Real = k_phi), [description = "Aileron per radian of bank error"]
        (kd_phi::Real = kd_phi), [description = "Aileron per rad/s of roll rate"]
        (T_i::Real = T_i), [description = "Trim-out time constant"]
        (delta_max::Real = delta_max), [description = "Control deflection limit"]
    end

    eqs = [
        # --- sample ---------------------------------------------------------
        theta_k ~ Sample(c)(theta),
        phi_k ~ Sample(c)(phi),
        V_k ~ Sample(c)(V),

        # Rotation is keyed to airspeed, not to a clock time, so the take-off
        # stays self-sequencing when the mass or the thrust changes.
        theta_cmd ~ theta_ground +
                    (theta_climb - theta_ground) * 0.5 * (1 + tanh((V_k - V_rotate) / V_blend)),
        e_theta ~ theta_k - theta_cmd,
        e_phi ~ phi_k - phi_cmd,

        # --- rates by backward difference, not by differentiation -----------
        dtheta ~ (theta_k - theta_k(k - 1)) / Ts,
        dphi ~ (phi_k - phi_k(k - 1)) / Ts,

        # --- integral, Euler, anti-windup off the *previous* output ---------
        # The factor is the local gain of the tanh, so the integrator gives up
        # exactly as fast as the control runs out of authority. It has to be
        # evaluated one tick back: a causal loop cannot use the output it is
        # still computing.
        xi_theta ~ xi_theta(k - 1) + Ts * e_theta * (1 - (de(k - 1) / delta_max)^2),
        xi_phi ~ xi_phi(k - 1) + Ts * e_phi * (1 - (da(k - 1) / delta_max)^2),

        # --- control law ----------------------------------------------------
        de ~ delta_max * tanh((k_theta * (e_theta + xi_theta / T_i) + kd_theta * dtheta) / delta_max),
        da ~ delta_max * tanh((-k_phi * (e_phi + xi_phi / T_i) - kd_phi * dphi) / delta_max),

        # --- zero-order hold back into continuous time ----------------------
        elevator ~ Hold(de),
        aileron ~ Hold(da),
    ]

    # The previous sample is seeded with the current one, so the first backward
    # difference is zero rather than a step out of nowhere.
    initialization_eqs = [
        theta_k(k - 1) ~ theta_k,
        phi_k(k - 1) ~ phi_k,
        xi_theta(k - 1) ~ 0.0,
        xi_phi(k - 1) ~ 0.0,
        de(k - 1) ~ 0.0,
        da(k - 1) ~ 0.0,
    ]
    # Value of the held outputs before the first tick.
    initial_conditions = Dict(Hold(de) => 0.0, Hold(da) => 0.0)

    System(eqs, t, [ins; outs; disc], collect(pars);
        name, initialization_eqs, initial_conditions)
end
