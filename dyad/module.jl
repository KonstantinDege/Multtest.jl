# Hand-written implementations that live alongside the Dyad sources.
# `generated/module.jl` includes this file automatically when it exists, so the
# methods attach to the stubs the compiler emits for `external` components.
#
# There are none at present. `DigitalAutopilot` used to be implemented here,
# because kernel 3.4.0 shipped no periodic clock source and an unbound clock
# parameter reached the compiler as an unresolved `InferredDiscrete`.
# `DiscreteComponents.PeriodicClock` supplies that source, so the controller is
# now written in Dyad in `avionics.dyad` and this file is empty.
