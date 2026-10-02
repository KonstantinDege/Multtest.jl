module Multtest

# SynchToolkit supplies the synchronous-clock compiler pass that the
# `DigitalAutopilot`'s clocked equations need. Loading it here registers the
# DyadInterface extension, so `simplify_model` picks the pass up on its own;
# `multibody` has to be told explicitly:
#
#     multibody(model; additional_passes = [SynchToolkit.compile_lustre])
using SynchToolkit

include("flightdynamics.jl")

include("../generated/module.jl")
    
end # module Multtest
