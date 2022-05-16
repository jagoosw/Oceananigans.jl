#####
##### Forms for NonhydrostaticModel constructor
#####

AuxiliaryPrognosticFields(::Nothing, grid, tracer_names, bcs, closure) =
    AuxiliaryPrognosticFields(grid, tracer_names, bcs, closure)

#####
##### Closures without precomputed diffusivities
#####

AuxiliaryPrognosticFields(grid, tracer_names, bcs, closure) = nothing

#####
##### Closure tuples
#####

AuxiliaryPrognosticFields(grid, tracer_names, bcs, closure_tuple::Tuple) =
    Tuple(AuxiliaryPrognosticFields(grid, tracer_names, bcs, closure) for closure in closure_tuple)

