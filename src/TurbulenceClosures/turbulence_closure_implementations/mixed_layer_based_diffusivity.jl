
struct MixedLayerVerticalDiffusivity{} <: AbstractScalarDiffusivity{TD, VerticalFormulation}

end

function AuxiliaryPrognosticFields(grid, tracer_names, bcs, ::MixedLayerVerticalDiffusivity) 
    
    kernel_function = Field((Center, Center, Nothing), grid)
    return nothing
end