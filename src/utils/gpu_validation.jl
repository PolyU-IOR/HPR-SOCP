# This file is included by ../utils.jl.

"""
    validate_gpu_parameters!(params::HPRSOCP_parameters)

Validates GPU-related parameters and adjusts settings if GPU is requested but not available.

# Arguments
- `params::HPRSOCP_parameters`: The solver parameters to validate

# Behavior
- If `use_gpu=true` but CUDA is not functional, sets `use_gpu=false` and warns user
- If `use_gpu=true` but device_number is invalid, sets `use_gpu=false` and warns user
- Validates that device_number is within valid range [0, num_devices-1]
"""
function validate_gpu_parameters!(params::HPRSOCP_parameters)
    if params.use_gpu
        # Check if CUDA is functional
        if !CUDA.functional()
            @warn "GPU requested but CUDA is not functional. Falling back to CPU execution."
            params.use_gpu = false
            return
        end

        # Check if device_number is valid
        num_devices = length(CUDA.devices())
        if params.device_number < 0 || params.device_number >= num_devices
            @warn "Invalid GPU device number $(params.device_number). Valid range is [0, $(num_devices-1)]. Falling back to CPU execution."
            params.use_gpu = false
            return
        end

        if params.auto_save
            @warn "GPU auto_save copies iterates to CPU for HDF5 output during the solve. Disable auto_save to keep the iteration path GPU-resident until final result collection."
        end
    end
end

