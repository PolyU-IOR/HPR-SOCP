import HPRSOCP

params = HPRSOCP.HPRSOCP_parameters()
params.time_limit = 3600
params.stoptol = 1e-6
params.device_number = 0
params.warm_up = false
params.verbose = true

data_path = "xxx" # Path to the directory containing the dataset files
result_path = "xxx" # Path to save the results


HPRSOCP.run_dataset(data_path, result_path, params)
println(" ✓ All problems solved successfully!")
