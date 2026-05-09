import HPRSOCP

params = HPRSOCP.HPRSOCP_parameters()
params.time_limit = 3600
params.stoptol = 1e-8
params.device_number = 0
params.warm_up = false
params.verbose = true
params.use_gpu = true

file = "data/model.cbf"
model = HPRSOCP.build_from_cbf(file)
result = HPRSOCP.optimize(model, params)
