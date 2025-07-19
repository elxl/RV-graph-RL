# Global parameters
INTERVAL = 60 # Time interval for simulation

# RV generation
PRUNING_RV_K = 0 # Maximum number of rv edges for each request. Set 0 to indicate no pruning.
PRUNING_RR_K = 0 # Maximum number of rr edges for each request. Rank by detour factor. Set 0 to indicate no pruning.

# RTV generation
ALPHA = 0.5
RTV_TIMELIMIT = 0 # Time limit for generating RTV graph. Set 0 to indicate no limit.
MAX_NEW = 20 # Maximum new pickup and dropoff stops.
RANDOM = 0.3 # Random drop rate for cliques

# Trip generation/feasibility check
LP_LIMITVALUE = 8 # LP_LIMITVALUE/2 is the limit of the number of new requests that is allowed to be assigned to a vehicle at one time
CTSP_OBJECTIVE = 'CTSP_VTT' # Routing objective. VTT: vehicle time travel; DELAY: total delay VMT: vehicle distance travel (TODO)
MAX_DETOUR = 600 # Maximum detour time of passenger / 600
DWELL_ALIGHT = 0 # Dropoff dwell time
DWELL_PICKUP = 0 # Pickup dwell time
MAX_WAITING = 300 # Maximum waiting time before pickup / 300
 # Accelration logic.
 # FIX_ONBOARD: Do not consider previous assignement. Keep onboarding dropoff order if onboarding + new passengers > CARSIZE. Reoptimize if less than CARSIZE. (default)
 # FIX_PREFIX: Consider previous assignement. If new request exceeds LP_LIMITVALUE/2, return infeasible.
 # Otherwise, follow the order of the previous assignment and reoptimize LP_LIMITVAUE stops.
 # FULL: Reoptimize all stops.
CTSP = "FULL" #"FIX_ONBOARD"
CARSIZE = 10
INITIAL_TIME = "00:00:00"
FINAL_TIME = "01:00:00"
VEHICLE_LIMIT = 1000

# Feasiblity check model
MODEL_PATH = "weights/s2v/s2v_8_optimized.pt"

# Assignement problem
ALGORITHM = 'ILP_FULL'
ASSIGNMENT_OBJECTIVE = 'AO_SERVICERATE'
MISS_COST = 10000000  # Cost for unassigned requests
RMT_REWARD = 100    # Reward multiplier for RMT objective
OPTIMIZER_VERBOSE = False  # Set to True for verbose output
ILP_TIMEOUT = 0  # Time limit for ILP optimization. Set 0 to indicate no limit.

# Simulation
LAST_MINUTE_SERVICE = False # If waiting at stop as long as possible

# Parition
VERSION = 0 # [0: original, 1: with partition embedded]
PARTITION = 'None' # [None, Modularity]
PARTITION_K = 4 # Number of partitions
GAP = 0.05 # Optimality gap for assignment ILP

# File directories
RESULTS_DIRECTORY = './results'
DATAROOT = './data'
TIMEFILE = 'map/times.csv'
DISTFILE = 'map/times.csv'
EDGECOST_FILE = 'map/edges.csv'
PRED_FILE = 'map/pred.csv'
REQUEST_DATA_FILE = "requests/requests.csv"
VEHICLE_DATA_FILE = "vehicles/vehicles.csv"
LOG_FILE = "results.log"

# Mapping string values to the corresponding algorithm, objective, or ctsp values
algorithm_index = {
    "ILP_FULL": "ILP_FULL"
}

ctsp_index = {
    "FULL": "FULL",
    "FIX_ONBOARD": "FIX_ONBOARD",
    "FIX_PREFIX": "FIX_PREFIX",
    "MEGA_TSP": "MEGA_TSP"
}

ctspobjective_index = {
    "CTSP_VTT": "CTSP_VTT",
    "CTSP_TOTALDROPOFFTIME": "CTSP_TOTALDROPOFFTIME",
    "CTSP_TOTALWAITING": "CTSP_TOTALWAITING",
    "CTSP_AVGDELAY": "CTSP_AVGDELAY"
}

assignmentobjective_index = {
    "AO_SERVICERATE": "AO_SERVICERATE",
    "AO_RMT": "AO_RMT"
}
