# RV generation
PRUNING_RV_K = 1000 # Maximum number of rv edges for each vehicle. Set 0 to indicate no pruning.
PRUNING_RR_K = 1000 # Maximum number of rr edges for each request. Rank by detour factor.

# RTV generation
RTV_TIMELIMIT = 0 # Time limit for generating RTV graph. Set 0 to indicate no limit.
MAX_NEW = 8 # Maximum new pickup and dropoff stops

# Trip generation/feasibility check
LP_LIMITVALUE = 8 # LP_LIMITVALUE/2 is the limit of the number of new requests that is allowed to be assigned to a vehicle at one time
CTSP_OBJECTIVE = 'CTSP_VTT' # Routing objective. VTT: vehicle time travel; VMT: vehicle distance travel (TODO)
MAX_DETOUR = 600 # Maximum detour time of passenger
DWELL_ALIGHT = 0 # Dropoff dwell time
DWELL_PICKUP = 0 # Pickup dwell time
MAX_WAITING = 0 # Maximum waiting time before pickup
 # Accelration logic. 
 # FIX_ONBOARD: Do not consider previous assignement. Keep onboarding dropoff order if there are more than 4 onboarding + new passengers. Reoptimize if less than 4. 
 # FIX_PREFIX: Consider previous assignement. If new request exceeds LP_LIMITVALUE/2, return infeasible. 
 # Otherwise, follow the order of the previous assignment and reoptimize LP_LIMITVAUE stops
CTSP = "FIX_ONBOARD"