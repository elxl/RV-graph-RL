import os, datetime
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from src.algo.assignment import ilp_assignement_full
from src.env.simulator.simulate import simulate_vehicles
from src.utils.parser import initialize
import src.utils.global_var as glo
from src.env.struct.Network import Network
from src.utils.helper import load_vehicles,load_requests,encode_time,get_active_vehicles,get_new_requests

colorama_init()

print(f"{Fore.WHITE}Starting Ridepool Simulator!!!{Style.RESET_ALL}!")
args = initialize()

# Head output with a description of the run
results_file = os.path.join(glo.RESULTS_DIRECTORY, "results.log")
with open(results_file, "w") as results:
    # Write basic configuration details
    results.write(f"DATAROOT {glo.DATAROOT}\n")
    results.write(f"RESULTS_DIRECTORY {glo.RESULTS_DIRECTORY}\n")
    results.write(f"TIMEFILE {glo.TIMEFILE}\n")
    results.write(f"EDGECOST_FILE {glo.EDGECOST_FILE}\n")
    results.write(f"VEHICLE_LIMIT {glo.VEHICLE_LIMIT}\n")
    results.write(f"MAX_WAITING {glo.MAX_WAITING}\n")
    results.write(f"MAX_DETOUR {glo.MAX_DETOUR}\n")
    results.write(f"REQUEST_DATA_FILE {glo.REQUEST_DATA_FILE}\n")
    results.write(f"VEHICLE_DATA_FILE {glo.VEHICLE_DATA_FILE}\n")
    results.write(f"CARSIZE {glo.CARSIZE}\n")
    results.write(f"INITIAL_TIME {glo.INITIAL_TIME}\n")
    results.write(f"FINAL_TIME {glo.FINAL_TIME}\n")

    # Write ALGORITHM information
    results.write("ALGORITHM ")
    if glo.ALGORITHM == "ILP_FULL":
        results.write("ILP_FULL\n")
    else:
        results.write("UNLABELED\n")

    # Write CTSP information
    results.write("CTSP ")
    if glo.CTSP == "FULL":
        results.write("FULL\n")
    elif glo.CTSP == "FIX_ONBOARD":
        results.write("FIX_ONBOARD\n")
    elif glo.CTSP == "FIX_PREFIX":
        results.write("FIX_PREFIX\n")
    else:
        results.write("UNLABELED\n")

    # Write CTSP_OBJECTIVE information
    results.write("CTSP_OBJECTIVE ")
    if glo.CTSP_OBJECTIVE == "CTSP_AVGDELAY":
        results.write("CTSP_AVGDELAY\n")
    else:
        results.write("NOT-AVGDELAY (other)\n")

    # Write LAST_MINUTE_SERVICE status
    if glo.LAST_MINUTE_SERVICE:
        results.write("LAST_MINUTE_SERVICE Active\n")

# Set up routing matrix
print(f"{Fore.WHITE}Setting up network{Style.RESET_ALL}")
config = {
    'DATAROOT':glo.DATAROOT,
    'TIMEFILE':glo.TIMEFILE,
    'DISTFILE':glo.DISTFILE,
    'EDGECOST_FILE':glo.EDGECOST_FILE,
    'DWELL_PICKUP':glo.DWELL_PICKUP,
    'DWELL_ALIGHT':glo.DWELL_ALIGHT
}
network = Network(config=config)
print(f"{Fore.GREEN}Network was loaded!{Style.RESET_ALL}")

# Load vehicles and requests
print(f"{Fore.WHITE}Loading vehicles and requests{Style.RESET_ALL}")
vehicles = load_vehicles(os.path.join(glo.DATAROOT, glo. VEHICLE_DATA_FILE))
requests = load_requests(os.path.join(glo.DATAROOT, glo.REQUEST_DATA_FILE),network)
active_requests = []
print(f"{Fore.GREEN}Vehicles and requests were loaded!{Style.RESET_ALL}")

# Statistics variables
service_count = 0
stats_dropoff_count = 0
stats_entry_count = 0
stats_total_waiting_time = 0
stats_pickup_count = 0
stats_total_in_vehicle_time = 0
stats_total_delay = 0
stats_shared_count = 0
storage_service_count = 0
storage_request_count = 0

print(f"{Fore.GREEN}Done with all set up!{Style.RESET_ALL}")
print(f"{Fore.CYAN}Starting iterations!{Style.RESET_ALL}")
current_time = glo.INITIAL_TIME - glo.INTERVAL
while current_time < glo.FINAL_TIME - glo.INTERVAL:

    current_time += glo.INTERVAL
    print(f"{Fore.WHITE}Updated simulation clock to {encode_time(current_time)} \
           \t System time {datetime.datetime.now().time()}{Style.RESET_ALL}")

    #############################################################
    # Get active vehicles and new requests for current iteration#
    #############################################################
    print(f"{Fore.YELLOW}Update request buffer...{Style.RESET_ALL}")
    active_vehicles = get_active_vehicles(vehicles,current_time)
    new_requests = get_new_requests(requests,current_time)
    stats_entry_count += len(new_requests)

    active_requests.extend(new_requests)

    print(f"{Fore.GREEN}Buffer update complete!{Style.RESET_ALL}")

    #########################################
    ########### Run trip assignement ########
    #########################################
    print(f"{Fore.YELLOW}Starting trip assignement...{Style.RESET_ALL}")
    assigned_trips = ilp_assignement_full(active_vehicles,active_requests,current_time,network,args.threads)

    # Remove blank trips
    blank_trips = {
        v for v in vehicles
        if v in assigned_trips and not v.passengers and not assigned_trips[v].requests
    }

    # Remove vehicles with blank trips from the assigned trips
    for v in blank_trips:
        del assigned_trips[v]

    # Collect all assigned requests from remaining trips
    assigned_requests = [
        r for trips in assigned_trips.values() for r in trips.requests
    ]

    print(f"{Fore.GREEN}{len(assigned_trips)} assignements have been made!{Style.RESET_ALL}")

    ###########################################
    ############## Move vehicles ##############
    ###########################################
    print(f"{Fore.YELLOW}Vehicle moving to assigned passengers...{Style.RESET_ALL}")
    simulate_vehicles(vehicles,assigned_trips,network,current_time,args.threads)
    print(f"{Fore.GREEN}Vehicle movement completed!{Style.RESET_ALL}")

    ##########################################################
    ############ Update statistics and log information #######
    ##########################################################
    for vehicle in vehicles:
        # Process boarded requests
        for r in vehicle.just_boarded:
            stats_total_waiting_time += r.boarding_time - r.entry_time
            stats_pickup_count += 1
            service_count += 1

        # Process alighted requests
        for r in vehicle.just_alighted:
            stats_dropoff_count += 1
            stats_total_in_vehicle_time += r.alighting_time - r.boarding_time
            stats_total_delay += r.alighting_time - r.boarding_time - r.ideal_traveltime
            stats_shared_count += int(r.shared)

    with open(results_file, "a") as f:
        f.write(f"Time stamp: {encode_time(current_time)} \t System time {datetime.datetime.now().time()}\n")

    # Service Rate
    if stats_entry_count > 0:
        service_rate = 100 * stats_pickup_count / stats_entry_count
    else:
        service_rate = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tService Rate\t{service_rate:.2f}\t%\n")
    print(f"{Fore.RED}Service rate is {service_rate:.2f}%.{Style.RESET_ALL}")

    # Average waiting time
    if stats_pickup_count > 0:
        average_waiting_time = stats_total_waiting_time / stats_pickup_count
    else:
        average_waiting_time = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tAvg Waiting\t{average_waiting_time:.2f}\n")

    # Average riding time
    if stats_dropoff_count > 0:
        average_riding_time = stats_total_in_vehicle_time / stats_dropoff_count
    else:
        average_riding_time = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tAvg Riding\t{average_riding_time:.2f}\n")

    # Average total delay
    if stats_dropoff_count > 0:
        average_total_delay = stats_total_delay / stats_dropoff_count
    else:
        average_total_delay = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tAvg Delay\t{average_total_delay:.2f}\n")

    # Mean passengers
    if current_time != glo.INITIAL_TIME and active_vehicles:
        mean_passengers = stats_total_in_vehicle_time / ((current_time - glo.INITIAL_TIME) * len(active_vehicles))
    else:
        mean_passengers = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tMean Passengers\t{mean_passengers:.2f}\n")

    # Shared rate
    if stats_dropoff_count > 0:
        shared_rate = 100 * stats_shared_count / stats_dropoff_count
    else:
        shared_rate = 0.0
    with open(results_file, "a") as f:
        f.write(f"\tShared rate\t{shared_rate:.2f}\t%\n")

    # Total shared
    with open(results_file, "a") as f:
        f.write(f"\tTotal shared\t{stats_shared_count}\n")    
    
    #########################################################
    ############# Update active requests buffer #############
    #########################################################
    print(f"{Fore.YELLOW}Updating the active requests list...{Style.RESET_ALL}")

    # Clear the active requests list
    active_num = len(active_requests)
    active_requests.clear()

    # Identify boarded requests
    boarded_requests = {r.id for v in vehicles for r in v.just_boarded}

    # Filter active requests
    active_requests.extend(
        r for r in assigned_requests if r.id not in boarded_requests and current_time < r.latest_boarding
    )

    # Collect final assigned requests
    final_assigned_requests = {
        r for trips in assigned_trips.values() for r in trips.requests
    }

    # Mark all final assigned requests as assigned (for statistics collection)
    for r in final_assigned_requests:
        r.assigned = True

    print(f"{Fore.GREEN}Current request buffer is updated!")
    print(f"Number of assigned passengers:{len(final_assigned_requests)}/{active_num}")
    print(f"Done with iteration{Style.RESET_ALL}")

#########################################################
############# Final statistics and summary #############
#########################################################
f = open(results_file, "a")
f.write("FINAL SUMMARY\n")

# Final statistics
final_count = stats_pickup_count
errors = 0

for r in active_requests:
    if r.assigned and r.boarding_time == 0:
        if r.entry_time + glo.MAX_WAITING < current_time:
            errors += 1
        else:
            final_count += 1

service_rate = 100 * final_count / stats_entry_count if stats_entry_count > 0 else 0.0
f.write(f"\tService Rate\t{service_rate:.2f}\t%\n")
f.write(f"\tServed\t{final_count}\n")
f.write(f"\tError Count\t{errors}\n")

# Calculate passenger time
passenger_time = stats_total_in_vehicle_time
for v in vehicles:
    for r in v.passengers:
        if r.alighting_time == 0:
            duration = current_time - r.boarding_time
            passenger_time += duration

mean_passengers = (passenger_time /
                    ((current_time - glo.INITIAL_TIME) * len(vehicles))
                    if (current_time != glo.INITIAL_TIME) and vehicles else 0.0)
f.write(f"\tMean Passengers\t{mean_passengers:.2f}\n")

# Vehicle state statistics
total_idle = sum(v.get_total_idle(current_time) for v in vehicles)
total_enroute = sum(v.get_total_enroute(current_time) for v in vehicles)
total_inuse = sum(v.get_total_inuse(current_time) for v in vehicles)

f.write(f"\tTotal Idle\t{total_idle}\n")
f.write(f"\tTotal En Route\t{total_enroute}\n")
f.write(f"\tTotal Inuse\t{total_inuse}\n")
f.close()