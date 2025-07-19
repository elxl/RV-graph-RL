import os, time, pickle, datetime
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from src.utils.parser import initialize
import src.utils.global_var as glo
from src.env.struct.Network import Network
from src.utils.helper import load_vehicles,load_requests,encode_time,decode_time,get_active_vehicles,get_new_requests
from src.algo.rr_partition_online_singlethread import rr_partition, tripgenerator_parallel, vehicle_assignment

if __name__ == "__main__":
    args = initialize()

    # Set up routing matrix
    print(f"{Fore.WHITE}Setting up network{Style.RESET_ALL}")
    config = {
        'DATAROOT':glo.DATAROOT,
        'TIMEFILE':glo.TIMEFILE,
        'DISTFILE':glo.DISTFILE,
        'EDGECOST_FILE':glo.EDGECOST_FILE,
        'PRED_FILE':glo.PRED_FILE,
        'DWELL_PICKUP':glo.DWELL_PICKUP,
        'DWELL_ALIGHT':glo.DWELL_ALIGHT
    }
    network = Network(config=config)
    print(f"{Fore.GREEN}Network was loaded!{Style.RESET_ALL}")

    # Load requests and vehicles
    print(f"{Fore.WHITE}Loading requests and vehicles{Style.RESET_ALL}")
    vehicles = load_vehicles(os.path.join(glo.DATAROOT, glo. VEHICLE_DATA_FILE))
    requests = load_requests(os.path.join(glo.DATAROOT, glo.REQUEST_DATA_FILE),network)

    # Load only partial data as the pre-booking data TODO: replaced by sample trips from the whole dataset
    initial_time = decode_time(glo.INITIAL_TIME)
    current_time = initial_time
    requests = get_new_requests(requests,current_time)
    print(f"{Fore.GREEN}{len(requests)} requests and {len(vehicles)} vehicles were loaded!{Style.RESET_ALL}")
    print(f"{Fore.RED}Computation starts with {args.THREADS} threads...{Style.RESET_ALL}")

    start_time = time.perf_counter()

    # Build graph and partition
    print(f"{Fore.YELLOW}********Partition start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    rr_graph_list, sizes = rr_partition(requests, current_time, network, mode=args.PARTITION, threads=args.THREADS)
    if args.PARTITION != 'None':
        print(f"{Fore.WHITE}Partitioned into {len(rr_graph_list)} subgraphs. Min size {min(sizes)}. Max size {max(sizes)}. All {sum(sizes)}. Variance {np.var(sizes)}. {Style.RESET_ALL}")
        print(f"{Fore.GREEN}Partition finished at {datetime.datetime.now().time()}!{Style.RESET_ALL}")
    else:
        print(f"{Fore.WHITE}No partition. Graph size {rr_graph_list[0].number_of_nodes()}.{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Building RR graph finished at {datetime.datetime.now().time()}!{Style.RESET_ALL}")

    # Get trips
    print(f"{Fore.YELLOW}********Trip generation start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    trips = tripgenerator_parallel(rr_graph_list, vehicles, network, current_time, threads=args.THREADS)
    trips = {vehicle: list(set(trips)) for vehicle, trips in trips.items()}
    total_trips = sum(len(trips) for trips in trips.values())
    print(f"{Fore.WHITE}{total_trips} trips generated.{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Trip generation finished at {datetime.datetime.now().time()}!{Style.RESET_ALL}")

    # Assign vehicles to trips
    print(f"{Fore.YELLOW}********Assignment start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    assignments, served, obj, icount = vehicle_assignment(trips, requests)

    end_time = time.perf_counter()

    print(f"{Fore.GREEN}Optimization finished at {datetime.datetime.now().time()}. {served} out of {len(requests)} requests were served!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Calculation finished with {(end_time - start_time):.2f}s.{Style.RESET_ALL}")

    # Save results
    results_file = os.path.join(glo.RESULTS_DIRECTORY, glo.LOG_FILE)
    # results = {
    #     'time': round(end_time - start_time),
    #     'requests': len(requests),
    #     'served': served,
    #     'assignments': assignments,
    #     'obj': obj,
    # }
    # pickle.dump(results, open(results_file, 'wb'))
    with open(results_file, "w") as results:
        # Write basic configuration details
        results.write(f"MODE ONLINE")
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
        results.write(f"INTERVAL {glo.INTERVAL}\n")
        results.write(f"PARTITION {args.PARTITION}\n")
        if args.PARTITION == 'METIS':
            results.write(f"PARTITION_K {args.PARTITION_K}\n")
        if args.PARTITION != 'None':
            results.write(f"{Fore.WHITE}Partitioned into {len(rr_graph_list)} subgraphs. Min size {min(sizes)}. Max size {max(sizes)}. Variance {np.var(sizes)}. {Style.RESET_ALL} \n")

        # Write results
        results.write(f"***********RESULT**********\n")
        results.write(f"TIME {round(end_time - start_time)}\n")
        results.write(f"REQUESTS {len(requests)}\n")
        results.write(f"SERVED {served}\n")
        results.write(f"ASSIGNMENTS {icount}\n")
        results.write(f"OBJ {obj}\n")