import os, time, pickle, datetime
import numpy as np
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from src.utils.parser import initialize
import src.utils.global_var as glo
from src.env.struct.Network import Network
from src.utils.helper import load_vehicles,load_requests,encode_time,decode_time,get_active_vehicles,get_new_requests
from src.algo.rr_partition import rr_partition, tripgenerator_parallel, vehicle_assignment

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

    # Load requests
    num_vehicles = glo.VEHICLE_LIMIT
    print(f"{Fore.WHITE}Loading requests{Style.RESET_ALL}")
    requests = load_requests(os.path.join(glo.DATAROOT, glo.REQUEST_DATA_FILE),network)

    # Load only partial data as the pre-booking data TODO: replaced by sample trips from the whole dataset
    initial_time = decode_time(glo.INITIAL_TIME)
    current_time = initial_time
    requests = get_new_requests(requests,current_time)
    print(f"{Fore.GREEN}{len(requests)} requests were loaded!{Style.RESET_ALL}")
    print(f"{Fore.RED}Computation starts with {args.THREADS} threads...{Style.RESET_ALL}")

    start_time = time.perf_counter()

    # Build graph and partition
    print(f"{Fore.YELLOW}********Partition start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    rr_graph_list, sizes = rr_partition(requests, current_time, network, mode=args.PARTITION, threads=args.THREADS)
    print(f"{Fore.WHITE}Partitioned into {len(rr_graph_list)} subgraphs. Min size {min(sizes)}. Max size {max(sizes)}. Variance {np.var(sizes)}. {Style.RESET_ALL}")
    print(f"{Fore.GREEN}Partition finished at {datetime.datetime.now().time()}!{Style.RESET_ALL}")

    # Get trips
    print(f"{Fore.YELLOW}********Trip generation start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    trips = tripgenerator_parallel(rr_graph_list, network, current_time, threads=args.THREADS)
    print(f"{Fore.WHITE}{len(trips)} trips generated.{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Trip generation finished at {datetime.datetime.now().time()}!{Style.RESET_ALL}")

    # Assign vehicles to trips
    print(f"{Fore.YELLOW}********Assignment start at {datetime.datetime.now().time()}********{Style.RESET_ALL}")
    assignments, served = vehicle_assignment(trips, num_vehicles)

    end_time = time.perf_counter()

    print(f"{Fore.GREEN}Optimization finished at {datetime.datetime.now().time()}. {served} out of {len(requests)} requests were served!{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Calculation finished with {(end_time - start_time):.2f}.{Style.RESET_ALL}")

    # Save results
    results_file = os.path.join(glo.RESULTS_DIRECTORY, glo.LOG_FILE)
    results = {
        'time': round(end_time - start_time),
        'requests': len(requests),
        'served': served,
        'assignments': assignments
    }
    pickle.dump(results, open(results_file, 'wb'))