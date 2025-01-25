import argparse
import src.utils.global_var as glo


def process_string(s: str) -> str:
    """
    Processes a string by removing any trailing slashes at the end of the string.

    Args:
        s (str): The string to process.

    Returns:
        str: The processed string without a trailing slash.
    """
    return s.rstrip('/') if s.endswith('/') else s


def initialize():
    """
    Initializes the global simulation settings based on command-line arguments and updates global variables.
    The argparse module is used to parse the input arguments, which include paths, simulation parameters, and algorithm options.
    These arguments are assigned to both a global `config` dictionary and individual global variables for easy access across modules.
    
    Returns:
        dict: A dictionary containing the parsed configuration settings.
    """

    parser = argparse.ArgumentParser(description="Initialize simulation settings based on command-line arguments.")

    # Defining arguments
    parser.add_argument('--DATAROOT', type=str, default=glo.DATAROOT, help="Root directory for data.")
    parser.add_argument('--RESULTS_DIRECTORY', type=str, default=glo.RESULTS_DIRECTORY, help="Directory for storing results.")
    parser.add_argument('--TIMEFILE', type=str, default=glo.TIMEFILE, help="File for time data.")
    parser.add_argument('--EDGECOST_FILE', type=str, default=glo.EDGECOST_FILE, help="File for edge cost data.")
    parser.add_argument('--LOG_FILE', type=str, default=glo.LOG_FILE, help="File for result log.")
    parser.add_argument('--VEHICLE_LIMIT', type=int, default=glo.VEHICLE_LIMIT, help="Limit on the number of vehicles.")
    parser.add_argument('--MAX_WAITING', type=int, default=glo.MAX_WAITING, help="Maximum waiting time.")
    parser.add_argument('--MAX_DETOUR', type=int, default=glo.MAX_DETOUR, help="Maximum detour time.")
    parser.add_argument('--REQUEST_DATA_FILE', type=str, default=glo.REQUEST_DATA_FILE, help="File for request data.")
    parser.add_argument('--VEHICLE_DATA_FILE', type=str, default=glo.VEHICLE_DATA_FILE, help="File for vehicle data.")
    parser.add_argument('--CARSIZE', type=int, default=glo.CARSIZE, help="Vehicle capacity.")
    parser.add_argument('--INITIAL_TIME', type=int, default=glo.INITIAL_TIME, help="Initial time (HHMMSS format).")
    parser.add_argument('--FINAL_TIME', type=int, default=glo.FINAL_TIME, help="Final time (HHMMSS format).")
    parser.add_argument('--ALGORITHM', type=str, choices=list(glo.algorithm_index.keys()), default=glo.ALGORITHM, help="Algorithm to use.")
    parser.add_argument('--CTSP', type=str, choices=list(glo.ctsp_index.keys()), default=glo.CTSP, help="CTSP variant to use.")
    parser.add_argument('--CTSP_OBJECTIVE', type=str, choices=list(glo.ctspobjective_index.keys()), default=glo.CTSP_OBJECTIVE, help="CTSP objective.")
    parser.add_argument('--ALPHA', type=float, default=glo.ALPHA, help="Alpha parameter.")
    parser.add_argument('--ASSIGNMENT_OBJECTIVE', type=str, choices=list(glo.assignmentobjective_index.keys()), default=glo.ASSIGNMENT_OBJECTIVE,
                        help="Assignment objective to use.")
    parser.add_argument('--LAST_MINUTE_SERVICE', type=str, choices=["true", "false"], default="false",
                        help="Enable or disable last minute service.")
    parser.add_argument('--INTERVAL', type=int, default=glo.INTERVAL, help="Time interval for simulation.")
    parser.add_argument('--RTV_TIMELIMIT', type=int, default=glo.RTV_TIMELIMIT, help="RTV time limit.")
    parser.add_argument('--DWELL_PICKUP', type=int, default=glo.DWELL_PICKUP, help="Dwell time for pickups.")
    parser.add_argument('--DWELL_ALIGHT', type=int, default=glo.DWELL_ALIGHT, help="Dwell time for alighting.")
    parser.add_argument('--PRUNING_RR_K', type=int, default=glo.PRUNING_RR_K, help="Pruning RR parameter.")
    parser.add_argument('--PRUNING_RV_K', type=int, default=glo.PRUNING_RV_K, help="Pruning RV parameter.")
    parser.add_argument('--ML', type=int, default=0, help="Whether to use feasibility prediction model.")
    parser.add_argument('--THREADS', type=int, default=1, help="Number of threads.")

    # Parse the arguments
    args = parser.parse_args()

    # Assign the parsed arguments to the config dictionary and global variables
    glo.DATAROOT = process_string(args.DATAROOT)
    glo.RESULTS_DIRECTORY = process_string(args.RESULTS_DIRECTORY)
    glo.TIMEFILE = process_string(args.TIMEFILE)
    glo.EDGECOST_FILE = process_string(args.EDGECOST_FILE)
    glo.LOG_FILE = process_string(args.LOG_FILE)
    glo.VEHICLE_LIMIT = args.VEHICLE_LIMIT
    glo.MAX_WAITING = args.MAX_WAITING
    glo.MAX_DETOUR = args.MAX_DETOUR
    glo.REQUEST_DATA_FILE = process_string(args.REQUEST_DATA_FILE)
    glo.VEHICLE_DATA_FILE = process_string(args.VEHICLE_DATA_FILE)
    glo.CARSIZE = args.CARSIZE
    glo.INITIAL_TIME = args.INITIAL_TIME
    glo.FINAL_TIME = args.FINAL_TIME
    glo.ALGORITHM = glo.algorithm_index[args.ALGORITHM]
    glo.CTSP = glo.ctsp_index[args.CTSP]
    glo.CTSP_OBJECTIVE = glo.ctspobjective_index[args.CTSP_OBJECTIVE]
    glo.ALPHA = args.ALPHA
    glo.ASSIGNMENT_OBJECTIVE = glo.assignmentobjective_index[args.ASSIGNMENT_OBJECTIVE]
    glo.LAST_MINUTE_SERVICE = args.LAST_MINUTE_SERVICE.lower() == "true"
    glo.INTERVAL = args.INTERVAL
    glo.RTV_TIMELIMIT = args.RTV_TIMELIMIT
    glo.DWELL_PICKUP = args.DWELL_PICKUP
    glo.DWELL_ALIGHT = args.DWELL_ALIGHT
    glo.PRUNING_RR_K = args.PRUNING_RR_K
    glo.PRUNING_RV_K = args.PRUNING_RV_K

    return args