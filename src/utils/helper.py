import os
from dataclasses import dataclass
from typing import List
from src.env.struct.Vehicle import Vehicle
from src.env.struct.Request import Request
import src.utils.global_var as glo
import networkx as nx
import pandas as pd

@dataclass
class DataPoint:
    graph: nx.Graph
    feasible: List[List[str]]
    infeasible: List[List[str]]

def get_request_delay(veh, path_new, network):
    """Calculate extra delay cased to onboard passengers by picking up a new request.

    Args:
        veh (Vehicle]): vehicle.
        path_new (list[NodeStop]): Order of stop including the new request.
        network (Network): current network
    
    Return:
        Average exra delay.
    """
    # Previous trip time
    previous = {n for n in veh.order_record if n.r in veh.passengers}
    if len(previous):
        travel_pre = 0
    else:
        travel_pre = network.get_time(veh.node, previous[0].node)

    for i in range(len(previous)-1):
        travel_pre += network.get_time(previous[i], previous[i+1])
    
    # New trip time
    travel_new = network.get_time(veh.node, path_new[0].node)
    for i in range(len(path_new)-1):
        travel_new += network.get_time(path_new[i].node, path_new[i+1].node)

    extra = (travel_new - travel_pre)/len(previous) if len(previous)!=0 else travel_new

    return extra

def read_time(timestring):
    """Process input time string to seconds
    """
    try:
        h, m, s = map(int, timestring.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError:
        raise ValueError(f"Invalid time string: {timestring}")   
    
def encode_time(timeint):
    """Process timestamp to string
    """
    hour = timeint//3600
    minute = (timeint//60)%60
    second = timeint%60

    return f'{hour:02}:{minute:02}:{second:02}'

def decode_time(timestring):
    """Process input time string to seconds
    """
    try:
        h, m, s = map(int, timestring.split(':'))
        return h * 3600 + m * 60 + s
    except ValueError as e:
        raise ValueError(f"Invalid time string: {timestring}") from e

def load_vehicles(filepath):
    """Load vehicles from file.

    Returns:
        A list of Vehicle objects.
    """
    if not os.path.exists(filepath):
        print("ERROR: Unable to open vehicles file.")
        print("\tSearching for vehicles file at:")
        print(f"\t\t{filepath}")
        raise FileNotFoundError("Vehicles file not found!")

    df = pd.read_csv(filepath, header=None, names=[
        "driver_id", "starting_node", "latitude", "longitude", "time_string", "capacity"
    ])

    # Drop rows with missing driver_id
    df = df.dropna(subset=["driver_id"])

    vehicles = []
    for _, row in df.iterrows():
        vehicle_capacity = glo.CARSIZE if glo.CARSIZE >= 0 else int(row["capacity"])
        vehicle = Vehicle(
            int(row["driver_id"]),
            0,
            vehicle_capacity,
            int(row["starting_node"]) - 1
        )
        vehicles.append(vehicle)
        if glo.VEHICLE_LIMIT > 0 and len(vehicles) >= glo.VEHICLE_LIMIT:
            break

    return vehicles

def load_requests(filepath, network):
    """Load requests from file

    Returns:
        A list of Request objects
    """
    if not os.path.exists(filepath):
        print("ERROR: Unable to open requests file.")
        print("\tSearching for requests file at:")
        print(f"\t\t{filepath}")
        raise FileNotFoundError("Requests file not found!")

    # Read CSV file into a DataFrame without a header (all values are data points)
    df = pd.read_csv(filepath, header=None, names=[
        "request_id", "origin_node", "origin_longitude", "origin_latitude",
        "destination_node", "destination_longitude", "destination_latitude",
        "requested_time_string"
    ])

    # Drop rows with missing request_id (e.g., empty lines at the end)
    df = df.dropna(subset=["request_id"])

    requests = []
    for _, row in df.iterrows():
        r = Request()
        r.origin_longitude = float(row["origin_longitude"])
        r.origin_latitude = float(row["origin_latitude"])
        r.destination_longitude = float(row["destination_longitude"])
        r.destination_latitude = float(row["destination_latitude"])
        r.origin = int(row["origin_node"]) - 1
        r.destination = int(row["destination_node"]) - 1
        r.id = int(row["request_id"])
        r.entry_time = read_time(row["requested_time_string"])
        r.latest_boarding = r.entry_time + glo.MAX_WAITING
        r.latest_alighting = r.entry_time + glo.MAX_DETOUR + network.get_time(r.origin, r.destination)
        r.ideal_traveltime = network.get_time(r.origin, r.destination)

        requests.append(r)

    return requests

def get_active_vehicles(vehicles, current_time):
    """
    Get a list of active vehicles (all vehicles are considered active in this case).

    Parameters:
        vehicles (list): A list of Vehicle objects.
        current_time (int): The current simulation time (not used in logic here).

    Returns:
        list: A list of references to Vehicle objects.
    """
    return vehicles  # Directly return the list since Python handles references automatically

def get_new_requests(requests, current_time):
    """
    Get a list of new requests that are active during the given interval.

    Parameters:
        requests (list): A list of Request objects.
        current_time (int): The current simulation time.

    Returns:
        list: A list of references to Request objects that match the time criteria.
    """
    return [
        r for r in requests
        if r.entry_time <= current_time < r.entry_time + glo.INTERVAL
    ]

def rr_weight(req1, req2, network):
    """Calculate the weight of the edge between two requests.

    Args:
        request1 (Request): The first request.
        request2 (Request): The second request.
        network (Network): The network object.

    Returns:
        float: The weight of the edge between the two requests.
    """
    # Calculate the weight
    o1, o2 = req1.origin, req2.origin
    d1, d2 = req1.destination, req2.destination
    t_base = network.get_time(o1, d1) + network.get_time(o2, d2)

    # Calculate detour for difference scenarios
    n = 0
    detour = 0
    scenarios = [[o1, o2, d2, d1],
                 [o1, o2, d1, d2],
                 [o1, d1, o2, d2],
                 [o2, o1, d1, d2],
                 [o2, o1, d2, d1],
                [o2, d2, o1, d1],]
    latest_visit = {o1: req1.latest_boarding, o2: req2.latest_boarding, 
                    d1: req1.latest_alighting, d2: req2.latest_alighting}
    earliest_depart = {o1: req1.entry_time + glo.DWELL_PICKUP, o2: req2.entry_time + glo.DWELL_PICKUP,
                       d1: 0, d2: 0}
    dwell_time = {o1: glo.DWELL_PICKUP, o2: glo.DWELL_PICKUP, 
                  d1: glo.DWELL_ALIGHT, d2: glo.DWELL_ALIGHT}
    for scenario in scenarios:
        current_time = 0
        feasible = True
        for i, stop in enumerate(scenario):
            if i != 0:
                current_time += network.get_time(scenario[i-1], stop)
                if current_time > latest_visit[stop]:
                    feasible = False
                    break
                else:
                    current_time = max(current_time + dwell_time[stop], earliest_depart[stop])
        if feasible:
            n += 1
            detour += current_time/(t_base+1e-5)
    
    if n == 0:
        return -1
    # Calculate the average detour
    weight = (6/n) * (detour/n)

    return weight