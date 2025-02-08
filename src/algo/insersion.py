"""
This file contains functions that compute the routes of assignment request(s) to a vehicle. This is the function
travel(v,r) in the original paper.
"""
from time import time
from typing import List, Tuple, Set
from src.env.struct.Trip import NodeStop
from src.env.struct.Request import Request
import src.utils.global_var as glo


class MetaNodeStop:
    """
    MetaNodeStop wraps a NodeStop and tracks other MetaNodeStops that must be unlocked 
    after this node stop is visited.

    Attributes:
        node (NodeStop): The node stop being wrapped.
        unlocks (List[MetaNodeStop]): Other MetaNodeStops that can be unlocked after this node. This enfores the precedence constraint.
    """
    def __init__(self, node: NodeStop, unlocks: List['MetaNodeStop']):
        self.node = node
        self.unlocks = unlocks

    def __lt__(self, other: 'MetaNodeStop') -> bool:
        """
        Comparison operator for sorting MetaNodeStops. Sorting is based on node values, 
        pickup/dropoff status, and unlocking sequence.
        """
        if self.node.node < other.node.node:
            return True
        if self.node.node > other.node.node:
            return False
        if self.node.is_pickup < other.node.is_pickup:
            return True
        if self.node.is_pickup > other.node.is_pickup:
            return False
        if self.node < other.node or (self.node == other.node and id(self.unlocks) < id(other.unlocks)):
            return True
        return False
    
def format_path(reverse_path: Tuple[int, List[NodeStop]], current_time: int) -> Tuple[int, List[NodeStop]]:
    """
    Formats the reverse path (reversed list of stops) into the correct order and adjusts the cost.

    Args:
        reverse_path (Tuple[int, List[NodeStop]]): A tuple with total cost and a list of NodeStops in reverse order.
        current_time (int): The current simulation time.

    Returns:
        Tuple[int, List[NodeStop]]: A tuple with the cost and the ordered list of NodeStops.
    """
    cost = reverse_path[0]
    if glo.CTSP_OBJECTIVE == 'CTSP_VTT' and cost >= 0:
        cost -= current_time
    ordered_rs = [reverse_path[1][i] for i in range(len(reverse_path[1])-1, -1, -1)]
    return cost, ordered_rs

def get_alight_deadline(r) -> int:
    """
    Returns the deadline by which a request must be dropped off.

    Args:
        r (Request): The request for which the alighting deadline is calculated.

    Returns:
        int: The alighting deadline for the request.
    """
    # return r.entry_time + r.ideal_traveltime + glo.MAX_DETOUR
    return r.latest_alighting

class Action:
    """
    Enum-like class for representing actions at each node stop.
    
    PICKUP: Represents a pickup action.
    DROPOFF: Represents a dropoff action.
    NO_ACTION: Represents no action.
    """
    PICKUP = 1
    DROPOFF = 2
    NO_ACTION = 3

def recursive_search(initial_location: int, residual_capacity: int, initially_available: Set[MetaNodeStop], 
                     network, current_time: int, best_time: int, prev_action = Action.NO_ACTION) -> Tuple[int, List[NodeStop]]:
    """
    Recursive search function to find the optimal sequence of stops (pickups/drop-offs) 
    while considering capacity and time constraints.

    Args:
        initial_location (int): The current location of the vehicle.
        residual_capacity (int): Remaining vehicle capacity.
        initially_available (Set[MetaNodeStop]): Set of available MetaNodeStops.
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.
        best_time (int): Current best time found for the route.
        prev_action (Action): The previous action (pickup/dropoff/no action).

    Returns:
        Tuple[int, List[NodeStop]]: Best time and the best sequence of NodeStops.
    """
    # Check if initially_available is a set (unsorted), then sort it.
    if isinstance(initially_available, set):
        # Convert the set into a sorted list
        initially_available = sorted(initially_available)
        
    if not initially_available:
        return (current_time, [])  # No more available stops

    best_tail = []
    previous = None

    for m in initially_available:
        if previous is not None and not m.node.is_pickup and previous.node.node == m.node.node:
            continue
        previous = m

        # Calculate the time when vehicle finishing serving m
        new_location = m.node.node
        arrival_time = current_time + network.get_time(initial_location, new_location) # arrivel_time is the time that the vehicle starts to serve the current node
        
        # if m.node.is_pickup and m.node.r.entry_time > arrival_time:
        #     arrival_time = m.node.r.entry_time

        if prev_action == Action.DROPOFF and (m.node.is_pickup or initial_location != new_location): # Pickup and dropoff at the same station
            arrival_time += glo.DWELL_ALIGHT # Add dropoff dwell time
        elif prev_action == Action.PICKUP and (not m.node.is_pickup or initial_location != new_location):
            arrival_time += glo.DWELL_PICKUP # Add pickup dwell time

        if m.node.is_pickup and m.node.r.entry_time > arrival_time:
            arrival_time = m.node.r.entry_time

        # Check optimality
        if best_time != -1 and arrival_time >= best_time:
            continue

        new_residual_capacity = residual_capacity
        if m.node.is_pickup:
            new_residual_capacity -= 1
        else:
            new_residual_capacity += 1

        if new_residual_capacity < 0:
            continue
        
        # Check time window constraint
        if m.node.is_pickup and arrival_time > m.node.r.latest_boarding:
            continue
        if get_alight_deadline(m.node.r) < arrival_time:
            continue
        
        # Remaining nodes to serve
        remaining_nodes = set(initially_available) - {m}
        for newnode in m.unlocks:
            remaining_nodes.add(newnode)

        # Check subsequent reachability
        basic_reachability = all(
            arrival_time + network.get_time(new_location, x.node.node) <=
            (x.node.r.latest_boarding if x.node.is_pickup else x.node.r.latest_alighting)
            for x in remaining_nodes
        )
        if not basic_reachability:
            continue
        
        # Update action for next node visit
        this_action = Action.PICKUP if m.node.is_pickup else Action.DROPOFF
        (tail_time, tail_stops) = recursive_search(new_location, new_residual_capacity, remaining_nodes, network, arrival_time, best_time, this_action)

        # Check feasibility of subsequent trip
        if tail_time == -1:
            continue

        if best_time == -1 or tail_time < best_time:
            best_time = tail_time
            best_tail = tail_stops
            best_tail.append(m.node) # First visit node last

    return (best_time, best_tail)

def new_travel(vehicle, requests: List['Request'], network, current_time: int) -> Tuple[int, List[NodeStop]]:
    """
    Optimizes a vehicle's route for onboard passengers and new requests.

    Args:
        vehicle (Vehicle): The vehicle with onboard passengers.
        requests (List[Request]): List of new requests.
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.

    Returns:
        Tuple[int, List[NodeStop]]: The optimized travel cost and list of NodeStops.
    """
    nodes = []
    meta_nodes = [] # Pickup nodes unlock dropoff nodes to ensure the pickup occuring before dropoff/preceding constraints for a route
    initially_available = set() # Stops that are immediately available for the vehicle to visit in the next step

    # Add stop nodes for new requests
    for r in requests:
        nodes.append(NodeStop(r, True, r.origin))  # Pickup
        nodes.append(NodeStop(r, False, r.destination))  # Dropoff
        meta_nodes.append(MetaNodeStop(nodes[-1], []))
        meta_nodes.append(MetaNodeStop(nodes[-2], [meta_nodes[-1]]))
        initially_available.add(meta_nodes[-1])
    
    # Add stop nodes for onboard passengers
    onboard = set(vehicle.passengers)
    for ns in vehicle.order_record:
        if ns.r in onboard:
            nodes.append(ns)
            meta_nodes.append(MetaNodeStop(ns, []))
            onboard.remove(ns.r)
    
    # Handling the FIX_ONBOARD case, considering the onboard passengers and new requests, and ignore pending requests
    if glo.CTSP == "FIX_ONBOARD" and len(requests) + len(vehicle.passengers) > glo.CARSIZE and len(vehicle.passengers)!=0:
        for i in range(len(vehicle.passengers) - 1):
            meta_nodes[-2 - i].unlocks = [meta_nodes[-1 - i]]
        initially_available.add(meta_nodes[-len(vehicle.passengers)]) # Follow the previous assigned order
    else:
        for i in range(len(vehicle.passengers)):
            initially_available.add(meta_nodes[-1 - i]) # Reoptimize dropoff order also for on board

    # Recompute the initially available set if the number of candidate nodes exceeds LP_LIMITVALUE
    if glo.CTSP == "FIX_PREFIX" and len(meta_nodes) > glo.LP_LIMITVALUE:
        # Determine which requests are unavailable in the previous ordering
        previous_requests = set(vehicle.pending_requests)
        new_requests = {r for r in requests if r not in previous_requests}

        if len(new_requests) * 2 > glo.LP_LIMITVALUE:
            return -1, []

        # Prepare an ordered list from the previous trip
        node_to_meta = {m.node: m for m in meta_nodes}
        previous_order = [node_to_meta[ns] for ns in vehicle.order_record if ns in node_to_meta]

        if len(previous_order) < len(meta_nodes) - glo.LP_LIMITVALUE:
            raise RuntimeError("Incorrect algebra in the FIX_PREFIX condition!")

        # Only reoptimize the last LP_LIMITVALUE orders
        captured = set(initially_available)
        initially_available = {previous_order[0]} # The order of the first len(meta_nodes) - LP_LIMITVALUE orders from the previous assignement need to be followed

        for i in range(len(meta_nodes) - glo.LP_LIMITVALUE):
            captured.remove(previous_order[i])
            for m in previous_order[i].unlocks:
                captured.add(m)
            if i + 1 < len(meta_nodes) - glo.LP_LIMITVALUE:
                previous_order[i].unlocks = [previous_order[i + 1]]
            else:
                previous_order[i].unlocks = list(captured)

    # Call the recursive cost function to compute the optimal route
    call_time = current_time + vehicle.offset
    start_node = vehicle.node
    if glo.CTSP_OBJECTIVE == "CTSP_VTT" or glo.CTSP_OBJECTIVE == "CTSP_DELAY":
        optimal = recursive_search(start_node, vehicle.capacity - len(vehicle.passengers),
                                   initially_available, network, call_time, -1)
    else:
        raise RuntimeError(f"{glo.CTSP_OBJECTIVE} is not a valid CTSP objective")

    return format_path(optimal, current_time)

def memory(vehicle, network, current_time: int) -> Tuple[int, List[NodeStop]]:
    """
    Retrieves the vehicle's route from memory, enforcing the sequence of stops from the previous assignment.
    This function is used when a pending request is assigned to the same vehicle in different time steps.

    Args:
        vehicle (Vehicle): The vehicle with a stored route.
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.

    Returns:
        Tuple[int, List[NodeStop]]: The travel cost and list of NodeStops from memory.
    """
    nodes = []
    meta_nodes = []
    for ns in vehicle.order_record:
        nodes.append(ns)
        meta_nodes.append(MetaNodeStop(ns, []))

    initially_available = {meta_nodes[0]} if meta_nodes else set()
    for i in range(1, len(meta_nodes)):
        meta_nodes[i - 1].unlocks = [meta_nodes[i]]

    call_time = current_time + vehicle.offset
    optimal = recursive_search(vehicle.node, vehicle.capacity - len(vehicle.passengers), initially_available, network, call_time, -1, Action.NO_ACTION)
    return format_path(optimal, current_time)

def travel(vehicle, requests: List['Request'], network, current_time: int, trigger='STANDARD') -> Tuple[int, List[NodeStop]]:
    """
    Determines the optimal route for a vehicle given the current state and a set of requests.

    Args:
        vehicle (Vehicle): The vehicle to route.
        requests (List[Request]): The list of new requests.
        trigger (str): The reason for the travel ("MEMORY", "REBALANCING", etc.).
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.

    Returns:
        Tuple[int, List[NodeStop]]: The travel cost and optimized sequence of NodeStops.
    """
    if trigger == "MEMORY":
        return memory(vehicle, network, current_time)
    elif trigger == "REBALANCING":
        raise NameError("Rebalancing moudle hasn't been implemented!")
    else:
        return new_travel(vehicle, requests, network, current_time)

#####################################################################################
###### The following are the time constrained version of the previous functions #####
#####################################################################################
def recursive_search_timed(initial_location: int, residual_capacity: int, initially_available: Set[MetaNodeStop],
                     network, current_time: int, best_time: int, start_time: float, time_limit: int, prev_action = Action.NO_ACTION) -> Tuple[int, List[NodeStop]]:
    """
    Recursive search function to find the optimal sequence of stops (pickups/drop-offs) 
    while considering capacity and time constraints.

    Args:
        initial_location (int): The current location of the vehicle.
        residual_capacity (int): Remaining vehicle capacity.
        initially_available (Set[MetaNodeStop]): Set of available MetaNodeStops.
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.
        best_time (int): Current best time found for the route.
        prev_action (Action): The previous action (pickup/dropoff/no action).

    Returns:
        Tuple[int, List[NodeStop]]: Best time and the best sequence of NodeStops.
    """
    # Check if initially_available is a set (unsorted), then sort it.
    if isinstance(initially_available, set):
        # Convert the set into a sorted list using the MetaNodeStop's __lt__ method
        initially_available = sorted(initially_available)
        
    if not initially_available:
        return (current_time, [])  # No more available stops

    best_tail = []
    previous = None

    for m in initially_available:
        # Check for a timeout
        if time_limit and (time() - start_time) > time_limit:
            break

        if previous is not None and not m.node.is_pickup and previous.node.node == m.node.node:
            continue
        previous = m

        # Calculate the time when vehicle finishing serving m
        new_location = m.node.node
        arrival_time = current_time + network.get_time(initial_location, new_location)
        
        # Compute time of action
        # if m.node.is_pickup and m.node.r.entry_time > arrival_time:
        #     arrival_time = m.node.r.entry_time

        if prev_action == Action.DROPOFF and (m.node.is_pickup or initial_location != new_location):
            arrival_time += glo.DWELL_ALIGHT # Add dropoff dwell time
        elif prev_action == Action.PICKUP and (not m.node.is_pickup or initial_location != new_location):
            arrival_time += glo.DWELL_PICKUP # Add pickup dwell time

        if m.node.is_pickup and m.node.r.entry_time > arrival_time:
            arrival_time = m.node.r.entry_time

        # Check optimality
        if best_time != -1 and arrival_time >= best_time:
            continue

        new_residual_capacity = residual_capacity
        if m.node.is_pickup:
            new_residual_capacity -= 1
        else:
            new_residual_capacity += 1

        if new_residual_capacity < 0:
            continue
        
        # Check time window constraint
        if m.node.is_pickup and arrival_time > m.node.r.latest_boarding:
            continue
        if get_alight_deadline(m.node.r) < arrival_time:
            continue
        
        # Remaining nodes to serve
        remaining_nodes = set(initially_available) - {m}
        for newnode in m.unlocks:
            remaining_nodes.add(newnode)

        # Check subsequent reachability
        basic_reachability = all(
            arrival_time + network.get_time(new_location, x.node.node) <=
            (x.node.r.latest_boarding if x.node.is_pickup else x.node.r.latest_alighting)
            for x in remaining_nodes
        )
        if not basic_reachability:
            continue
        
        # Update action for next node visit
        this_action = Action.PICKUP if m.node.is_pickup else Action.DROPOFF
        (tail_time, tail_stops) = recursive_search_timed(new_location, new_residual_capacity, remaining_nodes, network, arrival_time, best_time, start_time, time_limit, this_action)

        # Check feasibility of subsequent trip
        if tail_time == -1:
            continue

        if best_time == -1 or tail_time < best_time:
            best_time = tail_time
            best_tail = tail_stops
            best_tail.append(m.node) # First visit node last

    return (best_time, best_tail)

def new_travel_timed(vehicle, requests: List['Request'], network, current_time: int, start_time, time_limit: int) -> Tuple[int, List[NodeStop]]:
    """
    Optimizes a vehicle's route for onboard passengers and new requests.

    Args:
        vehicle (Vehicle): The vehicle with onboard passengers.
        requests (List[Request]): List of new requests.
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.

    Returns:
        Tuple[int, List[NodeStop]]: The optimized travel cost and list of NodeStops.
    """
    nodes = []
    meta_nodes = [] # Pickup nodes unlock dropoff nodes to ensure the pickup occuring before dropoff/preceding constraints for a route
    initially_available = set() # Stops that are immediately available for the vehicle to visit in the next step

    # Add stop nodes for new requests
    for r in requests:
        nodes.append(NodeStop(r, True, r.origin))  # Pickup
        nodes.append(NodeStop(r, False, r.destination))  # Dropoff
        meta_nodes.append(MetaNodeStop(nodes[-1], []))
        meta_nodes.append(MetaNodeStop(nodes[-2], [meta_nodes[-1]]))
        initially_available.add(meta_nodes[-1])
    
    # Add stop nodes for onboard passengers
    onboard = set(vehicle.passengers)
    for ns in vehicle.order_record:
        if ns.r in onboard:
            nodes.append(ns)
            meta_nodes.append(MetaNodeStop(ns, []))
            onboard.remove(ns.r)
    
    # Handling the FIX_ONBOARD case, considering the onboard passengers and new requests
    if glo.CTSP == "FIX_ONBOARD" and len(requests) + len(vehicle.passengers) > glo.CARSIZE and len(vehicle.passengers)!=0:
        for i in range(len(vehicle.passengers) - 1):
            meta_nodes[-2 - i].unlocks = [meta_nodes[-1 - i]]
        initially_available.add(meta_nodes[-len(vehicle.passengers)]) # Follow the previous drop off order for onboard
    else:
        for i in range(len(vehicle.passengers)):
            initially_available.add(meta_nodes[-1 - i]) # Reoptimize dropoff order for on board

    # Recompute the initially available set if the number of candidate nodes exceeds LP_LIMITVALUE
    if glo.CTSP == "FIX_PREFIX" and len(meta_nodes) > glo.LP_LIMITVALUE:
        # Determine which requests are unavailable in the previous ordering
        previous_requests = set(vehicle.pending_requests)
        new_requests = {r for r in requests if r not in previous_requests}

        if len(new_requests) * 2 > glo.LP_LIMITVALUE:
            return -1, []

        # Prepare an ordered list from the previous trip
        node_to_meta = {m.node: m for m in meta_nodes}
        previous_order = [node_to_meta[ns] for ns in vehicle.order_record if ns in node_to_meta]

        if len(previous_order) < len(meta_nodes) - glo.LP_LIMITVALUE:
            raise RuntimeError("Incorrect algebra in the FIX_PREFIX condition!")

        # Initialize states for recomputing availability
        captured = set(initially_available)
        initially_available = {previous_order[0]}

        for i in range(len(meta_nodes) - glo.LP_LIMITVALUE):
            captured.remove(previous_order[i])
            for m in previous_order[i].unlocks:
                captured.add(m)
            if i + 1 < len(meta_nodes) - glo.LP_LIMITVALUE:
                previous_order[i].unlocks = [previous_order[i + 1]]
            else:
                previous_order[i].unlocks = list(captured)

    # Call the recursive cost function to compute the optimal route
    call_time = current_time + vehicle.offset
    # start_time = time()
    start_node = vehicle.node
    if glo.CTSP_OBJECTIVE == "CTSP_VTT" or glo.CTSP_OBJECTIVE == "CTSP_DELAY":
        optimal = recursive_search_timed(start_node, vehicle.capacity - len(vehicle.passengers),
                                   initially_available, network, call_time, start_time, time_limit, -1)
    else:
        raise RuntimeError(f"{glo.CTSP_OBJECTIVE} is not a valid CTSP objective")

    return format_path(optimal, current_time)

def travel_timed(vehicle, requests: List['Request'], network, current_time: int, start_time=0, time_limit=0, trigger='STANDARD') -> Tuple[int, List[NodeStop]]:
    """
    Determines the optimal route for a vehicle given the current state and a set of requests.

    Args:
        vehicle (Vehicle): The vehicle to route.
        requests (List[Request]): The list of new requests.
        trigger (str): The reason for the travel ("MEMORY", "REBALANCING", etc.).
        network (Network): The network object for calculating travel times.
        current_time (int): The current simulation time.
        time_limit (int): Computation time limit

    Returns:
        Tuple[int, List[NodeStop]]: The travel cost and optimized sequence of NodeStops.
    """
    if not time_limit:
        return travel(vehicle, requests, network, current_time, trigger)
    else:
        return new_travel_timed(vehicle, requests, network, current_time, start_time, time_limit)