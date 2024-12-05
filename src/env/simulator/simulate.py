import inspect
import numpy as np
import src.utils.global_var as glo
from concurrent.futures import ThreadPoolExecutor
from src.algo.insersion import travel

def move_jobless_vehicle(vehicle, network):
    """Move vehicle without an assignment.

    Args:
        vehicle (Vehicle): Vehicle object.
        network (Network): network object.
        current_time (int): current time.
    """
    origin = vehicle.prev_node
    desination = vehicle.node

    if vehicle.offset <= glo.INTERVAL:
        # Move vehicle to the destination and update travel distance
        distance = network.get_distance(origin,desination)
        vehicle.add_distance(distance)
        vehicle.prev_node = desination
        vehicle.offset = 0
    else:
        # Update required time before arriving at destination
        vehicle.offset -= glo.INTERVAL
    vehicle.order_record.clear()

def move_vehicle(vehicle, trip, network, current_time):
    """ Move vehicle with assignment.

    Args:
        vehicle (Vehicle): Vehicle object.
        trip (Trip): Trip assigned to the vehicle.
        network (Network): network object.
        current_time (int): current time.
    """
    # Initial setup
    new_requests = trip.requests
    pending_requests = set(new_requests)
    rebalancing = trip.is_fake
    if rebalancing:
        trigger = 'REBALANCING'
    elif trip.use_memory:
        trigger = 'MEMORY'
    else:
        trigger = 'STANDARD'
    
    # Get rebalancing target if the trip is rebalancing trip
    vehicle.rebalance_target = trip.requests[0].origin if rebalancing else -1

    # Determine path and cost
    if not trip.order_record:
        raw_cost, path = travel(vehicle, new_requests, network, current_time, trigger=trigger)
    else:
        raw_cost, path = trip.cost, trip.order_record

    onboard = set(vehicle.passengers)

    # Check for simulation crash
    if raw_cost == -1:
        raise RuntimeError(f"Simulation error in simulator::Line {inspect.currentframe().f_lineno}")
    
    interrupted = False
    traveltime_left = glo.INTERVAL
    job_completed = 0

    # Set vehicle initial state
    if not rebalancing and path and not vehicle.passengers:
        vehicle.set_state("EnRoute", current_time)
    elif rebalancing:
        vehicle.set_state("Rebalancing", current_time)

    # Vehicle movement towrds current destination
    if vehicle.offset < traveltime_left:
        current_time += vehicle.offset
        traveltime_left -= vehicle.offset
        vehicle.offset = 0
        vehicle.prev_node = vehicle.node
    else:
        current_time += traveltime_left
        vehicle.offset -= traveltime_left
        traveltime_left = 0

    # Latest time to start traveling to each desitination in path and each action must be completed by from current location
    latest_start = np.zeros(len(path), dtype=int)
    latest_execution = []
    duration = []

    if glo.LAST_MINUTE_SERVICE:
        current_location = vehicle.node

        # Compute durations and deadlines
        for node in path:
            if node.is_pickup:
                latest_execution.append(node.r.latest_boarding)
            else:
                latest_execution.append(node.r.latest_alighting)
            
            # Calculate travel duration between nodes
            duration.append(network.get_time(current_location, node.node))
            current_location = node.node

        # Update deadlines from back, write start times
        for i in range(len(latest_execution) - 1, -1, -1):  # Iterate backward
            latest_start[i] = latest_execution[i] - duration[i]
            if i > 0:
                latest_execution[i - 1] = min(latest_execution[i - 1], latest_start[i])

    # Move along the assigned path
    for i, node_stop in enumerate(path):
        if traveltime_left <= 0:
            break
        
        r = node_stop.r
        is_pickup = node_stop.is_pickup
        target_node = node_stop.node
        
        # Wait at stop as long as possible
        if (glo.LAST_MINUTE_SERVICE) and (not rebalancing):
            delay = latest_start[i] - current_time
            if delay < 0:
                raise RuntimeError("Delay for last minute service was negative!!! In simulator.")
            current_time += delay
            traveltime_left -= delay

        # Simulate travel to the next node
        waypoints = network.dijkstra(vehicle.node, target_node)

        # If only one waypoint, update directly
        if len(waypoints) == 1:
            vehicle.prev_node = waypoints[0]
            vehicle.node = waypoints[0]
            vehicle.offset = 0
        else:
            # Traverse waypoints
            for origin, destination in zip(waypoints[:-1], waypoints[1:]):
                traveltime = network.get_time(origin, destination)
                vehicle.prev_node, vehicle.node = origin, destination

                if traveltime >= traveltime_left:
                    interrupted = True
                    current_time += traveltime_left
                    vehicle.offset = traveltime - traveltime_left
                    traveltime_left = 0
                    break
                else:
                    current_time += traveltime
                    traveltime_left -= traveltime
                    vehicle.add_distance(network.get_distance(origin, destination))
                    vehicle.prev_node = destination

        if interrupted:
            break
        
        if traveltime_left <= 0:
            interrupted = True
            break

        # Process dwell logic (arrived at destination)
        job_completed += 1
        # If this is end of rebalancing
        if rebalancing and target_node == new_requests[0].origin:
            vehicle.rebalance_target = -1
            vehicle.set_state("Idle", current_time)
            break

        if not is_pickup:  # Drop-off logic
            r.alighting_time = current_time
            vehicle.just_alighted.append(r)
            onboard.remove(r)
            if not onboard:
                vehicle.set_state("Idle", current_time)
        else:  # Pickup logic
            r.boarding_time = current_time
            vehicle.just_boarded.append(r)
            pending_requests.remove(r)
            onboard.add(r)
            vehicle.set_state("InUse", current_time)
            if len(onboard) > 1:
                for req in onboard:
                    req.shared = True

        # Simplified batched dwell logic. Must assigned the logic in Network.get_time().
        node_type = (
            -20 if not is_pickup and (i + 1 == len(path) or path[i + 1].is_pickup or path[i + 1].node != target_node)
            else -10 if is_pickup and (i + 1 == len(path) or not path[i + 1].is_pickup or path[i + 1].node != target_node)
            else target_node
        )

        dwell = network.get_time(node_type, vehicle.node)
        if dwell >= traveltime_left:
            vehicle.prev_node = node_type
            interrupted = True
            vehicle.offset = dwell - traveltime_left
            break
        else:
            traveltime_left -= dwell
            current_time += dwell

    # Final updates for the vehicle
    vehicle.passengers = list(onboard)
    vehicle.order_record.clear()
    if trigger != "REBALANCING":
        vehicle.order_record.extend(path[job_completed:])
    vehicle.pending_requests = list(pending_requests)

    if rebalancing:
        vehicle.set_state("Idle", current_time)

def simulate_vehicle(vehicle, assignments, network, current_time):
    """Simulate behavior of a single vehicle

    Args:
        vehicle (Vehicle): vehicle object
        assignments (Dict[Vehicle:[List[Trip]]]): trip assignement for each vehicle
        network (Network): Network object
        current_time (int): current time stamp
    """
    # Prepare simulation
    vehicle.just_boarded.clear()
    vehicle.just_alighted.clear()
    vehicle.pending_requests.clear()

    # Fetch the trip assigned to the vehicle
    trip = assignments.get(vehicle, None)

    # Dispatch by job type
    # if trip and (trip.requests or vehicle.passengers):
    #     print("True")
    if trip and (trip.requests or vehicle.passengers):
    # if trip and (len(trip.requests)!=0 or len(vehicle.passengers)!=0):
        # Move vehicle with delivery task
        move_vehicle(vehicle, trip, network, current_time)
    elif vehicle.offset:
        # Move vehicle without assigned task but still on way to last destination
        move_jobless_vehicle(vehicle, network)
    else:
        vehicle.order_record.clear()


def simulate_dispatch(start, end, vehicles, assignments, network, current_time):
    """Simulate a range of vehicles."""
    for i in range(start, end):
        simulate_vehicle(vehicles[i], assignments, network, current_time)


def simulate_vehicles(vehicles, assignments, network, current_time, num_threads=1):
    """Simulate all vehicles, using ThreadPoolExecutor for multithreading."""

    num_vehicles = len(vehicles)
    chunk_size = (num_vehicles + num_threads - 1) // num_threads  # Divide vehicles into chunks

    def chunk_simulation(chunk_start):
        """Run simulation for a specific chunk of vehicles."""
        chunk_end = min(chunk_start + chunk_size, num_vehicles)
        simulate_dispatch(chunk_start, chunk_end, vehicles, assignments, network, current_time)

    # Use ThreadPoolExecutor for parallel simulation
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(chunk_simulation, range(0, num_vehicles, chunk_size))