import time
import threading
from concurrent.futures import ThreadPoolExecutor
from src.algo.insersion import travel_timed
from src.env.struct.Trip import Trip
import src.utils.global_var as glo

mtx = threading.Lock()

def previoustrip(vehicle, network, current_time):
    """
    Generate the previous trip for a vehicle, using memory of its pending requests.

    Args:
        vehicle (Vehicle): The vehicle object.
        network (Network): The network data structure.
        current_time (int): The current time parameter.

    Returns:
        Trip: The previous trip constructed from the vehicle's pending requests.
    """
    # Initialize a new Trip object
    previous_trip = Trip()
    
    # Call the travel function with the 'MEMORY' mode
    previous_cost, previous_order = travel_timed(
        vehicle, vehicle.pending_requests, network, current_time, trigger='MEMORY'
    )
    
    # Set the attributes of the previous_trip
    previous_trip.cost = previous_cost
    previous_trip.order_record = previous_order
    previous_trip.requests = vehicle.pending_requests.copy()
    previous_trip.use_memory = True
    
    return previous_trip

def make_rtvgraph(wrap_data):
    """Generate RTV grah incrementally.

    Args:
        wrap_data (dict): dictionary containing rv graph and vehicles

    Return:
        Dictionary of feasible trips for each vehicle including both the new and pending requests. Include the previously assigned trips.
    """

    start = wrap_data['start'] # Start of job batch
    end = wrap_data['end'] # End of job batch
    data = wrap_data['data']

    current_time = data['time']
    rr_graph = data['rr_edges']
    rv_graph = data['rv_edges']
    trip_list = data['trip_list']
    network = data['network']
    vehicles = data['vehicles']

    for i in range(start, end):
        start_time = time.time()
        timeout = False

        vehicle = vehicles[i]
        rounds = []
        previous_assigned_passengers = set(vehicle.pending_requests)

        # Generate trip for onboard passengers with no new assignment
        baseline = Trip()
        result = travel_timed(vehicle, [], network, current_time, start_time, glo.RTV_TIMELIMIT, 'STANDARD')
        baseline.cost, baseline.order_record = result
        rounds.append([baseline])

        # Get initial pairing of requests connected to the vehicle in rv_graph
        with mtx:
            vehicle_id = vehicle.id
            if rv_graph.has_node(f'v{vehicle_id}'):
                # Get connected requests id of the vehicle on rv_graph
                neighbor_labels = set(label for label in rv_graph.neighbors(f'v{vehicle_id}'))
                # Retrieve the Request objects
                initial_pairing = {rv_graph.nodes[neighbor_label]['request'] for neighbor_label in neighbor_labels}
            else:
                initial_pairing = set()
        initial_pairing.update(vehicle.pending_requests) # Add assigned trip from the previous assignment

        # Generate trips with one request
        first_round = []
        for request in initial_pairing:
            path_cost, path_order = travel_timed(
                vehicle, [request], network, current_time, start_time, glo.RTV_TIMELIMIT, 'STANDARD'
            )
            if path_cost >= 0:
                trip = Trip(cost=path_cost, order_record=path_order, requests=[request])
                first_round.append(trip)
        rounds.append(first_round) # Add trip of length one

        k = 1  # Current trip size
        while rounds[k] and not timeout:
            k += 1
            if k > vehicle.capacity:
                break
            new_round = []
            existing_trips = {frozenset(trip.requests) for trip in rounds[k - 1]} # Trip list of size k-1

            for idx1, trip1 in enumerate(rounds[k - 1]):
                for idx2 in range(idx1 + 1, len(rounds[k - 1])):
                    # Timeout check
                    if glo.RTV_TIMELIMIT and (time.time() - start_time) * 1000 > glo.RTV_TIMELIMIT:
                        timeout = True
                        break

                    trip2 = rounds[k - 1][idx2]
                    combined_requests = set(trip1.requests) | set(trip2.requests)

                    # Skip if not exactly k requests or already considered
                    if len(combined_requests) != k or frozenset(combined_requests) in existing_trips:
                        continue

                    # Check for maximum new requests
                    new_requests = combined_requests - previous_assigned_passengers
                    if len(new_requests) * 2 > glo.MAX_NEW:
                        continue

                    # Check RR connectivity using rr_graph
                    if not is_rr_connected(combined_requests, rr_graph):
                        continue

                    # Check if all subsets exist
                    if not all_subsets_exist(combined_requests, rounds[k - 1]):
                        continue

                    # Check route feasibility
                    path_cost, path_order = travel_timed(
                        vehicle, list(combined_requests), network, current_time, start_time, glo.RTV_TIMELIMIT, 'STANDARD'
                    )
                    if path_cost < 0:
                        continue
                    else:
                        # Add the new trip
                        trip = Trip(cost=path_cost, order_record=path_order, requests=list(combined_requests))
                        new_round.append(trip)
                        existing_trips.add(frozenset(combined_requests))
            rounds.append(new_round)

        # Compile potential trip list
        potential_trips = [trip for round_trips in rounds for trip in round_trips if trip.cost >= 0]

        # Include previous assignment if any
        if vehicle.order_record:
            previous_trip = previoustrip(vehicle, network, current_time)
            if previous_trip.cost == -1:
                raise RuntimeError(f"Previous assignment no longer feasible for vehicle {vehicle.id}")
            potential_trips.append(previous_trip)

        # Update trip list
        with mtx:
            trip_list[vehicle] = potential_trips # trip_list: {Vehicle:[Trip]}

def is_rr_connected(requests, rr_graph):
    """Check if all requests are connected in the RR graph."""
    request_ids = [request.id for request in requests]
    for i, req_id1 in enumerate(request_ids):
        for req_id2 in request_ids[i + 1:]:
            with mtx:
                if not rr_graph.has_edge(f'r{req_id1}', f'r{req_id2}'):
                    return False
    return True

def all_subsets_exist(requests, previous_round):
    """Check if all subsets of size k-1 exist in the previous round."""
    requests_set = set(requests)
    for request in requests:
        subset = requests_set - {request}
        if not any(set(trip.requests) == subset for trip in previous_round):
            return False
    return True

def build_rtv_graph(current_time, rr_edges, rv_edges, vehicles, network, threads=1):
    """
    Build the RTV graph by sorting vehicles and running make_rtvgraph in parallel.
    """
    print("Building RTV graph")  # Assuming 'info' logs the message
    trip_list = {}  # Dictionary to store possible trips per vehicle

    # Sort the vehicles based on custom criteria
    sorted_vs = sorted(
        vehicles,
        key=lambda a: (
            # Priority 1: Vehicles that have entries in rv_edges
            len(list(rv_edges.neighbors(f'v{a.id}')))!=0,
            # Priority 2: Number of edges in rv_edges (descending)
            len(list(rv_edges.neighbors(f'v{a.id}'))),
            # Priority 3: Vehicle ID (ascending)
            -a.id
        ),
        reverse=True
    )

    # Prepare data for threading
    rtv_data = {
        'time': current_time,
        'rr_edges': rr_edges,
        'rv_edges': rv_edges,
        'trip_list': trip_list,
        'network': network,
        'vehicles': sorted_vs
    }

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=threads) as executor:
        # Calculate the range of vehicles each thread will process
        vehicles_per_thread = len(sorted_vs) // threads
        futures = []
        for i in range(threads):
            start = i * vehicles_per_thread
            end = (i + 1) * vehicles_per_thread if i < threads - 1 else len(sorted_vs)
            thread_data = {
                'start': start,
                'end': end,
                'data': rtv_data
            }
            futures.append(executor.submit(make_rtvgraph, thread_data))

        # Wait for all threads to complete
        for future in futures:
            future.result()

    return trip_list