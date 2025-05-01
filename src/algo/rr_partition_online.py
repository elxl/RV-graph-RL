from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math, copy, random
import time, os
import threading
import pymetis
import numpy as np
import networkx as nx
from src.algo.insersion import travel
from src.env.struct.Vehicle import Vehicle
from src.env.struct.Request import Request
from src.env.struct.Network import Network
from src.algo.insersion import travel_timed
from src.algo.rtvgenerator import previoustrip, delay_all
from src.env.struct.Trip import Trip, NodeStop
from src.utils.helper import rr_weight, graph_to_pymetis_inputs
from operator import itemgetter
from gurobipy import Model, GRB, quicksum
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering

import src.utils.global_var as glo
# Create locks for each shared resource
lock = threading.Lock()

def make_rrgraph(rr_data):
    """Build RR edge on RR graph using NetworkX with weights.

    Args:
        rr_data (nx.graph): RR graph.
    """

    # rr_graph = rr_data['rr_graph']
    rr_graph = nx.Graph()
    start = rr_data['start']
    end = rr_data['end']
    network = rr_data['network']
    requests = rr_data['requests']
    end = min(end, len(requests))
    current_time = rr_data['current_time']

    print(f"[Thread {threading.current_thread().name}] Processing requests {start} to {end}")

    for i in range(start, end):
        request1 = requests[i]
        with lock:
            rr_graph.add_node(f'r{request1.id}', request=request1, label='r')  # Add request node with label "r"
        compatible_requests = []

        for request2 in requests:
            if request1 == request2:
                continue

            # Prune the requests without calling the travel function
            min_wait = network.get_time(request1.origin, request2.origin)
            if min_wait+max(current_time,request1.entry_time) > request2.latest_boarding:
                continue

            weight = rr_weight(request1, request2, network)
            if weight >= 0:
                compatible_requests.append((request2,weight))
        # Keep the top k links
        compatible_requests.sort(key=lambda req: req[1])
        if glo.PRUNING_RR_K and len(compatible_requests) > glo.PRUNING_RR_K:
            compatible_requests = compatible_requests[:glo.PRUNING_RR_K]
        for request2, weight in compatible_requests:
            with lock:
                rr_graph.add_node(f'r{request2.id}', request=request2, label='r')  # Add request node with label "r"
                rr_graph.add_edge(f'r{request1.id}', f'r{request2.id}', weight=weight)  # Add rr edge with path cost
    return rr_graph

def make_rvgraph(vehicles, rr_graph, network, current_time):
    """Build RV edge on RV graph using NetworkX based on rr subgraph.

    Args:
        vehicles (List[Vehicle]): List of vehicles.
        rr_graph (nx.Graph): RR graph.
        network (Network): Network object.
        current_time (int): Current time step.
    returns:
        rv_graph(nx.Graph): RV graph with edges.
        vehicles_graph (List[Vehicle]): List of vehicles that are added to the graph.
    """

    rv_graph = nx.Graph()
    requests = [node[1]['request'] for node in rr_graph.nodes(data=True)]

    vehicles_graph = []

    for request in requests:

        with lock:
            rv_graph.add_node(f'r{request.id}', request=request, label='r')  # Add request node with label "r"

        nearest_vs = []
        buffer = 0

        for v in vehicles:
            min_wait = network.get_vehicle_time(v, request.origin) - buffer
            if current_time + min_wait > request.latest_boarding:
                continue
            nearest_vs.append((min_wait, v))

        # Sort the list based on `min_wait` (the first element of the tuple)
        nearest_vs.sort(key=itemgetter(0))

        count = 0
        for _,vehicle in nearest_vs:
            path = travel(vehicle, [request], network, current_time)
            if (glo.PRUNING_RV_K > 0) and (count >= glo.PRUNING_RV_K):
                break
            if path[0] >= 0:
                with lock:
                    rv_graph.add_node(f'v{vehicle.id}', vehicle=vehicle, label='v')  # Add vehicle node with label "v"
                    rv_graph.add_edge(f'v{vehicle.id}', f'r{request.id}', weight=path[0])  # Add rv edge with travel cost. TODO: calculate edge weights
                vehicles_graph.append(vehicle)
                count += 1
    return rv_graph, vehicles_graph

# Function to handle thread distribution
def auto_thread(job_count, function, arguments, thread_count, task):
    """Distribute jobs across threads to build graphs."""
    jobs_per_thread = job_count / float(thread_count)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for i in range(thread_count):
            start = math.ceil(i * jobs_per_thread)
            end = math.ceil((i + 1) * jobs_per_thread)
            if end > job_count:
                end = job_count  # Ensure the range doesn't exceed total job count

            if task == 'RR':
                rr_graph = arguments['rr_graph']
                network = arguments['network']
                requests = arguments['requests']
                current_time = arguments['current_time']
                data = {
                    'start': start,
                    'end': end,
                    'rr_graph': rr_graph,
                    'network': network,
                    'requests': requests,
                    'current_time': current_time,
                }
                executor.submit(function, data)

            elif task == 'TRIP':
                trip_list = arguments['trip_list']
                graph_list = arguments['graph_list']
                vehicles = arguments['vehicles']
                network = arguments['network']
                current_time = arguments['current_time']
                data = {
                    'start': start,
                    'end': end,
                    'trip_list': trip_list,
                    'graph_list': graph_list,
                    'vehicles': vehicles,
                    'network': network,
                    'current_time': current_time,
                }
                executor.submit(function, data)
            else:
                raise ValueError("Invalid task type. Use 'RR' or 'TRIP'.")


def rr_partition(requests, current_time, network, mode='None', threads=1):
    """Generate complete shareability graph using NetworkX.

    Args:
        requests (List[Request]): requests in graph.
        current_time (_type_): current time step.
        network (Network): network.
        threads (int, optional): number of threads for parallelization. Defaults to 1.

    Returns:
        merged_graph: complete graph contains both RV and RR subgraphs.
    """

    # Build RR graph
    rr_graph = nx.Graph()
    arguments = {
        'rr_graph': rr_graph,
        'network': network,
        'requests': requests,
        'current_time': current_time
    }
    auto_thread(
        job_count=len(requests),
        function=make_rrgraph,
        arguments=arguments,
        thread_count=threads,
        task='RR'
    )
    # rr_graph = make_rrgraph(arguments)

    # Partition TODO: customized partition
    if mode == 'None':
        rr_graph_lists = [rr_graph]
        sizes = [1]
    elif mode == 'Modularity':
        communities = greedy_modularity_communities(rr_graph, weight='weight')
        sizes, rr_graph_lists = zip(*[(len(community), copy.deepcopy(rr_graph.subgraph(community))) for community in communities])
    elif mode == 'Spectral':
        # Spectral clustering
        adjacency_matrix = nx.to_numpy_array(rr_graph)
        nodelist = np.array(rr_graph.nodes)
        clustering = SpectralClustering(n_clusters=glo.PARTITION_K, affinity='precomputed', assign_labels='discretize', random_state=0)
        labels = clustering.fit_predict(adjacency_matrix)
        rr_graph_lists = [copy.deepcopy(rr_graph.subgraph(nodelist[np.where(labels == i)[0]])) for i in range(glo.PARTITION_K)]
        sizes = [len(graph.nodes) for graph in rr_graph_lists]
    elif mode == 'METIS':
        # Map node labels
        nodelist = np.array(rr_graph.nodes)
        node_to_idx = {node: idx for idx, node in enumerate(rr_graph.nodes())}

        # Build adjacency
        adjacency = []
        for node in rr_graph.nodes():
            neighbors = [node_to_idx[neighbor] for neighbor in rr_graph.neighbors(node)]
            adjacency.append(neighbors)

        # Partition into 3 parts
        _, labels = pymetis.part_graph(glo.PARTITION_K, adjacency=adjacency)
        labels = np.array(labels)

        rr_graph_lists = [copy.deepcopy(rr_graph.subgraph(nodelist[np.where(labels == i)[0]])) for i in range(glo.PARTITION_K)]
        sizes = [len(graph.nodes) for graph in rr_graph_lists]        
    elif mode == 'METIS_W':
        idx_to_node, xadj, adjncy, adjwgt = graph_to_pymetis_inputs(rr_graph, weight_attr='weight')
        nodelist = np.array([idx_to_node[i] for i in range(len(idx_to_node))])

        # Partition
        _, labels = pymetis.part_graph(glo.PARTITION_K, xadj=xadj, adjncy=adjncy, eweights=adjwgt)
        labels = np.array(labels)

        rr_graph_lists = [copy.deepcopy(rr_graph.subgraph(nodelist[np.where(labels == i)[0]])) for i in range(glo.PARTITION_K)]
        sizes = [len(graph.nodes) for graph in rr_graph_lists]
    else:
        raise ValueError(f"Invalid partition mode {mode}.")

    return rr_graph_lists, sizes

def tripgenerator(wrap_data):
    """Generate trip from rr_graph incrementally.

    """

    trip_list = wrap_data['trip_list']
    vehicles = wrap_data['vehicles']
    start = wrap_data['start']
    end = wrap_data['end']
    rr_graphs = wrap_data['graph_list']
    end = min(end, len(rr_graphs))
    network = wrap_data['network']
    current_time = wrap_data['current_time']
    print(f"[{time.strftime('%H:%M:%S.%f')[:-2]}][Start thread {threading.current_thread().name}] Processing graphs {start} to {end}")

    for i in range(start, end):
        rr_graph = rr_graphs[i]
        rv_graph, vehicles_sub = make_rvgraph(vehicles, rr_graph, network, current_time)

        start_time = time.perf_counter()
        timeout = False

        for vehicle in vehicles_sub:
            rounds = []
            previous_assigned_passengers = set(vehicle.pending_requests)

            # Generate trip for onboard passengers with no new assignment (deliever onboard passengers).
            # If vehicle is already checked, skip it. Onboard trip is already included in trip_list.
            if vehicle not in trip_list:
                baseline = Trip()
                cost,path = travel_timed(vehicle, [], network, current_time, start_time, 0, 'STANDARD')
                if glo.CTSP_OBJECTIVE == "CTSP_DELAY":
                    cost = delay_all(vehicle,path,network,current_time)
                baseline.cost, baseline.order_record = cost,path
                rounds.append([baseline])
            else:
                rounds.append([])

            # Get initial pairing of requests connected to the vehicle in rv_graph
            with lock:
                vehicle_id = vehicle.id
                # Retrieve the Request objects
                initial_pairing = {rv_graph.nodes[neighbor_label]['request'] for neighbor_label in rv_graph.neighbors(f'v{vehicle_id}')}
            initial_pairing.update(vehicle.pending_requests) # Add assigned trip from the previous assignment

            # Generate trips with one request
            first_round = []
            for request in initial_pairing:
                path_cost, path_order = travel_timed(
                    vehicle, [request], network, current_time)
                if path_cost < 0:
                    print(f"Infeasible edge between v{vehicle.id} and r{request.id} at time {current_time}")
                else:
                    if glo.CTSP_OBJECTIVE == "CTSP_DELAY":
                        path_cost = delay_all(vehicle,path_order,network,current_time)
                    trip = Trip(cost=path_cost, order_record=path_order, requests=[request])
                    first_round.append(trip)
            rounds.append(first_round) # Add trip of length one

            # In round k+1, only take pairs from the previous round
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
                        if glo.RTV_TIMELIMIT and (time.perf_counter() - start_time) > glo.RTV_TIMELIMIT:
                            timeout = True
                            break

                        trip2 = rounds[k - 1][idx2]
                        combined_requests = set(trip1.requests) | set(trip2.requests)

                        # Skip if not exactly k requests or already considered
                        if len(combined_requests) != k or frozenset(combined_requests) in existing_trips:
                            continue

                        # Reject if there are too many new requests
                        new_requests = combined_requests - previous_assigned_passengers
                        if len(new_requests) * 2 > glo.MAX_NEW:
                            continue

                        # Check RR connectivity using rr_graph
                        if not is_rr_connected(trip1.requests, trip2.requests, rr_graph):
                            continue

                        # Check if all subsets exist
                        if not all_subsets_exist(combined_requests, rounds[k - 1]):
                            continue

                        # Calculate route and delay
                        path_cost, path_order = travel_timed(
                            vehicle, list(combined_requests), network, current_time, start_time, 0, trigger='STANDARD'
                        )
                        with lock:
                            if path_cost < 0:
                                continue
                            else:
                                # Add the new trip
                                if glo.CTSP_OBJECTIVE == "CTSP_DELAY":
                                    path_cost = delay_all(vehicle,path_order,network,current_time)
                                trip = Trip(cost=path_cost, order_record=path_order, requests=list(combined_requests))
                                new_round.append(trip)
                                existing_trips.add(frozenset(combined_requests))
                    if timeout:
                        break
                rounds.append(new_round)
                # print(len(new_round), end=" ")

            # Compile potential trip list
            potential_trips = [trip for round_trips in rounds for trip in round_trips]
            for trip in potential_trips:
                if trip.cost == -1:
                    raise RuntimeError("Negative cost in potential trips!!!")

            # Include previous assignment if any and if not already included
            if len(vehicle.pending_requests) < len(rounds):
                potential_trips_request_id = [[stop.r.id for stop in trip.order_record] for trip in rounds[len(vehicle.pending_requests)]]
            else:
                potential_trips_request_id = []
            if vehicle.order_record:
                request_id_vehicle = [stop.r.id for stop in vehicle.order_record]
                if request_id_vehicle not in potential_trips_request_id:               
                    previous_trip = previoustrip(vehicle, network, current_time)
                    if previous_trip.cost == -1:
                        raise RuntimeError(f"Previous assignment no longer feasible for vehicle {vehicle.id}")
                    potential_trips.append(previous_trip)
            # Update trip list
            with lock:
                if vehicle not in trip_list:
                    trip_list[vehicle] = potential_trips # trip_list: {Vehicle:[Trip]}
                else:
                    trip_list[vehicle].extend(potential_trips)

    print(f"[{time.strftime('%H:%M:%S.%f')[:-2]}][Finish thread {threading.current_thread().name}] Processing graphs {start} to {end}")

def is_rr_connected(requests1, requests2, rr_graph):
    """Check if all requests are connected in the RR graph."""
    request_ids1 = [request.id for request in requests1]
    request_ids2 = [request.id for request in requests2]
    for r1 in request_ids1:
        for r2 in request_ids2:
            if r1!=r2:
                if (not rr_graph.has_edge(f'r{r1}', f'r{r2}')):
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

def tripgenerator_parallel(rr_graph_list, vehicles, network, current_time, threads=1):
    """Generate trips in parallel from the RR graph list.
    Args:
        rr_graph_list (List[nx.Graph]): List of RR graphs.
        vehicles (List[Vehicle]): List of vehicles.
        network (Network): Network object.
        current_time (int): Calculation starts time step.
        threads (int, optional): Number of threads for parallelization. Defaults to 1.
    Returns:
        List[Trip]: List of generated trips.
    """
    trip_list = {}
    arguments = {
        'trip_list': trip_list,
        'graph_list': rr_graph_list,
        'vehicles': vehicles,
        'network': network,
        'current_time': current_time
    }
    auto_thread(
        job_count=len(rr_graph_list),
        function=tripgenerator,
        arguments=arguments,
        thread_count=threads,
        task='TRIP'
    )
    # trip_list = tripgenerator(arguments)
    return trip_list

def vehicle_assignment(trip_list, requests):
    """
    Solves the assignment problem using Integer Linear Programming with Gurobi.
`
    Args:
        trip_list (dict): A dictionary mapping Vehicle instances to a list of Trip instances.
        requests (list): A list of Request instances.

    Returns:
        dict: A dictionary mapping Vehicle instances to the assigned Trip instance.
        int: The number of requests served.
        float: The objective value of the optimization.
        int: The number of new assignments made.
    """

    # Simultaneously count variables, get cost vector, and build mapping for constraint 2
    k = len(requests)
    index = 0 # Trip index
    costs = []
    rids_to_trips = {}  # Mapping from request ID to set of trip indices that includes the request

    all_trips = []
    vehicles_list = []
    for vehicle, trips in trip_list.items():
        vehicles_list.append(vehicle)
        for trip in trips:
            costs.append(trip.cost)
            for request in trip.requests:
                rid = request.id
                if rid not in rids_to_trips:
                    rids_to_trips[rid] = set()
                rids_to_trips[rid].add(index)
            all_trips.append((vehicle, trip))
            index += 1

    if index == 0:
        return {}

    num_trips = index
    num_requests = k

    # Create a new Gurobi model
    model = Model("Assignment ILP")

    if not glo.OPTIMIZER_VERBOSE:
        model.Params.OutputFlag = 0  # Turn off Gurobi output

    # Variables
    e = model.addVars(num_trips, vtype=GRB.BINARY, name="e")  # Binary variables for trips
    x = model.addVars(num_requests, vtype=GRB.BINARY, name="x")  # Binary variables for requests. 1 stands for not assigned.

    # Objective function
    if glo.ASSIGNMENT_OBJECTIVE == 'AO_SERVICERATE':
        obj = quicksum(costs[i] * e[i] for i in range(num_trips)) + glo.MISS_COST * x.sum()
    elif glo.ASSIGNMENT_OBJECTIVE == 'AO_RMT':
        travel_times = [request.ideal_traveltime for request in requests]
        obj = quicksum(costs[i] * e[i] for i in range(num_trips)) + glo.RMT_REWARD * quicksum(travel_times[k] * x[k] for k in range(num_requests))
    else:
        # Default or raise an exception
        raise ValueError("Invalid assignment objective.")

    model.setObjective(obj, GRB.MINIMIZE)

    # Constraint One: Each vehicle is assigned at most one trip (or exactly one)
    count = 0
    for vehicle, trips in trip_list.items():
        num_vehicle_trips = len(trips)
        e_vars = [e[i] for i in range(count, count + num_vehicle_trips)]
        if glo.ALGORITHM != 'ILP_FULL':
            model.addConstr(quicksum(e_vars) <= 1, name=f"c1_{vehicle.id}")
        else:
            model.addConstr(quicksum(e_vars) == 1, name=f"c1_{vehicle.id}")
        count += num_vehicle_trips

    # Constraint Two: Each request is assigned to exactly one trip or marked as unassigned. Previously assigned request needs to be assigned.
    for k, request in enumerate(requests):
        rid = request.id
        indices = list(rids_to_trips.get(rid, []))
        e_vars = [e[i] for i in indices]
        if request.assigned:
            model.addConstr(quicksum(e_vars) == 1, name=f"c2_{rid}")
        else:
            model.addConstr(quicksum(e_vars) + x[k] == 1, name=f"c2_{rid}")

    # Optional: Set Gurobi parameters
    model.setParam("OutputFlag", 0)
    if glo.GAP:
        model.setParam("MIPGap", glo.GAP)

    # Solve the model
    model.optimize()

    # Check if the model was solved to optimality or acceptable solution
    status = model.Status
    if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print("Optimization was stopped with status", status)
        return {}
    print(f"Objective value: {model.ObjVal} \n Optimality gap: {model.MIPGap:.2f} \n Runtime {model.Runtime:.2f} seconds")

    # Retrieve assignments
    assigned_trips = {}
    count = 0
    icount = 0
    for vehicle, trips in trip_list.items():
        num_vehicle_trips = len(trips)
        e_values = [e[i].X for i in range(count, count + num_vehicle_trips)]
        for idx, val in enumerate(e_values):
            if val > 0.5:
                assigned_trips[vehicle] = trips[idx]
                if len(trips[idx].order_record)!=0:
                    icount += 1
                break
        count += num_vehicle_trips

    # Output statistics
    # icount = sum(1 for i in range(num_trips) if e[i].X > 0.5)
    print(f"Made {icount} new assignments.")
    served = 0
    for r in range(num_requests):
        if x[r].x < 0.5:
            served += 1

    return assigned_trips, served, model.ObjVal, icount