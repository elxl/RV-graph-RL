from concurrent.futures import ThreadPoolExecutor
import math, copy, random
import time
import threading
import pymetis
import torch
import numpy as np
import networkx as nx
from src.algo.insersion import travel_novehicle
from src.env.struct.Trip import Trip, NodeStop
from src.utils.helper import rr_weight
from gurobipy import Model, GRB, quicksum
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering

import src.utils.global_var as glo
# Create locks for each shared resource
lock = threading.Lock()
thread_local = threading.local()

def get_model():
    """Load the model for feasibility prediction.
    """
    if not hasattr(thread_local, "model"):
        # model = Struc2Vec(
        #     p_dim=16,
        #     nfeatures_vehicle=2,
        #     nfeatures_pickup=2,
        #     nfeatures_dropoff=2,
        #     nfeatures_edge=2,
        #     r=4
        # )
        # checkpoint = torch.load(glo.MODEL_PATH, map_location='cpu')
        # model.load_state_dict(checkpoint['model_state_dict'])
        thread_local.model = torch.jit.load(glo.MODEL_PATH)
        thread_local.model.eval()
        # thread_local.model.eval()
        # thread_local.model = ort.InferenceSession(glo.MODEL_PATH)
    return thread_local.model

def make_rrgraph(rr_data):
    """Build RR edge on RR graph using NetworkX with weights.

    Args:
        rr_data (nx.graph): RR graph.
    """

    rr_graph = rr_data['rr_graph']
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

            weight = rr_weight(request1, request2, network, current_time)
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

            elif task == 'TRIP':
                trip_list = arguments['trip_list']
                graph_list = arguments['graph_list']
                network = arguments['network']
                current_time = arguments['current_time']
                data = {
                    'start': start,
                    'end': end,
                    'trip_list': trip_list,
                    'graph_list': graph_list,
                    'network': network,
                    'current_time': current_time,
                }
            else:
                raise ValueError("Invalid task type. Use 'RR' or 'TRIP'.")

            executor.submit(function, data)


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
    else:
        raise ValueError(f"Invalid partition mode {mode}.")

    return rr_graph_lists, sizes

def tripgenerator(wrap_data):
    """Generate trip from rr_graph incrementally.

    """

    trip_list = wrap_data['trip_list']
    start = wrap_data['start']
    end = wrap_data['end']
    rr_graphs = wrap_data['graph_list']
    end = min(end, len(rr_graphs))
    network = wrap_data['network']
    current_time = wrap_data['current_time']
    model_mode = wrap_data['model_mode']

    model = get_model()
    print(f"[{time.strftime('%H:%M:%S.%f')[:-2]}][Start thread {threading.current_thread().name}] Processing graphs {start} to {end}")

    for i in range(start, end):
        rr_graph = rr_graphs[i]

        start_time = time.perf_counter()
        timeout = False

        # Make trip list up to size k.
        rounds = []

        # Intital request of size 1
        initial_pairing = {node[1]['request'] for node in rr_graph.nodes(data=True)}

        # Generate trips with one request
        first_round = []
        for request in initial_pairing:
            path_cost  = network.get_time(request.origin, request.destination)
            node1 = NodeStop(request, True, request.origin)
            node2 = NodeStop(request, False, request.destination)
            path_order = [node1, node2]
            trip = Trip(cost=path_cost, order_record=path_order, requests=[request])
            first_round.append(trip)
        rounds.append(first_round) # Add trip of length one

        # In round k+1, only take pairs from the previous round
        k = 1  # Current trip size
        while rounds[k-1] and not timeout:
            k += 1
            if k > glo.MAX_NEW:
                break
            new_round = []
            existing_trips = {frozenset(trip.requests) for trip in rounds[k - 2]} # Trip list of size k-1

            for idx1, trip1 in enumerate(rounds[k - 2]):
                for idx2 in range(idx1 + 1, len(rounds[k - 2])):
                    # Timeout check
                    if glo.RTV_TIMELIMIT and (time.perf_counter() - start_time) > glo.RTV_TIMELIMIT:
                        timeout = True
                        break

                    trip2 = rounds[k - 2][idx2]
                    combined_requests = set(trip1.requests) | set(trip2.requests)

                    # Skip if not exactly k requests or already considered
                    if len(combined_requests) != k or frozenset(combined_requests) in existing_trips:
                        continue

                    # Check RR connectivity using rr_graph
                    if not is_rr_connected(trip1.requests, trip2.requests, rr_graph):
                        continue

                    # Check if all subsets exist
                    if not all_subsets_exist(combined_requests, rounds[k - 2]):
                        continue

                    # # Process clique into nn input
                    # requests = list(combined_requests)
                    # x_vehicle, x_pickup, x_dropoff, edge_index, edge_attr, node_types = process_trip_lists(current_time, [vehicle] + requests, network, evaluation=True, directed=True)
                    # node_num = x_vehicle.size(0) + x_pickup.size(0) + x_dropoff.size(0)
                    # mu = torch.zeros((node_num, 16), dtype=torch.float32)
                    # batch_index = torch.zeros(node_num, dtype=torch.int64)
                    # proba = model(x_vehicle, x_pickup, x_dropoff, edge_index, edge_attr, node_types, mu, batch_index)
                    # # Check route feasibility
                    # # if random.random() > proba[0]:
                    # #     continue

                    path_cost_min, path_order_min = travel_novehicle(list(combined_requests), network, current_time)
                    if path_cost_min < 0:
                        continue
                    else:
                        # Add the new trip
                        trip = Trip(cost=path_cost_min, order_record=path_order_min, requests=list(combined_requests))
                        new_round.append(trip)
                        existing_trips.add(frozenset(combined_requests))
                if timeout:
                    break
            rounds.append(new_round)

        # Compile potential trip list
        potential_trips = [trip for round_trips in rounds for trip in round_trips]
        for trip in potential_trips:
            if trip.cost == -1:
                raise RuntimeError("Negative cost in potential trips!!!")

        # Update trip list
        trip_list.extend(potential_trips) # trip_list: [Trip]
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

def tripgenerator_parallel(rr_graph_list, network, current_time, model, threads=1):
    """Generate trips in parallel from the RR graph list.
    Args:
        rr_graph_list (List[nx.Graph]): List of RR graphs.
        network (Network): Network object.
        current_time (int): Calculation starts time step.
        threads (int, optional): Number of threads for parallelization. Defaults to 1.
    Returns:
        List[Trip]: List of generated trips.
    """
    trip_list = []
    arguments = {
        'start': 0,
        'end': len(rr_graph_list),
        'trip_list': trip_list,
        'graph_list': rr_graph_list,
        'network': network,
        'current_time': current_time,
        'model_mode': model,
    }
    auto_thread(
        job_count=len(rr_graph_list),
        function=tripgenerator,
        arguments=arguments,
        thread_count=threads,
        task='TRIP'
    )
    # tripgenerator(arguments)
    return trip_list

def vehicle_assignment(trip_list, v_num):
    """Assign vehicles to trips based on the vehicle number.

    Args:
        trip_list (List[Trip]): List of feasible trips.
        v_num (int): Number of vehicles available.
    Returns:
        assignment (np.ndarray): Vehicle assignment matrix.
        served (int): Number of served requests.
        obj (float): Objective value of the optimization model.
    """
    requests = list({request.id for trip in trip_list for request in trip.requests})
    trip_to_request = np.zeros((len(trip_list), len(requests)), dtype=bool)
    # Contained requests in each trip
    for i, trip in enumerate(trip_list):
        for request in trip.requests:
            index = requests.index(request.id)
            trip_to_request[i, index] = True
    n_trip = len(trip_list)

    # Create optimization model
    model = Model("VehicleAssignment")

    # Decision variables
    # x[v, t] = 1 if vehicle v is assigned to trip t, 0 otherwise
    x = model.addVars(v_num, n_trip, vtype=GRB.BINARY, name="x")
    # y[r] = 1 if request r is served, 0 otherwise
    y = model.addVars(len(requests), vtype=GRB.BINARY, name="y")

    # Objective: Maximize served passengers and minimize travel time
    model.setObjective(glo.MISS_COST * quicksum(1-y[i] for i in range(len(requests))) +
                    quicksum(trip.cost * x[v, t] for v in range(v_num) for t, trip in enumerate(trip_list)), 
                    GRB.MINIMIZE)

    # Constraint: Each passenger is served at most once
    for r in range(len(requests)):
        trips_with_r = [t for t in range(n_trip) if trip_to_request[t, r]]
        model.addConstr(
            y[r] == quicksum(x[v, t] for v in range(v_num) for t in trips_with_r)
        )

    # Constraint: Each vehicle is assigned to at most one trip
    for v in range(v_num):
        model.addConstr(
            quicksum(x[v, t] for t in range(len(trip_list))) <= 1
        )

    # Optional: Set Gurobi parameters
    model.setParam("OutputFlag", 0)
    if glo.GAP:
        model.setParam("MIPGap", glo.GAP)
    elif glo.ILP_TIMEOUT:
        model.setParam("TimeLimit", glo.ILP_TIMEOUT)


    # Solve the model
    model.optimize()

    # Extract results
    if model.status == GRB.OPTIMAL:
        print(f"Objective value: {model.ObjVal} \n Optimality gap: {model.MIPGap:.2f} \n Runtime {model.Runtime:.2f} seconds")
        assignment = np.zeros((v_num, 3), dtype=float)
        served = 0
        assigned = 0
        for v in range(v_num):
            for t, trip in enumerate(trip_list):
                if x[v, t].x > 0.5:  # Vehicle v is assigned to trip t
                    assignment[v,:] = [1, trip.cost, len(trip.requests)]
                    assigned += 1
        for r in range(len(requests)):
            if y[r].x > 0.5:
                served += 1
        print(f"Assigned {assigned} vehicles to trips.")
        return assignment, served, model.ObjVal
    else:
        raise RuntimeError("Optimization model did not find an optimal solution.")