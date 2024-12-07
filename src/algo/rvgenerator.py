from concurrent.futures import ThreadPoolExecutor
import math
import threading
import networkx as nx
from src.algo.insersion import travel
from src.env.struct.Vehicle import Vehicle
from src.env.struct.Request import Request
from src.env.struct.Network import Network
from operator import itemgetter

import src.utils.global_var as glo
# Create locks for each shared resource
rv_graph_lock = threading.Lock()
rr_graph_lock = threading.Lock()

def detour_factor(req1, req2, network):
    """Calculate detour factor between two requests if they are picked one after another.

    Args:
        req1 (Request): the first request.
        req2 (Request): the second request.
        network (Network): Current network

    Returns:
        float: detour factor.
    """
    best = float('inf')
    o1, o2 = req1.origin, req2.origin
    d1, d2 = req1.destination, req2.destination

    # Get the direct distance for request a (from origin o1 to destination d1)
    onedist = network.get_time(o1, d1)
    
    if onedist:
        # Calculate detour if we visit o2 (origin of b) on the way to d1 (destination of a)
        ratio = network.get_time(o1, o2) + network.get_time(o2, d1)
        ratio /= onedist
        best = min(best, ratio)
    
    # Get the direct distance for request b (from origin o2 to destination d2)
    twodist = network.get_time(o2, d2)
    
    if twodist:
        # Calculate detour if we visit o1 (origin of a) on the way to d2 (destination of b)
        ratio = network.get_time(o2, o1) + network.get_time(o1, d2)
        ratio /= twodist
        best = min(best, ratio)
    
    # If neither onedist nor twodist are valid, set best to 0
    if not onedist and not twodist:
        best = 0
    
    return best


def make_rvgraph(rv_data):
    """Build RV edge on RV graph using NetworkX.

    Args:
        rv_data (nx.graph): RV graph.
    """
    start = rv_data['start']
    end = rv_data['end']
    rv_graph = rv_data['rv_graph']
    network = rv_data['network']
    requests = rv_data['requests']
    vehicles = rv_data['vehicles']
    current_time = rv_data['time']
    lock = rv_data['lock']

    for i in range(start, end):
        request = requests[i]
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
                count += 1

def make_rrgraph(rr_data):
    """Build RR edge on RR graph using NetworkX.

    Args:
        rr_data (nx.graph): RR graph.
    """
    start = rr_data['start']
    end = rr_data['end']
    rr_graph = rr_data['rr_graph']
    network = rr_data['network']
    requests = rr_data['requests']
    current_time = rr_data['time']
    lock = rr_data['lock']

    for i in range(start, end):
        request1 = requests[i]
        with lock:
            rr_graph.add_node(f'r{request1.id}', request=request1, label='r')  # Add request node with label "r"
        compatible_requests = []

        for request2 in requests:
            if request1 == request2:
                continue

            # Prune the requests without calling the travel function
            buffer = 0
            min_wait = network.get_time(request1.origin, request2.origin) - buffer
            if min_wait+max(current_time,request1.entry_time) > request2.latest_boarding:
                continue

            dummyvehicle = Vehicle(0, 0, 4, request1.origin)
            path = travel(dummyvehicle, [request1, request2], network, current_time)
            if path[0] >= 0:
                compatible_requests.append((request2,path[0]))
        # Kepp the top k links
        compatible_requests.sort(key=lambda req: detour_factor(request1, req[0], network))
        if glo.PRUNING_RR_K and len(compatible_requests) > glo.PRUNING_RR_K:
            compatible_requests = compatible_requests[:glo.PRUNING_RR_K]
        for request2, cost in compatible_requests:
            with lock:
                rr_graph.add_node(f'r{request2.id}', request=request2, label='r')  # Add request node with label "r"
                rr_graph.add_edge(f'r{request1.id}', f'r{request2.id}', weight=cost)  # Add rr edge with path cost. TODO: calculate edge weights

# Function to handle thread distribution
def auto_thread(job_count, function, graph, requests, network, current_time, thread_count, edge_type, vehicles=None):
    """Distribute jobs across threads to build graphs."""
    jobs_per_thread = job_count / float(thread_count)

    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        for i in range(thread_count):
            start = math.ceil(i * jobs_per_thread)
            end = math.ceil((i + 1) * jobs_per_thread)
            if end > job_count:
                end = job_count  # Ensure the range doesn't exceed total job count

            if edge_type == 'RV':
                data = {
                    'start': start,
                    'end': end,
                    'rv_graph': graph,
                    'network': network,
                    'requests': requests,
                    'vehicles': vehicles,
                    'time': current_time,
                    'lock': rv_graph_lock
                }
                # Submit the task to the thread pool
                executor.submit(function, data)
            elif edge_type == 'RR':
                data = {
                    'start': start,
                    'end': end,
                    'rr_graph': graph,
                    'network': network,
                    'requests': requests,
                    'time': current_time,
                    'lock': rr_graph_lock
                }
                executor.submit(function, data)

def rvgenerator(vehicles, requests, current_time, network, threads=1):
    """Generate complete shareability graph using NetworkX.

    Args:
        vehicles (List[Vehicle]): vehicles in graph.
        requests (List[Request]): requests in graph.
        current_time (_type_): current time step.
        network (Network): network.
        threads (int, optional): number of threads for parallelization. Defaults to 1.

    Returns:
        merged_graph: complete graph contains both RV and RR subgraphs.
    """
    # Build RV graph
    rv_graph = nx.Graph()  # Directed graph for RV (vehicle -> request)
    auto_thread(
        job_count=len(requests),
        function=make_rvgraph,
        edge_type='RV',
        graph=rv_graph,
        requests=requests,
        vehicles=vehicles,
        network=network,
        current_time=current_time,
        thread_count=threads
    )

    # Build RR graph
    rr_graph = nx.DiGraph()  # Directed graph for RR (request -> request)
    auto_thread(
        job_count=len(requests),
        function=make_rrgraph,
        edge_type='RR',
        graph=rr_graph,
        requests=requests,
        network=network,
        current_time=current_time,
        thread_count=threads
    )

    # Merge RV and RR graphs (since they share the same request nodes)
    # merged_graph = nx.compose(rv_graph, rr_graph)

    return [rv_graph, rr_graph]

# Example usage:
if __name__ == "__main__":
    config = {
        'DATAROOT': 'data/test/',
        'TIMEFILE': 'travel_time.txt',
        'DISTFILE': 'travel_time.txt',
        'EDGECOST_FILE': None,
        'DWELL_PICKUP': 0,
        'DWELL_ALIGHT': 0
    }
    net = Network(config)

    r1 = Request(1,5,1,100,0,0,0,0,200)
    r2 = Request(2,2,0,200,0,0,0,0,200)
    r3 = Request(3,3,4,200,0,0,0,0,100)

    v1 = Vehicle(1,0,3,4)
    v2 = Vehicle(2,0,3,0)

    # Generate merged RV and RR graphs
    requests = [r1,r2,r3]
    vehicles = [v1,v2]
    rv, rr = rvgenerator(vehicles,requests,0,net,3)
    graph = nx.compose(rv,rr)

    # Output graph information
    print("Merged Graph Nodes:", graph.nodes(data=True))
    print("Merged Graph Edges:", graph.edges(data=True))
