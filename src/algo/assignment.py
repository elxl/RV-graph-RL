from gurobipy import Model, GRB, quicksum
from src.algo.rvgenerator import rvgenerator
from src.algo.rtvgenerator import build_rtv_graph
import src.utils.global_var as glo


def ilp_assignment(trip_list, requests, time_param):
    """
    Solves the assignment problem using Integer Linear Programming with Gurobi.
`
    Args:
        trip_list (dict): A dictionary mapping Vehicle instances to a list of Trip instances.
        requests (list): A list of Request instances.
        time_param (int): The current time parameter.

    Returns:
        dict: A dictionary mapping Vehicle instances to assigned Trip instances.
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
    if not glo.ILP_TIMEOUT:
        model.Params.TimeLimit = 120  # Time limit in seconds
        model.Params.MIPGap = 1e-8
        # model.Params.MIPGapAbs = 0.0
        # model.Params.BestObjStop = GRB.INFINITY
    else:
        model.Params.TimeLimit = glo.ILP_TIMEOUT
        # model.Params.MIPGap = 0.05
        # model.Params.MIPGapAbs = 5
        # model.Params.BestObjStop = GRB.INFINITY

    # Solve the model
    model.optimize()

    # Check if the model was solved to optimality or acceptable solution
    status = model.Status
    if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        print("Optimization was stopped with status", status)
        return {}

    # Retrieve assignments
    assigned_trips = {}
    count = 0
    for vehicle, trips in trip_list.items():
        num_vehicle_trips = len(trips)
        e_values = [e[i].X for i in range(count, count + num_vehicle_trips)]
        for idx, val in enumerate(e_values):
            if val > 0.5:
                assigned_trips[vehicle] = trips[idx]
                break
        count += num_vehicle_trips

    # Output statistics
    icount = sum(1 for i in range(num_trips) if e[i].X > 0.5)
    print(f"Made {icount} assignments.")

    # Write statistics to a file (adjust the path as needed)
    if glo.OPTIMIZER_VERBOSE:
        with open(f"{glo.RESULTS_DIRECTORY}/ilp.csv", "a") as ilpfile:
            ilpfile.write(f"{time_param}\t")
            ilpfile.write(f"{model.ObjVal}\t")
            ilpfile.write(f"{model.Runtime}\t")
            ilpfile.write(f"{model.Params.MIPGapAbs}\t")
            ilpfile.write(f"{model.Params.MIPGap}\t")
            ilpfile.write(f"{icount}\t")
            is_optimal = status == GRB.OPTIMAL
            ilpfile.write(f"{'Optimal' if is_optimal else 'Suboptimal'}\n")

    return assigned_trips, model.ObjVal

def ilp_assignement_full(vehicles, requests, current_time, network, model, threads=1):
    """ Performs all steps in the algorithm.

    Args:
        vehicles (list): List of Vehicle instances.
        requests (list): List of Request instances.
        current_time (int): The current time parameter.
        network (Network): The network data structure.
        threads (int): Number of threads.

    Returns:
        dict: A dictionary mapping Vehicle instances to assigned Trip instances.
    """
    print("Building RV graph")
    rv_edges, rr_edges = rvgenerator(vehicles, requests, current_time, network, threads)

    print("Building RTV graph")
    trip_list, feasible, infeasible, all_cal, pass_cal = build_rtv_graph(current_time, rr_edges, rv_edges, vehicles, network, model, threads=threads)
    # trip_list_ml, _, _,_,_ = build_rtv_graph(current_time, rr_edges, rv_edges, vehicles, network, model=1, threads=threads)
    # trip_list_navie, _, _ ,_,_= build_rtv_graph(current_time, rr_edges, rv_edges, vehicles, network, model=2, threads=threads)

    # Count total number of trips
    total_trips = sum(len(trips) for trips in trip_list.values())
    # total_trips_ml = sum(len(trips) for trips in trip_list_ml.values())
    # total_trips_navie = sum(len(trips) for trips in trip_list_navie.values())
    # print(f"Trip list is of size {total_trips}\t {total_trips_ml}\t {total_trips_navie}")
    print(f"Trip list is of size {total_trips}")
    if model == 1:
        print(f"pass {pass_cal} over {all_cal}")

    # Check to ensure no previously assigned requests were rejected
    assigned_request_ids = {request.id for request in requests if request.assigned}
    included_request_ids = {request.id for trips in trip_list.values() for trip in trips for request in trip.requests}

    missing_assigned_requests = assigned_request_ids - included_request_ids
    if missing_assigned_requests:
        for request_id in missing_assigned_requests:
            print(f"Help! Request {request_id} was not included!")
        raise RuntimeError("Some previously assigned requests were not included in any trip.")

    # Check that all previous trips are included in future possibilities
    for vehicle in vehicles:
        prev_requests = set(vehicle.pending_requests)
        found = False
        for trip in trip_list.get(vehicle, []):
            if set(trip.requests) == prev_requests:
                found = True
                break
        if not found:
            print(f"Vehicle ID {vehicle.id}")
            raise RuntimeError("Did not replicate the previous trip!")

    # Optionally, output trace of generated trip_list
    # You can write to a file or process as needed
    # For example:
    # with open("rtv.log", "a") as rtv_file:
    #     rtv_file.write(f"TIME STAMP {time_param}\n")
    #     for vehicle, trips in trip_list.items():
    #         for trip in trips:
    #             trip_info = {
    #                 'v': vehicle.id,
    #                 'rs': [request.id for request in trip.requests],
    #                 'c': trip.cost
    #             }
    #             rtv_file.write(f"{trip_info}\n")

    # Perform the ILP assignment
    assigned_trips, obj = ilp_assignment(trip_list, requests, current_time)
    # _, obj_ml = ilp_assignment(trip_list_ml, requests, current_time)
    # _, obj_navie = ilp_assignment(trip_list_navie, requests, current_time)

    return assigned_trips, feasible, infeasible, obj
