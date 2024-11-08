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