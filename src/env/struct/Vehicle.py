"""
This module defines the class of vehicles including all necessary features describing a vehicle's current and history state.
"""

class Vehicle:
    """Class for vehicle
    """

    # States as constants
    IDLE = "Idle"
    ENROUTE = "EnRoute"
    IN_USE = "InUse"
    REBALANCING = "Rebalancing"

    def __init__(self, vehicle_id, start_time, capacity, node):
        """
        Initializes a Vehicle object with the given ID, start time, capacity, and initial node.

        Args:
            vehicle_id (int): The unique identifier for the vehicle.
            start_time (int): The start time for the vehicle's availability.
            capacity (int): The vehicle's capacity.
            node (int): The initial node (location) of the vehicle.
        """
        self.id = vehicle_id
        self.start_time = start_time
        self.capacity = capacity
        self.is_rebalancing = False
        self.rebalance_target = -1
        self.prev_node = node
        self.node = node
        self.offset = 0 # Time required to arrive at the next node (self.node)
        self.total_rebalance_distance = 0.0
        self.total_distance_traveled = 0.0
        self.state = Vehicle.IDLE
        self.total_idle = 0
        self.total_rebalancing = 0
        self.total_enroute = 0
        self.total_inuse = 0
        self.time_stamp = 0

        self.passengers = [] # Onboard passengers
        self.just_boarded = []
        self.just_alighted = []
        self.pending_requests = [] # Assigned passengers not yet picked up
        self.order_record = [] # Order of events. List of type NodeStop. Sequence of node stops scheduled to be visited.

    def add_distance(self, distance):
        """
        Adds the given distance to the total distance traveled by the vehicle.
        If the vehicle is rebalancing, the distance is also added to the total rebalance distance.

        Args:
            distance (float): The distance traveled to be added.
        """
        self.total_distance_traveled += distance
        if self.is_rebalancing:
            self.total_rebalance_distance += distance

    def get_distance_traveled(self):
        """
        Returns the total distance traveled by the vehicle.

        Returns:
            float: The total distance traveled.
        """
        return self.total_distance_traveled

    def get_rebalance_distance(self):
        """
        Returns the total distance traveled while rebalancing.

        Returns:
            float: The total rebalance distance.
        """
        return self.total_rebalance_distance

    def set_state(self, state, current_time):
        """
        Changes the vehicle's state and updates time spent in the previous state.
        The time spent in each state (idle, enroute, in use, rebalancing) is updated accordingly.

        Args:
            state (int): The new state of the vehicle (IDLE, ENROUTE, IN_USE, REBALANCING).
            current_time (int): The current time when the latest state change occurs.
        """
        if self.time_stamp == -1:
            self.time_stamp = current_time
        if state != self.state:
            # Change of state
            duration = current_time - self.time_stamp

            if self.state == Vehicle.IDLE:
                self.total_idle += duration
            elif self.state == Vehicle.ENROUTE:
                self.total_enroute += duration
            elif self.state == Vehicle.IN_USE:
                self.total_inuse += duration
            else:  # Rebalancing
                self.total_rebalancing += duration

            self.state = state
            self.time_stamp = current_time

    def get_total_idle(self, time):
        """
        Returns the total time the vehicle has been idle, including the current idle duration if applicable.

        Args:
            time (int): The current time.

        Returns:
            int: The total idle time.
        """
        if self.state == Vehicle.IDLE:
            return self.total_idle + (time - self.time_stamp)
        return self.total_idle

    def get_total_rebalancing(self, time):
        """
        Returns the total time the vehicle has spent rebalancing, including the current rebalancing duration if applicable.

        Args:
            time (int): The current time.

        Returns:
            int: The total rebalancing time.
        """
        if self.state == Vehicle.REBALANCING:
            return self.total_rebalancing + (time - self.time_stamp)
        return self.total_rebalancing

    def get_total_enroute(self, time):
        """
        Returns the total time the vehicle has spent enroute, including the current enroute duration if applicable.

        Args:
            time (int): The current time.

        Returns:
            int: The total enroute time.
        """
        if self.state == Vehicle.ENROUTE:
            return self.total_enroute + (time - self.time_stamp)
        return self.total_enroute

    def get_total_inuse(self, time):
        """
        Returns the total time the vehicle has been in use, including the current in-use duration if applicable.

        Args:
            time (int): The current time.

        Returns:
            int: The total in-use time.
        """
        if self.state == Vehicle.IN_USE:
            return self.total_inuse + (time - self.time_stamp)
        return self.total_inuse

    def get_state(self):
        """
        Returns the current state of the vehicle.

        Returns:
            int: The current state (IDLE, ENROUTE, IN_USE, REBALANCING).
        """
        return self.state
