class NodeStop:
    """
    Stop of request
    """
    def __init__(self, request, is_pickup, node):
        """
        Initializes a NodeStop object representing a stop (either pickup or drop-off) for a Request.

        Args:
            request (Request): The Request associated with the stop.
            is_pickup (bool): Whether the stop is a pickup.
            node (int): The node (location) of the stop.
        """
        self.r = request
        self.is_pickup = is_pickup
        self.node = node

    def __lt__(self, other):
        """
        Defines the behavior of the '<' (less than) operator for NodeStop objects.
        Stops are first compared by request, then by whether the stop is a pickup.

        Args:
            other (NodeStop): Another NodeStop object to compare against.

        Returns:
            bool: True if the current NodeStop comes before the other, False otherwise.
        """
        if self.r < other.r:
            return True
        elif self.r == other.r and self.is_pickup < other.is_pickup:
            return True
        else:
            return False
    
    def __eq__(self, other):
        """
        Defines the behavior of the '=' (equal) operator for NodeStop objects.
        Stops are compared by request and whether the stop is a pickup.

        Args:
            other (NodeStop): Another NodeStop object to compare against.

        Returns:
            bool: True if the current NodeStop is the same as the other, False otherwise.
        """
        if self.r == other.r and self.is_pickup == other.is_pickup:
            return True
        else:
            return False
        
class Trip:
    """
    Trip of vehicle
    """
    def __init__(self, cost=0.0, is_fake=False, use_memory=False, order_record=None, requests=None):
        self.cost = cost                      # Cost of the trip
        self.is_fake = is_fake                # Flag indicating if it's a fake trip
        self.use_memory = use_memory          # Memory usage flag
        self.order_record = order_record if order_record else []  # Order of nodes (NodeStop instances)
        self.requests = requests if requests else []  # List of Request objects