# Add parent path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.utils.global_var as glo

class Request:
    def __init__(self, request_id=None, origin=None, destination=None, entry_time=0, origin_longitude=None, origin_latitude=None, destination_longitude=None, destination_latitude=None, ideal_traveltime=0, boarding_time=0, alighting_time=0, latest_boarding=None, latest_alighting=None,
                 shared=False, assigned=False):
        """
        Initializes a Request object representing a passenger request with details such as origin, destination, timings, and coordinates.

        Args:
            request_id (int): The unique ID of the request.
            origin (int): Origin node.
            destination (int): Destination node.
            entry_time (int): The time of entry (request creation).
            boarding_time (int): The time of boarding.
            alighting_time (int): The time of alighting (arrival).
            latest_boarding (int): The latest possible boarding time.
            latest_alighting (int): The latest possible alighting time.
            shared (bool): Whether the request involves shared rides.
            assigned (bool): Whether the request is assigned to a vehicle.
            origin_longitude (float): Longitude of the origin.
            origin_latitude (float): Latitude of the origin.ß
            destination_longitude (float): Longitude of the destination.
            destination_latitude (float): Latitude of the destination.
            ideal_traveltime (int, optional): Ideal travel time. Defaults to 0.
        """
        self.id = request_id
        self.origin = origin
        self.destination = destination
        self.ideal_traveltime = ideal_traveltime
        self.entry_time = entry_time
        self.boarding_time = boarding_time
        self.alighting_time = alighting_time
        if latest_boarding is None:
            self.latest_boarding = entry_time + glo.MAX_WAITING
        else:
            self.latest_boarding = latest_boarding
        if latest_alighting is None:
            self.latest_alighting = entry_time + ideal_traveltime + glo.MAX_DETOUR
        else:
            self.latest_alighting = latest_alighting
        self.shared = shared
        self.assigned = assigned
        self.origin_longitude = origin_longitude
        self.origin_latitude = origin_latitude
        self.destination_longitude = destination_longitude
        self.destination_latitude = destination_latitude

    def __lt__(self, other):
        """
        Defines the behavior of the '<' (less than) operator for Request objects, comparing them by their ID.

        Args:
            other (Request): Another Request object to compare against.

        Returns:
            bool: True if the current Request's ID is less than the other's, False otherwise.
        """
        return self.id < other.id

    def __eq__(self, other):
        """
        Defines the behavior of the '==' (equal) operator for Request objects, comparing them by their ID.

        Args:
            other (Request): Another Request object to compare against.

        Returns:
            bool: True if the current Request's ID is equal to the other's, False otherwise.
        """
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)