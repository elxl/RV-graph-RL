"""
This module represents a network where travel times and distances between nodes are calculated.
It includes methods to handle vehicle travel offsets and shortest-path calculations using Dijkstra's algorithm.
"""

import os
import math
from collections import deque, defaultdict, namedtuple

Neighbor = namedtuple("Neighbor", ["target","weight"])
def return_none():
    return None

class Network:
    def __init__(self, config):
        """
        Initializes the network by loading time and distance matrices from files.
        Additionally, it loads an adjacency list for Dijkstra's shortest path calculations.
        
        Args:
            config (dict): The configuration dictionary containing all settings, 
                           including time file paths and dwell times.
        """
        self.time_matrix = self.load_matrix(os.path.join(config['DATAROOT'], config['TIMEFILE']))
        self.distance_matrix = self.load_matrix(os.path.join(config['DATAROOT'], config['DISTFILE'])) 
        if config['EDGECOST_FILE'] is not None:
            self.adjacency_list = self.load_adjacency_list(os.path.join(config['DATAROOT'], config['EDGECOST_FILE']))
        self.dwell_pickup = config['DWELL_PICKUP']
        self.dwell_alight = config['DWELL_ALIGHT']
        self.shortest_paths = defaultdict(return_none) # Store shortest path to avoid recomputation

    def load_matrix(self, file_path):
        """
        Loads a matrix from a CSV file.
        
        Args:
            file_path (str): Path to the file containing the matrix data.
        
        Returns:
            list of list of int: The loaded matrix.
        """
        matrix = []
        with open(file_path, 'r') as f:
            for line in f:
                row = [int(value) for value in line.strip().split(",")]
                matrix.append(row)
        return matrix

    def load_adjacency_list(self, file_path):
        """
        Loads the adjacency list for Dijkstra's algorithm.
        
        Args:
            file_path (str): Path to the file containing adjacency list data.
        
        Returns:
            list of list of tuple: The adjacency list, where each entry is a tuple of (target, weight).
        """
        adjacency_list = []
        with open(file_path, 'r') as f:
            for line in f:
                origin, dest, length = map(int, line.strip().split(","))
                origin, dest = origin - 1, dest - 1  # Adjust to zero-index
                if len(adjacency_list) < origin + 1:
                    adjacency_list.extend([[] for _ in range(origin + 1 - len(adjacency_list))])
                adjacency_list[origin].append(Neighbor(target=dest, weight=length))
        return adjacency_list

    def get_time(self, node_one, node_two):
        """
        Returns the travel time between two nodes.
        
        Args:
            node_one (int): The starting node. -10 indicates pickup, -20 indicates dropoff.
            node_two (int): The destination node.
        
        Returns:
            int: The travel time.
        """
        if node_one == -10:
            return self.dwell_pickup
        elif node_one == -20:
            return self.dwell_alight
        return self.time_matrix[node_one][node_two]

    def get_distance(self, node_one, node_two):
        """
        Returns the distance between two nodes.
        
        Args:
            node_one (int): The starting node.
            node_two (int): The destination node.
        
        Returns:
            int: The distance (same as time for now).
        """
        distance = self.distance_matrix[node_one][node_two]
        
        if node_one == -10 or node_one == -20:
            return 0
        return distance
    
    def get_vehicle_offset(self, vehicle):
        """
        Gets the distance offset for a given vehicle.
        
        Args:
            vehicle (Vehicle): The vehicle object containing current and previous nodes and offset (traveled distance from the previous node).
        
        Returns:
            int: The distance offset.
        """
        origin = vehicle.prev_node
        destination = vehicle.node

        if origin < 0 or destination < 0:  # Error checker, can be removed if not needed
            return 0

        travel_time = self.get_time(origin, destination)
        distance = self.get_distance(origin, destination)
        time_offset = travel_time - vehicle.offset

        if travel_time == 0:
            return 0
        else:
            percent = time_offset / travel_time
            distance_offset_true = distance * percent
            distance_offset = int(math.floor(distance_offset_true))
            return distance_offset

    def get_vehicle_distance(self, vehicle, node):
        """
        Calculates the total distance a vehicle must travel from its current position to a given node.
        
        Args:
            vehicle (Vehicle): The vehicle object containing current and previous nodes.
            node (int): The destination node.
        
        Returns:
            int: The total distance the vehicle will travel.
        """
        origin = vehicle.prev_node
        destination = vehicle.node
        current_leg = self.get_distance(origin, destination) - self.get_vehicle_offset(vehicle)
        final_leg = self.get_distance(destination, node)
        return current_leg + final_leg

    def get_vehicle_time(self, vehicle, node):
        """
        Calculates the time a vehicle needs to travel from its current node to a given node.
        
        Args:
            vehicle (Vehicle): The vehicle object containing the current node and offset.
            node (int): The destination node.
        
        Returns:
            int: The total travel time.
        """
        return vehicle.offset + self.get_time(vehicle.node, node)

    def dijkstra(self, origin, destination):
        """
        Finds the shortest path from origin to destination using a custom Dijkstra's-like algorithm.
        
        Args:
            origin (int): The starting node.
            destination (int): The target node.
        
        Returns:
            list: The shortest path from origin to destination as a list of node indices.
        """
        if not (self.shortest_paths[(origin,destination)] is None):
            return self.shortest_paths[(origin,destination)]
        
        path = [origin]
        here = origin
        count = 0

        while here != destination and here != -1 and count < 200:
            best = self.get_time(here, destination) + 1
            node = -1

            # Try to take a strict step.
            for neighbor in self.adjacency_list[here]:
                n_node = neighbor.target
                weight = neighbor.weight

                if n_node == destination:
                    node = n_node
                    break

                follow_up = self.get_time(n_node, destination)
                if weight > 0 and weight + follow_up < best:
                    best = weight + follow_up
                    node = n_node

            # Logic for handling zero-weight paths if no good choice was found.
            if node == -1:
                zeros = deque()
                heritage = defaultdict(list)
                comparison = self.get_time(here, destination)

                for neighbor in self.adjacency_list[here]:
                    if neighbor.weight + self.get_time(neighbor.target, destination) <= comparison:
                        zeros.append(neighbor)
                        heritage[neighbor.target].append(neighbor.target)

                while zeros and node == -1:
                    n = zeros.popleft()

                    for child in self.adjacency_list[n.target]:
                        if child.weight + self.get_time(child.target, destination) <= comparison:
                            if child.weight > 0 or child.target == destination:
                                path.extend(heritage[n.target])
                                node = child.target
                                break

                            if child.target not in heritage:
                                zeros.append(child)
                                new_heritage = heritage[n.target] + [child.target]
                                heritage[child.target] = new_heritage

            path.append(node)
            here = node
            count += 1

        if count > 200 or here == -1:
            print("Oops! Network dijkstra messed up somehow!")
            print(f"Query from {origin} to {destination}")
            for i in path:
                print(f"{i}\t{self.get_time(i, destination)}")
            input("Press Enter to continue...")  # Equivalent to `getchar()` in C++
        
        self.shortest_paths[(origin,destination)] = path

        return path