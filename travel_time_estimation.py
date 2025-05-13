import math
import torch

class TravelTimeEstimator:
    def __init__(self, speed_limit=60):
        """
        Initialize travel time estimator with default speed limit
        
        :param speed_limit: Maximum speed in km/h (default 60)
        """
        self.speed_limit = speed_limit
        self.intersection_delay = 30  # seconds per intersection
    
    def calculate_speed_from_flow(self, flow):
        """
        Calculate speed based on traffic flow using the fundamental diagram
        
        :param flow: Number of vehicles per hour (can be a tensor or scalar)
        :return: Estimated speed in km/h
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.item()  # Convert tensor to scalar if necessary
        
        if flow <= 351:
            return self.speed_limit
        
        def speed_under_capacity(flow):
            a = -1.4648375
            b = 93.75
            c = -flow
            discriminant = b**2 - 4*a*c
            speed1 = (-b + math.sqrt(discriminant)) / (2*a)
            speed2 = (-b - math.sqrt(discriminant)) / (2*a)
            return max(speed1, speed2)
        
        def speed_over_capacity(flow):
            return min(32, self.speed_limit * (1500 / flow))
        
        if flow <= 1500:
            return speed_under_capacity(flow)
        else:
            return speed_over_capacity(flow)
    
    def calculate_travel_time(self, distance, flow, num_intersections=1):
        """
        Calculate travel time considering traffic flow and intersections
        
        :param distance: Distance between SCATS sites in km
        :param flow: Traffic flow (vehicles per hour)
        :param num_intersections: Number of intersections to pass
        :return: Total travel time in hours
        """
        speed = self.calculate_speed_from_flow(flow)
        time_hours = distance / speed
        intersection_delay_hours = (num_intersections * self.intersection_delay) / 3600
        return time_hours + intersection_delay_hours
    
    def estimate_route_travel_time(self, route_segments):
        """
        Estimate total travel time for a complete route
        
        :param route_segments: List of dictionaries with keys:
                                - 'distance': distance between SCATS sites
                                - 'flow': traffic flow at the start of the segment
                                - 'intersections': number of intersections
        :return: Total estimated travel time in hours
        """
        total_time = 0
        for segment in route_segments:
            segment_time = self.calculate_travel_time(
                segment['distance'], 
                segment['flow'], 
                segment.get('intersections', 1)
            )
            total_time += segment_time
        return total_time

# Example usage
if __name__ == '__main__':
    # Create estimator
    estimator = TravelTimeEstimator()
    
    # Example route segments
    route = [
        {'distance': 2.5, 'flow': 500, 'intersections': 2},
        {'distance': 1.8, 'flow': 1200, 'intersections': 1}
    ]
    
    # Estimate total route time
    total_travel_time = estimator.estimate_route_travel_time(route)
    print(f"Estimated Route Travel Time: {total_travel_time:.2f} hours")