"""
app/services/part2_extraction/graph_engine/cad_graph.py
=======================================================
Converts Shapely geometry (from DXF/PDF vector paths) into a NetworkX graph.
Allows spatial filtering of lines using DocTR text bounding boxes.
"""

import networkx as nx
from shapely.geometry import LineString, Polygon, MultiPolygon
from shapely.ops import linemerge
from loguru import logger
import math

class CADGraph:
    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: distance in mm to merge endpoints (snap nodes).
        """
        self.graph = nx.Graph()
        self.tolerance = tolerance

    def add_lines(self, lines: list[LineString]):
        """Add shapely LineStrings to the graph."""
        for line in lines:
            if line.is_empty:
                continue
            # Break down into segments
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = self._snap_point(coords[i])
                p2 = self._snap_point(coords[i+1])
                if p1 != p2:
                    # weight is length
                    length = math.dist(p1, p2)
                    self.graph.add_edge(p1, p2, weight=length, linestring=LineString([p1, p2]))
                    
    def _snap_point(self, pt: tuple[float, float]) -> tuple[float, float]:
        """Snap a point to an existing node within tolerance to ensure connectivity."""
        for node in self.graph.nodes:
            if math.dist(pt, node) <= self.tolerance:
                return node
        return (round(pt[0], 2), round(pt[1], 2))

    def filter_by_masks(self, masks: list[Polygon]):
        """
        Remove any edge that intersects with any of the mask Polygons.
        Used to erase dimension lines that touch text bounding boxes.
        """
        edges_to_remove = []
        for u, v, data in self.graph.edges(data=True):
            line = data['linestring']
            for mask in masks:
                if line.intersects(mask):
                    edges_to_remove.append((u, v))
                    break
                    
        self.graph.remove_edges_from(edges_to_remove)
        logger.debug(f"[Graph] Removed {len(edges_to_remove)} edges overlapping with masks.")
        
    def get_longest_path(self) -> list[LineString]:
        """
        Extracts the longest continuous connected component (the physical shell).
        Returns a list of connected LineStrings.
        """
        # Find all connected components
        components = list(nx.connected_components(self.graph))
        if not components:
            return []
            
        # Sort by total length of edges in the component
        components.sort(key=lambda comp: sum(self.graph[u][v]['weight'] for u, v in self.graph.subgraph(comp).edges()), reverse=True)
        
        largest_comp = components[0]
        subgraph = self.graph.subgraph(largest_comp)
        
        lines = [data['linestring'] for u, v, data in subgraph.edges(data=True)]
        return lines
