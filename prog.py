import networkx as nx
import numpy as np
import heapq
import random
from tqdm import tqdm

# Constants from Simulation 1 in the paper
SIMULATION_LIMIT = 1_000_000
TRANSMISSION_TIME = 2
LINK_AVAILABILITY = 0.9
GEN, TRANS, RECEIVE = 'GEN', 'TRANS', 'RECEIVE'

class Event:
    def __init__(self, time, event_type, node, packet):
        self.time = time
        self.event_type = event_type
        self.node = node
        self.packet = packet

    def __lt__(self, other):
        return self.time < other.time

class Packet:
    def __init__(self, src, dst, path):
        self.src = src
        self.dst = dst
        self.path = path
        self.current_hop = 0
        self.rerouted = False

class NetworkSimulator:
    def __init__(self, G, lam, ct, pbar):
        self.G = G
        self.lambda_rate = lam
        self.ct = ct
        self.pbar = pbar
        self.event_queue = []
        self.current_time = 0

        self.queues = {node: [] for node in G.nodes}  # Per-node FIFO queues
        self.node_busy = {node: False for node in G.nodes}  # Transmission lock per node

        self.Nsucc = 0
        self.Ndrop = 0
        self.Ngen = 0
        self.Ncong = 0
        self.Nfail = 0
        self.Ngood = 0

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def initialize(self):
        for node in self.G.nodes:
            self.schedule_event(Event(0, GEN, node, None))

    def is_congested(self, node):
        return len(self.queues[node]) >= self.ct

    def is_link_available(self):
        return random.random() < LINK_AVAILABILITY

    def enqueue_packet(self, node, packet):
        self.queues[node].append(packet)

    def dequeue_packet(self, node):
        return self.queues[node].pop(0)

    def process_events(self):
        completed = 0
        while self.event_queue:
            if (self.Nsucc + self.Ndrop) >= SIMULATION_LIMIT:
                break
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            if event.event_type == GEN:
                self.handle_gen(event)
            elif event.event_type == TRANS:
                self.handle_trans(event)
            elif event.event_type == RECEIVE:
                self.handle_receive(event)

            progress = self.Nsucc + self.Ndrop
            if progress - completed >= 100:
                self.pbar.update(progress - completed)
                completed = progress

    def handle_gen(self, event):
        node = event.node
        dst = random.choice([n for n in self.G.nodes if n != node])
        try:
            path = nx.shortest_path(self.G, node, dst)
        except nx.NetworkXNoPath:
            return
        packet = Packet(node, dst, path)

        if not self.is_congested(node):
            self.enqueue_packet(node, packet)
            if not self.node_busy[node]:
                self.node_busy[node] = True
                self.schedule_event(Event(self.current_time, TRANS, node, None))
        else:
            self.Ncong += 1
            self.Ndrop += 1

        self.Ngen += 1
        interarrival = np.random.exponential(1 / self.lambda_rate)
        self.schedule_event(Event(self.current_time + interarrival, GEN, node, None))

    def handle_trans(self, event):
        node = event.node
        if not self.queues[node]:
            self.node_busy[node] = False
            return
        packet = self.dequeue_packet(node)
        path = packet.path
        hop = packet.current_hop

        if hop >= len(path) - 1:
            self.node_busy[node] = False
            return

        next_hop = path[hop + 1]

        if self.is_congested(next_hop):
            self.Ncong += 1
            if not packet.rerouted:
                packet.rerouted = True
                try:
                    new_path = nx.shortest_path(self.G, node, packet.dst)
                    if new_path != path:
                        packet.path = new_path
                        packet.current_hop = 0
                        next_hop = new_path[1]
                        if self.is_congested(next_hop):
                            self.Ncong += 1
                            self.Ndrop += 1
                            self.node_busy[node] = False
                            return
                        if not self.is_link_available():
                            self.Nfail += 1
                            self.Ndrop += 1
                            self.node_busy[node] = False
                            return
                        self.enqueue_packet(node, packet)
                        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, TRANS, node, None))
                        return
                except:
                    pass
            self.Ndrop += 1
            self.node_busy[node] = False
            return

        if not self.is_link_available():
            self.Nfail += 1
            if not packet.rerouted:
                packet.rerouted = True
                try:
                    new_path = nx.shortest_path(self.G, node, packet.dst)
                    if new_path != path:
                        packet.path = new_path
                        packet.current_hop = 0
                        next_hop = new_path[1]
                        if self.is_congested(next_hop):
                            self.Ncong += 1
                            self.Ndrop += 1
                            self.node_busy[node] = False
                            return
                        if not self.is_link_available():
                            self.Nfail += 1
                            self.Ndrop += 1
                            self.node_busy[node] = False
                            return
                        self.enqueue_packet(node, packet)
                        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, TRANS, node, None))
                        return
                except:
                    pass
            self.Ndrop += 1
            self.node_busy[node] = False
            return

        self.Ngood += 1
        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, RECEIVE, next_hop, packet))
        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, TRANS, node, None))

    def handle_receive(self, event):
        node = event.node
        packet = event.packet
        packet.current_hop += 1

        if node == packet.dst:
            self.Nsucc += 1
        elif packet.current_hop + 1 < len(packet.path):
            if not self.is_congested(node):
                self.enqueue_packet(node, packet)
                if not self.node_busy[node]:
                    self.node_busy[node] = True
                    self.schedule_event(Event(self.current_time, TRANS, node, None))
            else:
                self.Ncong += 1
                self.Ndrop += 1

    def get_results(self):
        r_all = self.Nsucc / (self.Nsucc + self.Ndrop) if (self.Nsucc + self.Ndrop) else 0
        p_cong = self.Ncong / (self.Ngood + self.Nfail + self.Ncong) if (self.Ngood + self.Nfail + self.Ncong) else 0
        return r_all, p_cong

def create_network():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=2)
    return G

def run_simulation():
    ct = int(input("Enter congestion threshold (CT): "))
    G = create_network()
    lam = 0.1
    while lam <= 1.0:
        with tqdm(total=SIMULATION_LIMIT, desc=f"Lambda {lam:.2f}") as pbar:
            sim = NetworkSimulator(G.copy(), lam, ct, pbar)
            sim.initialize()
            sim.process_events()
            r, p = sim.get_results()
        print(f"Lambda: {lam:.2f}, R_all: {r:.4f}, P_cong: {p:.4f}, Nsucc+Ndrop: {sim.Nsucc + sim.Ndrop}")
        if r < 0.1:
            print("Network is effectively deadlocked.")
            break
        lam = round(lam + 0.1, 2)

run_simulation()
