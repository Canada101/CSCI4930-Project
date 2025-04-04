import networkx as nx
import numpy as np
import heapq
import random
from tqdm import tqdm

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
        self.congestion_threshold = ct
        self.pbar = pbar

        self.event_queue = []
        self.current_time = 0

        self.queues = {node: {} for node in G.nodes()}
        self.Nsucc = 0
        self.Ndrop = 0
        self.Ngen = 0
        self.Ncong = 0
        self.Nfail = 0
        self.Ngood = 0

        self.max_gen = SIMULATION_LIMIT

    def initialize(self):
        for node in self.G.nodes:
            self.schedule_event(Event(0, GEN, node, None))

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def is_link_available(self):
        return random.random() < LINK_AVAILABILITY

    def is_congested(self, node, dst):
        return len(self.queues[node].get(dst, [])) >= self.congestion_threshold

    def enqueue_packet(self, node, dst, packet):
        self.queues[node].setdefault(dst, []).append(packet)

    def dequeue_packet(self, node, dst):
        if self.queues[node].get(dst):
            return self.queues[node][dst].pop(0)
        return None

    def process_events(self):
        last_progress = 0
        while self.event_queue:
            if self.Ngen >= self.max_gen and (self.Nsucc + self.Ndrop) >= self.Ngen:
                break

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if event.event_type == GEN:
                if self.Ngen < self.max_gen:
                    self.handle_gen(event)
            elif event.event_type == TRANS:
                self.handle_trans(event)
            elif event.event_type == RECEIVE:
                self.handle_receive(event)

            completed = self.Nsucc + self.Ndrop
            if (completed - last_progress) >= 100:
                self.pbar.update(completed - last_progress)
                last_progress = completed

    def handle_gen(self, event):
        src = event.node
        dst = random.choice([n for n in self.G.nodes if n != src])

        try:
            path = nx.shortest_path(self.G, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            return

        packet = Packet(src, dst, path)
        self.enqueue_packet(src, dst, packet)
        self.schedule_event(Event(self.current_time, TRANS, src, packet))

        inter_arrival = np.random.exponential(1 / self.lambda_rate)
        self.schedule_event(Event(self.current_time + inter_arrival, GEN, src, None))
        self.Ngen += 1

    def handle_trans(self, event):
        node = event.node
        packet = event.packet
        dst = packet.dst

        path = packet.path
        hop = packet.current_hop

        if hop >= len(path) - 1:
            return

        u, v = path[hop], path[hop + 1]

        # Remove packet from the queue when transmitting
        self.dequeue_packet(node, dst)

        if self.is_congested(node, dst):
            self.Ncong += 1
            if not packet.rerouted:
                try:
                    alt_path = nx.shortest_path(self.G, node, dst, weight='weight')
                    if alt_path != packet.path:
                        packet.path = alt_path
                        packet.current_hop = 0
                        packet.rerouted = True
                        self.enqueue_packet(node, dst, packet)
                        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, TRANS, node, packet))
                        return
                except nx.NetworkXNoPath:
                    pass
            self.Ndrop += 1
            return

        if not self.is_link_available():
            self.Nfail += 1
            if not packet.rerouted:
                try:
                    alt_path = nx.shortest_path(self.G, node, dst, weight='weight')
                    if alt_path != packet.path:
                        packet.path = alt_path
                        packet.current_hop = 0
                        packet.rerouted = True
                        self.enqueue_packet(node, dst, packet)
                        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, TRANS, node, packet))
                        return
                except nx.NetworkXNoPath:
                    pass
            self.Ndrop += 1
            return

        self.Ngood += 1
        self.schedule_event(Event(self.current_time + TRANSMISSION_TIME, RECEIVE, v, packet))

    def handle_receive(self, event):
        node = event.node
        packet = event.packet

        packet.current_hop += 1

        if node == packet.dst:
            self.Nsucc += 1
        else:
            self.enqueue_packet(node, packet.dst, packet)
            self.schedule_event(Event(self.current_time, TRANS, node, packet))

    def get_results(self):
        r_all = self.Nsucc / (self.Nsucc + self.Ndrop) if (self.Nsucc + self.Ndrop) > 0 else 0
        p_cong = self.Ncong / (self.Ncong + self.Ngood + self.Nfail) if (self.Ncong + self.Ngood + self.Nfail) > 0 else 0
        return r_all, p_cong

def create_network():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=2)
    return G

def run_multiple_lambdas():
    ct = int(input("Enter congestion threshold (CT): "))
    G_base = create_network()

    lam = 0.1
    while lam <= 1.0:
        G = G_base.copy()
        with tqdm(total=SIMULATION_LIMIT, desc=f"Lambda {lam:.2f}") as pbar:
            sim = NetworkSimulator(G, lam, ct, pbar)
            sim.initialize()
            sim.process_events()
            r_all, p_cong = sim.get_results()

        print(f"Lambda: {lam:.2f}, R_all: {r_all:.4f}, P_cong: {p_cong:.4f}")

        if r_all < 0.1:
            print("Network is effectively deadlocked.")
            break

        lam = round(lam + 0.1, 2)

run_multiple_lambdas()
