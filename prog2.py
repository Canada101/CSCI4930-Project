import networkx as nx
import numpy as np
import heapq
import random
from tqdm import tqdm
import csv
import os


#Simulation 1 (3-Node Model with Packet Dropping) 
SIMULATION_LIMIT = 1_000_000
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
        self.current_hop = 0   # Index in the path list
        self.rerouted = False

class NetworkSimulator:
    def __init__(self, G, lam, tt, ct, pbar):
        self.G = G
        self.lambda_rate = lam
        self.tt = tt        
        self.ct = ct         
        self.pbar = pbar

        # Each node now has a dictionary of queues keyed by destination.
        self.queues = {node: {} for node in G.nodes}
        self.node_busy = {node: False for node in G.nodes}

        self.Nsucc = 0
        self.Ndrop = 0
        self.Ngen = 0
        self.Ncong = 0
        self.Nfail = 0
        self.Ngood = 0

        self.event_queue = []
        self.current_time = 0

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def initialize(self):
        for node in self.G.nodes:
            self.schedule_event(Event(0, GEN, node, None))

    def is_congested(self, node, dst):
        return len(self.queues[node].get(dst, [])) >= self.ct

    def is_link_available(self):
        return random.random() < LINK_AVAILABILITY

    def enqueue_packet(self, node, dst, packet):
        self.queues[node].setdefault(dst, []).append(packet)

    def dequeue_packet(self, node, dst):
        q = self.queues[node].get(dst, [])
        if q:
            return q.pop(0)
        return None

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
        if not self.is_congested(node, dst):
            self.enqueue_packet(node, dst, packet)
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
        # Try to get a packet from any destination queue
        packet = None
        for dst, q in self.queues[node].items():
            if q:
                packet = q.pop(0)
                break
        if packet is None:
            self.node_busy[node] = False
            return

        path = packet.path
        hop = packet.current_hop
        if hop >= len(path) - 1:
            self.node_busy[node] = False
            return

        next_hop = path[hop + 1]

        # Check congestion at next hop
        if self.is_congested(next_hop, packet.dst):
            if packet.current_hop == 0 and not packet.rerouted:
                try:
                    new_path = nx.shortest_path(self.G, node, packet.dst)
                    if new_path != path:
                        packet.path = new_path
                        packet.current_hop = 0
                        packet.rerouted = True
                        self.enqueue_packet(node, packet.dst, packet)
                        self.schedule_event(Event(self.current_time + self.tt, TRANS, node, None))
                        return
                except nx.NetworkXNoPath:
                    pass
            self.Ncong += 1
            self.Ndrop += 1
            self.node_busy[node] = False
            return

        # Check link availability
        if not self.is_link_available():
            if packet.current_hop == 0 and not packet.rerouted:
                try:
                    new_path = nx.shortest_path(self.G, node, packet.dst)
                    if new_path != path:
                        packet.path = new_path
                        packet.current_hop = 0
                        packet.rerouted = True
                        self.enqueue_packet(node, packet.dst, packet)
                        self.schedule_event(Event(self.current_time + self.tt, TRANS, node, None))
                        return
                except nx.NetworkXNoPath:
                    pass
            self.Nfail += 1
            self.Ndrop += 1
            self.node_busy[node] = False
            return

        self.Ngood += 1
        self.schedule_event(Event(self.current_time + self.tt, RECEIVE, next_hop, packet))
        self.schedule_event(Event(self.current_time + self.tt, TRANS, node, None))

    def handle_receive(self, event):
        node = event.node
        packet = event.packet
        packet.current_hop += 1
        if node == packet.dst:
            self.Nsucc += 1
        else:
            if not self.is_congested(node, packet.dst):
                self.enqueue_packet(node, packet.dst, packet)
                if not self.node_busy[node]:
                    self.node_busy[node] = True
                    self.schedule_event(Event(self.current_time, TRANS, node, None))
            else:
                self.Ncong += 1
                self.Ndrop += 1

    def get_results(self):
        total = self.Nsucc + self.Ndrop
        r_all = self.Nsucc / total if total > 0 else 0
        p_cong = self.Ncong / (self.Ngood + self.Nfail + self.Ncong) if (self.Ngood + self.Nfail + self.Ncong) > 0 else 0
        return r_all, p_cong

def create_network():
    G = nx.Graph()
    G.add_edge("A", "B", weight=1)
    G.add_edge("B", "C", weight=1)
    G.add_edge("A", "C", weight=2)
    return G

def run_simulation_and_return():
    # Get inputs for one simulation run.
    tt = float(input("Enter transmission time (TT): "))
    ct = int(input("Enter congestion threshold (CT): "))
    start_lambda = float(input("Enter starting lambda value: "))
    end_lambda = float(input("Enter ending lambda value: "))
    step_lambda = float(input("Enter lambda increment: "))

    G = create_network()
    lam = start_lambda
    results = []
    while lam <= end_lambda:
        with tqdm(total=SIMULATION_LIMIT, desc=f"Lambda {lam:.2f}") as pbar:
            sim = NetworkSimulator(G.copy(), lam, tt, ct, pbar)
            sim.initialize()
            sim.process_events()
            r, p = sim.get_results()
        total_packets = sim.Nsucc + sim.Ndrop
        print(f"Lambda: {lam:.2f}, R_all: {r:.4f}, P_cong: {p:.4f}, Packets: {total_packets}")
        results.append({"Lambda": lam, "R_all": r, "P_cong": p, "Packets": total_packets})
        if r < 0.1:
            print("Network is effectively deadlocked.")
            break
        lam = round(lam + step_lambda, 10)
    return results

def run_all_simulations(runs=4):
    filename = "Simulation 1 Results.csv"
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a", newline="") as csvfile:
        fieldnames = ["Run", "Lambda", "R_all", "P_cong", "Packets"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for run in range(1, runs + 1):
            print(f"\n--- Starting Simulation 1 (3-Node Model with Packet Dropping), \n Table {run}")
            results = run_simulation_and_return()
            for row in results:
                row["Run"] = run
                writer.writerow(row)
    print(f"Simulation 1 done. Results saved to {filename}")

run_all_simulations()
