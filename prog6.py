import networkx as nx
import numpy as np
import heapq
import random
import csv
import os
from tqdm import tqdm

# Simulation 4 Constants
SIMULATION_LIMIT = 1_000_000
GEN, TRANS, RECEIVE, RETRY = 'GEN', 'TRANS', 'RECEIVE', 'RETRY'
LINK_AVAILABILITY = 0.9

class Event:
    def __init__(self, time, event_type, node, packet=None):
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
        self.retry_count = 0

class NetworkSimulator:
    def __init__(self, G, lam, tt, ct, pbar):
        self.G = G
        self.lambda_rate = lam
        self.tt = tt
        self.ct = ct
        self.pbar = pbar

        self.event_queue = []
        self.current_time = 0

        self.queues = {node: [] for node in G.nodes}
        self.node_busy = {node: False for node in G.nodes}

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
            self.schedule_event(Event(0, GEN, node))

    def is_congested(self, node):
        return len(self.queues[node]) >= self.ct

    def is_link_available(self):
        return random.random() < LINK_AVAILABILITY

    def enqueue_packet(self, node, packet):
        self.queues[node].append(packet)

    def dequeue_packet(self, node):
        if self.queues[node]:
            return self.queues[node].pop(0)
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
            elif event.event_type == RETRY:
                self.handle_retry(event)
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
        self.Ngen += 1
        if not self.is_congested(node):
            self.enqueue_packet(node, packet)
            if not self.node_busy[node]:
                self.node_busy[node] = True
                self.schedule_event(Event(self.current_time, TRANS, node))
        else:
            self.Ncong += 1
            self.Ndrop += 1

        interarrival = np.random.exponential(1 / self.lambda_rate)
        self.schedule_event(Event(self.current_time + interarrival, GEN, node))

    def handle_trans(self, event):
        node = event.node
        packet = self.dequeue_packet(node)
        if packet is None:
            self.node_busy[node] = False
            return

        path = packet.path
        hop = packet.current_hop
        if hop >= len(path) - 1:
            self.node_busy[node] = False
            return

        next_hop = path[hop + 1]

        if self.is_congested(next_hop):
            self.Ncong += 1
            if packet.retry_count == 0:
                packet.retry_count += 1
                self.schedule_event(Event(self.current_time + self.tt, RETRY, node, packet))
            else:
                self.Ndrop += 1
            self.node_busy[node] = False
            return

        if not self.is_link_available():
            self.Nfail += 1
            if packet.retry_count == 0:
                packet.retry_count += 1
                self.schedule_event(Event(self.current_time + self.tt, RETRY, node, packet))
            else:
                self.Ndrop += 1
            self.node_busy[node] = False
            return

        self.Ngood += 1
        self.schedule_event(Event(self.current_time + self.tt, RECEIVE, next_hop, packet))
        self.schedule_event(Event(self.current_time + self.tt, TRANS, node))

    def handle_receive(self, event):
        node = event.node
        packet = event.packet
        packet.current_hop += 1
        if node == packet.dst:
            self.Nsucc += 1
        else:
            if not self.is_congested(node):
                self.enqueue_packet(node, packet)
                if not self.node_busy[node]:
                    self.node_busy[node] = True
                    self.schedule_event(Event(self.current_time, TRANS, node))
            else:
                self.Ncong += 1
                self.Ndrop += 1

    def handle_retry(self, event):
        node = event.node
        packet = event.packet
        if not self.is_congested(node):
            self.enqueue_packet(node, packet)
            if not self.node_busy[node]:
                self.node_busy[node] = True
                self.schedule_event(Event(self.current_time, TRANS, node))
        else:
            self.Ndrop += 1

    def get_results(self):
        total = self.Ngen + self.Ndrop
        r_all = self.Nsucc / total if total > 0 else 0
        p_cong = self.Ncong / (self.Ngood + self.Nfail + self.Ncong) if (self.Ngood + self.Nfail + self.Ncong) > 0 else 0
        return r_all, p_cong

def create_network(topology_choice):
    G = nx.Graph()
    if topology_choice == 1:
        G.add_edge("A", "B", weight=1)
        G.add_edge("A", "C", weight=1)
        G.add_edge("A", "D", weight=1)
        G.add_edge("B", "C", weight=1)
        G.add_edge("C", "E", weight=1)
        G.add_edge("D", "E", weight=1)
    else:
        G.add_edge("A", "B", weight=1)
        G.add_edge("B", "C", weight=1)
        G.add_edge("C", "D", weight=1)
        G.add_edge("D", "E", weight=1)
        G.add_edge("E", "A", weight=1)
    return G

def run_simulation_4():
    tt = float(input("Enter transmission time (TT): "))
    ct = int(input("Enter congestion threshold (CT): "))
    start_lambda = float(input("Enter starting lambda value: "))
    end_lambda = float(input("Enter ending lambda value: "))
    step_lambda = float(input("Enter lambda increment: "))
    topology_choice = int(input("Enter 1 for Topology 1 or 2 for Topology 2: "))

    G = create_network(topology_choice)
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

def run_all_simulations(runs=2):
    filename = "Simulation_4_Results.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as csvfile:
        fieldnames = ["Run", "Lambda", "R_all", "P_cong", "Packets"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for run in range(1, runs + 1):
            print(f"\n--- Starting Simulation 4 (5-Node Model, Static Shortest Path Routing), Table {run}")
            results = run_simulation_4()
            for row in results:
                row["Run"] = run
                writer.writerow(row)
    print(f"Simulation 4 done. Results saved to {filename}")

if __name__ == "__main__":
    run_all_simulations()
