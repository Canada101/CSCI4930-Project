#!/usr/bin/env python3
import networkx as nx
import numpy as np
import heapq
import random
import csv
from collections import deque
from tqdm import tqdm

# 11. Termination by simulation time (10^5 time units)
SIM_TIME = 100_000.0

# 6. Link availability probability
LINK_AVAILABILITY = 0.9

# 4. Event types
GEN, TRANS, RECEIVE = 'GEN', 'TRANS', 'RECEIVE'


class Event:
    def __init__(self, time, ev_type, node, packet):
        self.time = time
        self.ev_type = ev_type
        self.node = node
        self.packet = packet
    def __lt__(self, other):
        return self.time < other.time


class FloodPacket:
    def __init__(self, pid, src, dst, hop=0):
        # unique packet ID, source, destination, hop counter
        self.id = pid
        self.src = src
        self.dst = dst
        self.hop = hop


class FloodingSimulator:
    def __init__(self, G, lam, tt, ct, rpl):
        # 1. Network as a graph
        self.G = G
        # 2. Poisson packet generation rate λ
        self.lam = lam
        # transmission time per hop
        self.tt = tt
        # 7. Congestion threshold
        self.ct = ct
        # 10. Received-packet-limit (duplicate suppression)
        self.rpl = rpl

        # 3. Single FIFO queue per node
        self.queues = {n: [] for n in G.nodes()}
        self.node_busy = {n: False for n in G.nodes()}

        # 10. Duplicate suppression buffers
        self.received_ids = {n: deque(maxlen=rpl) for n in G.nodes()}

        # Statistics
        self.Ngen  = 0    # generated
        self.Nsucc = 0    # delivered
        self.Ndrop = 0    # dropped copies
        self.Ncong = 0    # congestion drops
        self.Nfail = 0    # link-failure drops
        self.Ngood = 0    # successful link transmissions

        self.next_pid = 0
        self.ev_queue  = []
        # 9. Hop-limit = N - 1
        self.hop_limit = len(G.nodes()) - 1

    def schedule(self, ev):
        # 4. Schedule an event
        heapq.heappush(self.ev_queue, ev)

    def initialize(self):
        # 4. Seed initial GEN events at t=0 for each node
        for n in self.G.nodes():
            self.schedule(Event(0.0, GEN, n, None))

    def run(self):
        # 11. Terminate by simulation time
        while self.ev_queue:
            ev = heapq.heappop(self.ev_queue)
            if ev.time > SIM_TIME:
                break
            if ev.ev_type == GEN:
                self.handle_gen(ev)
            elif ev.ev_type == TRANS:
                self.handle_trans(ev)
            else:
                self.handle_recv(ev)

    def handle_gen(self, ev):
        src = ev.node
        # 2. Poisson packet generation
        pid = self.next_pid; self.next_pid += 1
        dst = random.choice([n for n in self.G.nodes() if n != src])
        pkt = FloodPacket(pid, src, dst, hop=0)

        self.Ngen += 1
        self.queues[src].append(pkt)  # 3. enqueue packet
        if not self.node_busy[src]:
            self.node_busy[src] = True
            self.schedule(Event(ev.time, TRANS, src, None))

        # schedule next GEN
        dt = np.random.exponential(1.0 / self.lam)
        self.schedule(Event(ev.time + dt, GEN, src, None))

    def handle_trans(self, ev):
        node = ev.node
        if not self.queues[node]:
            self.node_busy[node] = False
            return

        pkt = self.queues[node].pop(0)
        # 9. Hop-limit drop
        if pkt.hop >= self.hop_limit:
            self.Ndrop += 1
            self.node_busy[node] = False
            return

        self.Ngood += 1  # 8. successful link transmission
        next_hop = pkt.hop + 1

        # 5. Flood to all neighbors
        for nbr in self.G.neighbors(node):
            # 7. congestion check
            if len(self.queues[nbr]) >= self.ct:
                self.Ncong += 1
                self.Ndrop += 1
                continue
            # 6. link availability check
            if random.random() > LINK_AVAILABILITY:
                self.Nfail += 1
                self.Ndrop += 1
                continue
            # enqueue at neighbor
            new_pkt = FloodPacket(pkt.id, pkt.src, pkt.dst, hop=next_hop)
            self.schedule(Event(ev.time + self.tt, RECEIVE, nbr, new_pkt))

        # schedule next TRANS if still packets
        if self.queues[node]:
            self.schedule(Event(ev.time + self.tt, TRANS, node, None))
        else:
            self.node_busy[node] = False

    def handle_recv(self, ev):
        node = ev.node
        pkt = ev.packet
        # 10. Duplicate suppression
        if pkt.id in self.received_ids[node]:
            return
        self.received_ids[node].append(pkt.id)
        # count success only at destination
        if node == pkt.dst:
            self.Nsucc += 1
        else:
            # continue flooding hop-by-hop
            if pkt.hop < self.hop_limit:
                pkt.hop += 1
                self.queues[node].append(pkt)
                if not self.node_busy[node]:
                    self.node_busy[node] = True
                    self.schedule(Event(ev.time, TRANS, node, None))

    def get_results(self):
        # 12a. R_all = Nsucc / Ngen
        r_all = self.Nsucc / self.Ngen if self.Ngen > 0 else 0
        # 12b. P_cong = Ncong / (Ngood + Nfail + Ncong)
        denom = self.Ngood + self.Nfail + self.Ncong
        p_cong = self.Ncong / denom if denom > 0 else 0
        return r_all, p_cong


def run_3node():
    # Table 9 parameters
    G = nx.Graph([("A","B"),("B","C"),("A","C")])
    TT  = 2.0
    RPL = 50
    experiments = [
        ([0.05],                 10),
        ([0.10, 0.15, 0.20, 0.25], 20),
        ([0.24, 0.25, 0.27],       30),
    ]
    results = []
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G.copy(), lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r, p = sim.get_results()
            R_str = "Deadlock" if r < 1e-3 else f"{r:.4f}"
            results.append({"h":f"{h:.2f}", "CT":CT, "P_cong":f"{p:.4f}", "R_all":R_str})
    with open("table9_3node.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        writer.writeheader(); writer.writerows(results)
    print("Table 9 (3-node) results → table9_3node.csv")


def run_5node_model1():
    # Table 10 parameters
    G = nx.Graph([
        ("A","B"),("B","C"),("C","D"),("D","E"),
        ("E","A"),("B","D")
    ])
    TT  = 1.0
    RPL = 60
    experiments = [
        ([0.01,0.03,0.05,0.07], 60),
        ([0.07,0.08,0.09,0.10,0.12,0.13], 80),
    ]
    results = []
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G.copy(), lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r, p = sim.get_results()
            R_str = "Deadlock" if r < 1e-3 else f"{r:.4f}"
            results.append({"h":f"{h:.2f}", "CT":CT, "P_cong":f"{p:.4f}", "R_all":R_str})
    with open("table10_5node1.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        writer.writeheader(); writer.writerows(results)
    print("Table 10 (5-node model 1) → table10_5node1.csv")


def run_5node_model2():
    # Table 11 parameters
    G = nx.Graph([
        ("A","B"),("B","C"),("C","D"),("D","E"),
        ("E","A"),("B","D")
    ])
    TT  = 1.0
    RPL = 60
    experiments = [
        ([0.02,0.05,0.07,0.10,0.12,0.13], 60),
    ]
    results = []
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G.copy(), lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r, p = sim.get_results()
            R_str = "Deadlock" if r < 1e-3 else f"{r:.4f}"
            results.append({"h":f"{h:.2f}", "CT":CT, "P_cong":f"{p:.4f}", "R_all":R_str})
    with open("table11_5node2.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        writer.writeheader(); writer.writerows(results)
    print("Table 11 (5-node model 2) → table11_5node2.csv")


if __name__ == "__main__":
    run_3node()
    run_5node_model1()
    run_5node_model2()
