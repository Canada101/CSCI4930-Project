import networkx as nx
import numpy as np
import heapq
import random
import csv
from collections import deque

SIM_TIME = 100_000.0
LINK_AVAILABILITY = 0.9
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
        # unique packet ID, source, destination, hop count
        self.id = pid
        self.src = src
        self.dst = dst
        self.hop = hop

class FloodingSimulator:
    def __init__(self, G, lam, tt, ct, rpl):
        self.G   = G
        
        self.lam = lam
        self.tt  = tt
        self.ct  = ct
        self.rpl = rpl
        #One FIFO queue per node
        self.queues   = {n: [] for n in G.nodes()}
        self.busy     = {n: False for n in G.nodes()}
        self.recv_ids = {n: deque(maxlen=rpl) for n in G.nodes()}
        # stats
        self.Ngen  = 0
        self.Nsucc = 0
        self.Ndrop = 0
        self.Ncong = 0
        self.Nfail = 0
        self.Ngood = 0

        self.next_pid = 0
        self.evq       = []
        #hop‐limit = N−1
        self.hop_lim   = len(G.nodes()) - 1

    def schedule(self, ev):
        heapq.heappush(self.evq, ev)

    def initialize(self):
        #seed GEN=0 for every node
        for n in self.G.nodes():
            self.schedule(Event(0.0, GEN, n, None))

    def run(self):
        while self.evq:
            if self.Nsucc >= SIM_TIME:
                break
            ev = heapq.heappop(self.evq)
       
            if ev.ev_type == GEN:
                self.handle_gen(ev)
            elif ev.ev_type == TRANS:
                self.handle_trans(ev)
            else:
                self.handle_recv(ev)

    def handle_gen(self, ev):
        if self.Ngen >= SIM_TIME :
            return
        src = ev.node
        pid = self.next_pid; self.next_pid += 1
        dst = random.choice([n for n in self.G.nodes() if n != src])
        pkt = FloodPacket(pid, src, dst, hop=0)
        self.Ngen += 1
        self.queues[src].append(pkt)        #enqueue
        if not self.busy[src]:
            self.busy[src] = True
            self.schedule(Event(ev.time, TRANS, src, None))
        # schedule next GEN (Poisson)
        dt = np.random.exponential(1/self.lam)
        self.schedule(Event(ev.time+dt, GEN, src, None))

    def handle_trans(self, ev):
        node = ev.node
        if not self.queues[node]:
            self.busy[node] = False
            return
        pkt = self.queues[node].pop(0)
        #hop‐limit drop
        if pkt.hop >= self.hop_lim:
            self.Ndrop += 1
            self.busy[node] = False
            return

        self.Ngood += 1                  #count success attempt
        #flood to all neighbors
        for nbr in self.G.neighbors(node):
            #congestion
            if len(self.queues[nbr]) >= self.ct:
                self.Ncong += 1
                self.Ndrop += 1
                continue
            if random.random() > LINK_AVAILABILITY:
                self.Nfail += 1
                self.Ndrop += 1
                continue
            new_pkt = FloodPacket(pkt.id, pkt.src, pkt.dst, hop=pkt.hop)
            self.schedule(Event(ev.time+self.tt, RECEIVE, nbr, new_pkt))

        # keep sending if still queued
        if self.queues[node]:
            self.schedule(Event(ev.time+self.tt, TRANS, node, None))
        else:
            self.busy[node] = False

    def handle_recv(self, ev):
        node = ev.node
        pkt  = ev.packet
        if pkt.id in self.recv_ids[node]:
            return
        self.recv_ids[node].append(pkt.id)

        if node == pkt.dst:
            self.Nsucc += 1
        else:
            # hop‐by‐hop forwarding
            if pkt.hop < self.hop_lim:
                pkt.hop += 1
                self.queues[node].append(pkt)
                if not self.busy[node]:
                    self.busy[node] = True
                    self.schedule(Event(ev.time, TRANS, node, None))

    def get_results(self):
        r_all = self.Nsucc / self.Ngen if self.Ngen>0 else 0        # Eq 2
        denom = self.Ngood + self.Nfail + self.Ncong
        p_cong = self.Ncong / denom if denom>0 else 0               # Eq 3
        return r_all, p_cong


def run_table9():
    """Table 9: 3-node triangle, RPL=50, TT=2.0"""
    G = nx.Graph([("A","B"),("B","C"),("A","C")])
    TT  = 2.0
    RPL = 50
    experiments = [
        ([0.05],                   10),
        ([0.10,0.15,0.20,0.25],    20),
        ([0.24,0.25,0.27],         30),
    ]
    rows = []
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G, lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r,p = sim.get_results()
            rows.append({
                "h": f"{h:.2f}",
                "CT": CT,
                "P_cong": f"{p:.8f}",
                "R_all": "Deadlock" if r<1e-3 else f"{r:.4f}"
            })
    with open("table9_sim3.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        w.writeheader(); w.writerows(rows)
    print("Table 9 done → table9_sim3.csv")


def run_table10():
    """Table 10: 5-node model 1, RPL=60, TT=1.0"""
    G = nx.Graph([
        ("A","B"),("B","C"),("C","D"),
        ("D","E"),("E","A"),("B","D")
    ])
    TT  = 1.0
    RPL = 60
    experiments = [
        ([0.01,0.03,0.05,0.07], 60),
        ([0.07,0.08,0.09,0.10,0.12,0.13], 80),
    ]
    rows=[]
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G, lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r,p = sim.get_results()
            rows.append({
                "h": f"{h:.2f}",
                "CT": CT,
                "P_cong": f"{p:.8f}",
                "R_all": "Deadlock" if r<1e-3 else f"{r:.4f}"
            })
    with open("table10_sim3.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        w.writeheader(); w.writerows(rows)
    print("Table 10 done → table10_sim3.csv")


def run_table11():
    """Table 11: 5-node model 2, RPL=60, TT=1.0"""
    # same 5-node topology
    G = nx.Graph([
        ("A","B"),("B","C"),("C","D"),
        ("D","E"),("E","A"),("B","D")
    ])
    TT  = 1.0
    RPL = 60
    # h-values from paper’s Table 11
    experiments = [
        ([0.02,0.05,0.07,0.10,0.12,0.13,0.14], 60),
    ]
    rows=[]
    for h_vals, CT in experiments:
        for h in h_vals:
            sim = FloodingSimulator(G, lam=h, tt=TT, ct=CT, rpl=RPL)
            sim.initialize(); sim.run()
            r,p = sim.get_results()
            rows.append({
                "h": f"{h:.2f}",
                "CT": CT,
                "P_cong": f"{p:.8f}",
                "R_all": "Deadlock" if r<1e-3 else f"{r:.4f}"
            })
    with open("table11_sim3.csv","w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h","CT","P_cong","R_all"])
        w.writeheader(); w.writerows(rows)
    print("Table 11 done → table11_sim3.csv")


if __name__ == "__main__":
    run_table9()
    run_table10()
    run_table11()
