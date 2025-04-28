
# simulation 2 single queue version with custom input
import networkx as nx
import numpy as np
import heapq
import random
import csv
import time

# simulation parameters tt and ct will be asked from user
SIMULATION_LIMIT_NGEN = 1_000_000 # target generated packets
LINK_AVAILABILITY = 0.9

GEN, TRANS, RECEIVE = 'GEN', 'TRANS', 'RECEIVE' # event types

class Event:
    # represents an event in the simulation future event list fel
    def __init__(self, time, event_type, node, packet):
        self.time = time
        self.event_type = event_type
        self.node = node
        self.packet = packet

    def __lt__(self, other): # comparison for heapq sorts by time
        if self.time != other.time:
            return self.time < other.time
        event_order = {GEN: 0, TRANS: 1, RECEIVE: 2} # tie breaking optional
        return event_order.get(self.event_type, 99) < event_order.get(other.event_type, 99)

class Packet:
    # represents a data packet
    def __init__(self, packet_id, src, dst, path, creation_time):
        self.id = packet_id
        self.src = src
        self.dst = dst
        self.path = path # pre calculated path
        self.current_hop = 0
        self.creation_time = creation_time

class NetworkSimulator:
    # manages the simulation logic and state
    def __init__(self, G, lam, tt, ct):
        self.G = G
        self.lambda_rate = lam
        self.tt = tt
        self.ct = ct
        self.event_queue = [] # fel
        self.current_time = 0
        self.packet_counter = 0

        # queuing model single fifo queue per node
        self.queues = {node: [] for node in G.nodes}
        self.node_busy = {node: False for node in G.nodes} # transmission lock

        # statistics counters
        self.Nsucc, self.Ngen, self.Ncong, self.Nfail, self.Ngood = 0, 0, 0, 0, 0

        # deadlock stall detection parameters
        self.max_simulation_time_limit = 0 # safeguard
        self.last_progress_time = 0
        self.no_progress_threshold = tt * 1000 # how long to wait for progress before declaring stall

    def schedule_event(self, event):
        heapq.heappush(self.event_queue, event)

    def initialize(self):
        # sets up initial generation events estimates max runtime
        num_nodes = len(self.G.nodes)
        if num_nodes > 0 and self.lambda_rate > 0:
            estimated_avg_time = (SIMULATION_LIMIT_NGEN / (num_nodes * self.lambda_rate))
            self.max_simulation_time_limit = estimated_avg_time * 50
        else:
            self.max_simulation_time_limit = SIMULATION_LIMIT_NGEN * self.tt * 20

        self.last_progress_time = 0
        for node in self.G.nodes:
            # exponential arrivals based on lambda
            initial_delay = np.random.exponential(1 / self.lambda_rate) if self.lambda_rate > 0 else float('inf')
            self.schedule_event(Event(initial_delay, GEN, node, None))

    def is_congested(self, node):
        return len(self.queues.get(node, [])) >= self.ct

    def is_link_available(self, u, v):
        if not self.G.has_edge(u, v): return False
        return random.random() < LINK_AVAILABILITY

    def enqueue_packet(self, node, packet):
        # adds packet schedules trans if node was idle
        if node not in self.queues: return
        self.queues[node].append(packet)
        if not self.node_busy[node]:
            self.node_busy[node] = True
            self.schedule_event(Event(self.current_time, TRANS, node, None))

    def dequeue_packet(self, node):
        # fifo dequeue
        if node not in self.queues or not self.queues[node]: return None
        return self.queues[node].pop(0)

    def process_events(self):
        # runs the main simulation loop
        simulation_start_wall_time = time.time()
        processed_events = 0
        status = "OK"

        while self.event_queue and self.Ngen < SIMULATION_LIMIT_NGEN:
            # deadlock stall checks
            if self.max_simulation_time_limit > 0 and self.current_time > self.max_simulation_time_limit:
                status = "timelimitexceeded"; print(f"\nwarning simulation time exceeded limit {self.max_simulation_time_limit:.2f} stopping"); break
            if self.current_time > self.last_progress_time + self.no_progress_threshold:
                status = "stalled"; print(f"\nwarning no successful delivery for {self.no_progress_threshold:.2f} time units stopping potential deadlock stall"); break

            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            processed_events += 1

            # event handling
            if event.event_type == GEN:
                self.handle_gen(event)
            elif event.event_type == TRANS:
                self.handle_trans(event)
            elif event.event_type == RECEIVE:
                success = self.handle_receive(event)
                if success: self.last_progress_time = self.current_time

        # simulation end reporting
        simulation_end_wall_time = time.time()
        duration = simulation_end_wall_time - simulation_start_wall_time
        final_ngen = min(self.Ngen, SIMULATION_LIMIT_NGEN)
        print(f"loop finished status {status} time {self.current_time:.2f} ngen {final_ngen} nsucc {self.Nsucc} events {processed_events} duration {duration:.2f}s")
        if not self.event_queue and self.Ngen < SIMULATION_LIMIT_NGEN and status=="OK":
            status = "queueempty"; print("simulation ended prematurely event queue empty")

        return status

    def handle_gen(self, event):
        if self.Ngen >= SIMULATION_LIMIT_NGEN: return

        node = event.node
        possible_destinations = [n for n in self.G.nodes if n != node]
        if not possible_destinations: return

        dst = random.choice(possible_destinations)
        try: # use shortest path routing
            path = nx.shortest_path(self.G, source=node, target=dst)
        except nx.NetworkXNoPath: path = None

        if path:
            self.packet_counter += 1
            packet_id = f"{node}-{dst}-{self.packet_counter}"
            packet = Packet(packet_id, node, dst, path, self.current_time)
            self.Ngen += 1
            self.enqueue_packet(node, packet)

        # schedule next generation for this node
        interarrival = np.random.exponential(1 / self.lambda_rate) if self.lambda_rate > 0 else float('inf')
        self.schedule_event(Event(self.current_time + interarrival, GEN, node, None))

    def handle_trans(self, event):
        # handles transmission attempt
        node = event.node
        if not self.queues.get(node):
            self.node_busy[node] = False; return

        packet = self.queues[node][0] # peek

        # validity check
        if not hasattr(packet, 'path') or not packet.path or packet.current_hop >= len(packet.path) - 1:
            self.dequeue_packet(node)
            if self.queues.get(node): self.schedule_event(Event(self.current_time, TRANS, node, None))
            else: self.node_busy[node] = False
            return

        next_hop_node = packet.path[packet.current_hop + 1]

        # check link availability
        if not self.is_link_available(node, next_hop_node):
            self.Nfail += 1
            # model 2 retry reschedule trans for this node after delay packet stays
            self.schedule_event(Event(self.current_time + self.tt, TRANS, node, None)); return

        # check next hop congestion
        if self.is_congested(next_hop_node):
            self.Ncong += 1
            # model 2 retry reschedule trans for this node after delay packet stays
            self.schedule_event(Event(self.current_time + self.tt, TRANS, node, None)); return

        # transmission ok
        self.Ngood += 1
        packet_to_send = self.dequeue_packet(node)

        if packet_to_send: # schedule arrival
            self.schedule_event(Event(self.current_time + self.tt, RECEIVE, next_hop_node, packet_to_send))

        # check queue again for next transmission
        if self.queues.get(node): self.schedule_event(Event(self.current_time, TRANS, node, None))
        else: self.node_busy[node] = False

    def handle_receive(self, event):
        # handles packet arrival
        node = event.node
        packet = event.packet
        if packet is None: return False

        success = False
        if node == packet.dst: # destination reached
            self.Nsucc += 1; success = True
        else: # intermediate node
            packet.current_hop += 1
            if packet.current_hop < len(packet.path) -1 :
                 self.enqueue_packet(node, packet) # forward
            # else path ended packet lost silently

        return success # indicate if final destination was reached

    def get_results(self):
        # calculates final metrics
        actual_ngen = min(self.Ngen, SIMULATION_LIMIT_NGEN)
        # r_all successes generations model 2 definition
        r_all = self.Nsucc / actual_ngen if actual_ngen > 0 else 0
        # p_cong congestion failures total attempts
        total_link_attempts = self.Ngood + self.Nfail + self.Ncong
        p_cong = self.Ncong / total_link_attempts if total_link_attempts > 0 else 0
        return r_all, p_cong, actual_ngen, self.Nsucc

def create_network():
    # creates 3 node graph fig 1a
    G = nx.Graph(); G.add_edge(0, 1); G.add_edge(1, 2); G.add_edge(0, 2); return G

def run_simulation_suite():
    # get user input and run simulations

    # get user input
    while True:
        try:
            tt_str = input("enter transmission time tt e g 2 or 0 0002 ")
            tt_param = float(tt_str)
            if tt_param <= 0: raise ValueError("tt must be positive")

            ct_str = input("enter congestion threshold ct e g 20 or 50 ")
            ct_param = int(ct_str)
            if ct_param <= 0: raise ValueError("ct must be positive")

            break
        except ValueError as e:
            print(f"invalid input {e} please try again")
        except EOFError:
            print("\ninput stream closed aborting")
            return

    # set up based on input
    output_csv_file = f"simulation2_SINGLE_QUEUE_results_TT{tt_param}_CT{ct_param}_Ngen{SIMULATION_LIMIT_NGEN}.csv"
    G = create_network()
    # select lambda range based on tt heuristic
    if tt_param > 0.1: lambda_values = np.arange(0.1, 0.9, 0.1)
    else: lambda_values = np.arange(1000, 9000, 2000)

    results_data = []

    print(f"running simulation suite model 2 single queue")
    print(f"parameters tt {tt_param} ct {ct_param} link avail {LINK_AVAILABILITY}")
    print(f"target ngen {SIMULATION_LIMIT_NGEN} stall threshold {tt_param * 1000:.2f}")
    print("-" * 40)

    # run simulation for each lambda
    for lam in lambda_values:
        lam = round(lam, 4)
        print(f"running lambda = {lam}...")
        # create simulator with user defined tt ct
        sim = NetworkSimulator(G.copy(), lam, tt_param, ct_param)
        sim.initialize()
        final_status = sim.process_events() # run sim get status
        r, p, ngen_final, nsucc_final = sim.get_results()

        # format results row marking deadlock
        if final_status != "OK":
            results_row = [lam, ct_param, tt_param, "deadlock", "deadlock", ngen_final, nsucc_final, final_status]
            print(f"lambda {lam:.4f} status {final_status} ngen {ngen_final} nsucc {nsucc_final}")
        else:
            results_row = [lam, ct_param, tt_param, f"{r:.4f}", f"{p:.4f}", ngen_final, nsucc_final, final_status]
            print(f"lambda {lam:.4f} r_all {r:.4f} p_cong {p:.4f} ngen {ngen_final} nsucc {nsucc_final}")

        results_data.append(results_row)

        # stop suite if deadlock occurs
        if final_status != "OK":
            print(f"stopping suite after status '{final_status}' at lambda = {lam}")
            break

    # write results to csv
    try:
        with open(output_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['lambda', 'ct', 'tt', 'r_all', 'p_cong', 'ngen_completed', 'nsucc_completed', 'status']) # header
            writer.writerows(results_data) # data rows
        print(f"\nsimulation results saved to {output_csv_file}")
    except IOError as e:
        print(f"\nerror writing results to csv file {e}")

# main execution starts here
if __name__ == "__main__":
    run_simulation_suite()
