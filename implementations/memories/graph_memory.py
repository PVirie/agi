import numpy as np

try:
    from interfaces.memory import Graph_Memory, Graph_Memory_Operation_Type
except ImportError:
    # Define dummy classes for testing without the full interface
    from enum import Flag, auto

    class Graph_Memory_Operation_Type(Flag):
        IDLE = 0
        RESET = auto()
        CREATE = auto()
        WRITE = auto()
        MOVE = auto()
        LINK = auto()
        ROTATE = auto()

    class Graph_Memory:
        pass


class NP_Graph_Memory(Graph_Memory):

    def __init__(self, num_batches, num_nodes, max_edges_per_node, node_dim):

        self.C = 1 + max_edges_per_node

        self.nodes = np.zeros((num_batches, num_nodes, node_dim), dtype=np.float32)
        self.edges = np.full((num_batches, num_nodes, max_edges_per_node), -1, dtype=np.int32) # store the source node index for each edge slot, -1 means no edge

        self.next_free_node = np.ones((num_batches,), dtype=np.int32)  # track the next free node index for each batch
        self.next_free_edge = np.zeros((num_batches, num_nodes), dtype=np.int32)  # track the next free edge index for each node in each batch

        self.head = np.zeros((num_batches,), dtype=np.int32)  # track the head node index for each batch

        # update trace for training
        self.timestep = 0
        self.edge_cause_time = np.full((num_batches, num_nodes, max_edges_per_node), -1, dtype=np.int32)  # store time index from which the edge was last affected. -1 means never updated


    def total_used_nodes(self):
        return self.next_free_node


    def total_used_edges(self):
        # each undirected edge is stored twice (once per endpoint), so divide by 2
        return np.sum(self.next_free_edge, axis=1) // 2
    

    def get_node_context(self):
        # fetch content at the head and gather all connected nodes' content, using numpy advanced indexing and broadcasting
        batch_size = self.nodes.shape[0]
        node_dim = self.nodes.shape[2]
        max_edges_per_node = self.edges.shape[2]
        context = np.zeros((batch_size, self.C, node_dim), dtype=np.float32)
        batch_idx = np.arange(batch_size)
        context[:, 0, :] = self.nodes[batch_idx, self.head, :]
        
        # gather connected nodes' content with numpy advanced indexing and broadcasting
        neighbor_indices = self.edges[batch_idx, self.head, :]  # (batch_size, max_edges_per_node), source nodes (-1 = empty)
        safe_indices = np.clip(neighbor_indices, 0, self.nodes.shape[1] - 1)  # clamp -1 to valid range before indexing
        context[:, 1:, :] = self.nodes[batch_idx[:, None], safe_indices, :]  # (batch_size, max_edges_per_node, node_dim)

        # zero out free (unused) edge slots
        used_edges = self.next_free_edge[batch_idx, self.head]  # (batch_size,)
        slot_indices = np.arange(max_edges_per_node)  # (max_edges_per_node,)
        valid_mask = slot_indices[None, :] < used_edges[:, None]  # (batch_size, max_edges_per_node)
        context[:, 1:, :] *= valid_mask[:, :, None]

        return context
    

    def get_cause_times(self):
        # fetch the edge update time at the head
        # return shape: (batch_size, max_edges_per_node)
        batch_size = self.nodes.shape[0]
        batch_idx = np.arange(batch_size)
        return self.edge_cause_time[batch_idx, self.head, :]


    def reset_timestamp(self):
        self.timestep = 0
        self.edge_cause_time.fill(-1)


    def reset(self, batch_indices):
        batch_indices = np.asarray(batch_indices)
        self.nodes[batch_indices, :, :] = 0.0
        self.edges[batch_indices, :, :] = -1
        self.next_free_node[batch_indices] = 1  # node 0 always exists after reset
        self.next_free_edge[batch_indices, :] = 0
        self.head[batch_indices] = 0

        self.edge_cause_time[batch_indices, :, :] = -1

        return np.ones(len(batch_indices), dtype=bool)


    def write(self, batch_indices, write_value):
        batch_indices = np.asarray(batch_indices)
        self.nodes[batch_indices, self.head[batch_indices], :] = write_value  # write to head node

        # update trace: for all nodes with head as edge source (head→node), update edge_cause_time
        if len(batch_indices) > 0:
            max_edges = self.edges.shape[2]
            heads = self.head[batch_indices]                                    # (B,)
            edge_vals = self.edges[batch_indices]                               # (B, num_nodes, max_edges)
            used = self.next_free_edge[batch_indices]                           # (B, num_nodes)
            slot_range = np.arange(max_edges)
            occupied = slot_range[None, None, :] < used[:, :, None]            # (B, num_nodes, max_edges)
            points_to_head = edge_vals == heads[:, None, None]                 # (B, num_nodes, max_edges)
            update_mask = points_to_head & occupied
            temp = self.edge_cause_time[batch_indices]
            temp[update_mask] = self.timestep
            self.edge_cause_time[batch_indices] = temp

        return np.ones(len(batch_indices), dtype=bool)


    def move(self, batch_indices, next_edge):
        batch_indices = np.asarray(batch_indices)
        next_edge = np.asarray(next_edge, dtype=np.int32)
        # check that next_edge is a valid (occupied) edge slot on the head node
        used_edges = self.next_free_edge[batch_indices, self.head[batch_indices]]
        success = next_edge < used_edges

        move_batches = batch_indices[success]
        if len(move_batches) > 0:
            next_node = self.edges[move_batches, self.head[move_batches], next_edge[success]]
            self.head[move_batches] = next_node
        return success


    def create(self, batch_indices, write_value):
        # check: free node slot AND free edge slot on head
        # the new node always has a free back-link slot since it starts empty
        num_nodes = self.nodes.shape[1]
        batch_indices = np.asarray(batch_indices)
        write_value = np.asarray(write_value)
        free_nodes = self.next_free_node[batch_indices]
        head_free_edges = self.next_free_edge[batch_indices, self.head[batch_indices]]
        success = (free_nodes < num_nodes) & (head_free_edges < self.edges.shape[2])

        write_batches = batch_indices[success]
        if len(write_batches) > 0:
            new_node_indices = self.next_free_node[write_batches]  # index of the newly created node
            self.nodes[write_batches, new_node_indices, :] = write_value[success]
            self.next_free_node[write_batches] += 1
            head_indices = self.head[write_batches]
            # new_node → head (new_node is source; stored at head's edge list)
            link_slots = self.next_free_edge[write_batches, head_indices]
            self.edges[write_batches, head_indices, link_slots] = new_node_indices
            self.next_free_edge[write_batches, head_indices] += 1
            # head → new_node (head is source; stored at new_node's edge list)
            back_slots = self.next_free_edge[write_batches, new_node_indices]
            self.edges[write_batches, new_node_indices, back_slots] = head_indices
            self.next_free_edge[write_batches, new_node_indices] += 1
            # update trace: a->z direction (new node's slot pointing back to head = head caused new node)
            self.edge_cause_time[write_batches, new_node_indices, back_slots] = self.timestep

        return success


    def link(self, batch_indices, edge_1, edge_2):
        # edge_1 and edge_2 are edge slot indices on the head node
        # resolve them to actual node indices via the head's edge list
        # edges are undirected: both endpoints are updated
        max_edges = self.edges.shape[2]
        batch_indices = np.asarray(batch_indices)
        edge_1 = np.asarray(edge_1, dtype=np.int32)
        edge_2 = np.asarray(edge_2, dtype=np.int32)

        # check that both edge_1 and edge_2 are valid (occupied) slots on the head
        used_edges = self.next_free_edge[batch_indices, self.head[batch_indices]]
        valid_slots = (edge_1 < used_edges) & (edge_2 < used_edges)

        safe_used = np.maximum(used_edges, 1)
        # get the node indices that edge_1 and edge_2 of the head point to
        src_nodes = np.where(valid_slots, self.edges[batch_indices, self.head[batch_indices], np.minimum(edge_1, safe_used - 1)], 0)
        dst_nodes = np.where(valid_slots, self.edges[batch_indices, self.head[batch_indices], np.minimum(edge_2, safe_used - 1)], 0)

        # both src and dst must have a free edge slot (undirected edge stored on both ends)
        free_on_src = self.next_free_edge[batch_indices, src_nodes]
        free_on_dst = self.next_free_edge[batch_indices, dst_nodes]
        success = valid_slots & (free_on_src < max_edges) & (free_on_dst < max_edges)

        write_batches = batch_indices[success]
        write_src = src_nodes[success]
        write_dst = dst_nodes[success]
        if len(write_batches) > 0:
            # dst → src (dst is source; stored at src's edge list)
            slots_src = self.next_free_edge[write_batches, write_src]
            self.edges[write_batches, write_src, slots_src] = write_dst
            self.next_free_edge[write_batches, write_src] += 1
            # src → dst (src is source; stored at dst's edge list)
            slots_dst = self.next_free_edge[write_batches, write_dst]
            self.edges[write_batches, write_dst, slots_dst] = write_src
            self.next_free_edge[write_batches, write_dst] += 1
            # update trace: a->src and a->dst (head's slot in src/dst edge lists = head caused src/dst to change)
            heads_write = self.head[write_batches]
            src_head_slots = np.argmax(self.edges[write_batches, write_src, :] == heads_write[:, None], axis=1)
            self.edge_cause_time[write_batches, write_src, src_head_slots] = self.timestep
            dst_head_slots = np.argmax(self.edges[write_batches, write_dst, :] == heads_write[:, None], axis=1)
            self.edge_cause_time[write_batches, write_dst, dst_head_slots] = self.timestep

        return success
    

    def _remove_edge(self, batches, src_nodes, dst_nodes):
        # remove the dst_node→src_node edge (slot at src_node storing dst_node) via swap-with-last
        # precondition: the edge is guaranteed to exist
        max_edges = self.edges.shape[2]
        slot_range = np.arange(max_edges)
        src_edge_list = self.edges[batches, src_nodes, :]           # (n, max_edges)
        used = self.next_free_edge[batches, src_nodes]              # (n,)
        valid = slot_range[None, :] < used[:, None]                 # (n, max_edges)
        matches = (src_edge_list == dst_nodes[:, None]) & valid     # (n, max_edges)
        slots = np.argmax(matches, axis=1)                          # first matching slot per item
        last = used - 1
        last_nodes = self.edges[batches, src_nodes, last]
        self.edges[batches, src_nodes, slots] = last_nodes
        self.edges[batches, src_nodes, last] = -1
        # propagate edge_cause_time for the swapped slot; removal counts as a change on the last slot
        self.edge_cause_time[batches, src_nodes, slots] = self.edge_cause_time[batches, src_nodes, last]
        self.edge_cause_time[batches, src_nodes, last] = self.timestep
        self.next_free_edge[batches, src_nodes] -= 1

    def rotate(self, batch_indices, edge_1, edge_2):
        # edge_1 is the pivot slot on head, edge_2 is the src slot on head
        # edges are undirected, so rotation:
        #   removes head <-> pivot
        #   adds    src  <-> pivot
        # condition: both slots valid on head; src must have a free edge slot
        # (pivot always gains a free slot from the removal before the addition)
        batch_indices = np.asarray(batch_indices)
        edge_1 = np.asarray(edge_1, dtype=np.int32)
        edge_2 = np.asarray(edge_2, dtype=np.int32)

        heads = self.head[batch_indices]
        used_edges = self.next_free_edge[batch_indices, heads]
        valid_slots = (edge_1 < used_edges) & (edge_2 < used_edges)

        safe_used = np.maximum(used_edges, 1)
        e1_clamped = np.minimum(edge_1, safe_used - 1)
        e2_clamped = np.minimum(edge_2, safe_used - 1)

        pivot_nodes = self.edges[batch_indices, heads, e1_clamped]  # node to be re-parented
        src_nodes   = self.edges[batch_indices, heads, e2_clamped]  # node that adopts pivot

        # after removing head<->pivot, pivot frees one slot — only src needs a free slot checked now
        free_on_src = self.next_free_edge[batch_indices, src_nodes]
        success = valid_slots & (free_on_src < self.edges.shape[2])

        rotate_batches = batch_indices[success]
        if len(rotate_batches) > 0:
            rotate_heads  = heads[success]
            rotate_pivots = pivot_nodes[success]
            rotate_srcs   = src_nodes[success]

            # remove head <-> pivot (both directions)
            self._remove_edge(rotate_batches, rotate_heads,  rotate_pivots)
            self._remove_edge(rotate_batches, rotate_pivots, rotate_heads)

            # add pivot→src and src→pivot edges (each stored at the destination's edge list)
            slots_src = self.next_free_edge[rotate_batches, rotate_srcs]
            self.edges[rotate_batches, rotate_srcs, slots_src] = rotate_pivots
            self.next_free_edge[rotate_batches, rotate_srcs] += 1

            slots_pivot = self.next_free_edge[rotate_batches, rotate_pivots]
            self.edges[rotate_batches, rotate_pivots, slots_pivot] = rotate_srcs
            self.next_free_edge[rotate_batches, rotate_pivots] += 1
            # update trace: a->src (head's slot in src = head caused src to gain pivot)
            # and s->pivot (new slot in pivot = src/head caused pivot to be re-parented)
            src_head_slots = np.argmax(self.edges[rotate_batches, rotate_srcs, :] == rotate_heads[:, None], axis=1)
            self.edge_cause_time[rotate_batches, rotate_srcs, src_head_slots] = self.timestep
            self.edge_cause_time[rotate_batches, rotate_pivots, slots_pivot] = self.timestep

        return success
    

    def execute(self, operations, write_value, edge_1, edge_2):
        # operations is a list of Graph_Memory_Operation_Type
        # group together operations by type and execute in batches for efficiency,
        # then reassemble results in the original argument order
        ops_arr = np.array(operations)
        batch_indices = np.arange(len(operations))
        success = np.zeros(len(operations), dtype=bool)
        for op in set(operations):
            op_indices = batch_indices[ops_arr == op]
            if op == Graph_Memory_Operation_Type.CREATE:
                op_success = self.create(op_indices, write_value[op_indices])
            elif op == Graph_Memory_Operation_Type.LINK:
                op_success = self.link(op_indices, edge_1[op_indices], edge_2[op_indices])
            elif op == Graph_Memory_Operation_Type.WRITE:
                op_success = self.write(op_indices, write_value[op_indices])
            elif op == Graph_Memory_Operation_Type.MOVE:
                op_success = self.move(op_indices, edge_1[op_indices])
            elif op == (Graph_Memory_Operation_Type.WRITE | Graph_Memory_Operation_Type.MOVE):
                write_ok = self.write(op_indices, write_value[op_indices])
                move_ok = self.move(op_indices, edge_1[op_indices])
                op_success = write_ok & move_ok
            elif op == Graph_Memory_Operation_Type.ROTATE:
                op_success = self.rotate(op_indices, edge_1[op_indices], edge_2[op_indices])
            elif op == Graph_Memory_Operation_Type.RESET:
                op_success = self.reset(op_indices)
            else:
                op_success = np.ones(len(op_indices), dtype=bool)  # IDLE: always successful
            success[op_indices] = op_success

        self.timestep += 1  # increment timestep for trace
        return success



if __name__ == "__main__":

    # ------------------------------------------------------------------ helpers
    def assert_equal(name, actual, expected):
        if not np.array_equal(actual, expected):
            raise AssertionError(f"FAIL [{name}]\n  expected: {expected}\n  actual:   {actual}")
        print(f"  PASS  {name}")

    def assert_allclose(name, actual, expected):
        if not np.allclose(actual, expected):
            raise AssertionError(f"FAIL [{name}]\n  expected: {expected}\n  actual:   {actual}")
        print(f"  PASS  {name}")

    # ================================================================== TEST 1
    # initial state: node 0 pre-allocated, no edges
    print("TEST 1: initial state")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    assert_equal("total_used_nodes", m.total_used_nodes(), [1, 1])
    assert_equal("total_used_edges", m.total_used_edges(), [0, 0])

    # ================================================================== TEST 2
    # create: bidirectional edges — new_node→head stored at head, head→new_node stored at new_node
    print("TEST 2: create bidirectional back-link")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[0] = 1  # already 1 by default (node 0 pre-allocated); explicit for clarity
    ok = m.create(np.array([0]), np.array([[1.0, 2.0]]))
    assert_equal("create success", ok, [True])
    assert_equal("next_free_node after create", m.next_free_node, [2])
    assert_allclose("new node value written", m.nodes[0, 1], [1.0, 2.0])
    assert_equal("head stores new_node (new_node→head edge)", m.edges[0, 0, 0], 1)
    assert_equal("new_node stores head (head→new_node edge)", m.edges[0, 1, 0], 0)
    assert_equal("head edge count", m.next_free_edge[0, 0], 1)
    assert_equal("new_node edge count", m.next_free_edge[0, 1], 1)
    assert_equal("total_used_edges = 1", m.total_used_edges(), [1])

    # ================================================================== TEST 3
    # create: fresh initial state — node 0 pre-exists, first create produces node 1 (no self-loop)
    print("TEST 3: create from fresh initial state")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    ok = m.create(np.array([0]), np.array([[5.0, 6.0]]))
    assert_equal("create success", ok, [True])
    assert_allclose("node 1 written", m.nodes[0, 1], [5.0, 6.0])
    assert_equal("next_free_node", m.next_free_node[0], 2)
    assert_equal("head stores node 1 (node1→head edge)", m.edges[0, 0, 0], 1)
    assert_equal("node 1 stores head (head→node1 edge)", m.edges[0, 1, 0], 0)
    assert_equal("head edge count", m.next_free_edge[0, 0], 1)
    assert_equal("node 1 edge count", m.next_free_edge[0, 1], 1)
    assert_equal("total_used_edges = 1", m.total_used_edges(), [1])

    # ================================================================== TEST 4
    # create: fails when no free node slot
    print("TEST 4: create overflow — no free nodes")
    m = NP_Graph_Memory(num_batches=1, num_nodes=2, max_edges_per_node=3, node_dim=2)
    m.next_free_node[0] = 1
    m.create(np.array([0]), np.array([[1.0, 0.0]]))  # creates node 1, now full
    ok = m.create(np.array([0]), np.array([[9.0, 9.0]]))
    assert_equal("create overflow fails", ok, [False])
    assert_equal("node count stays at 2", m.next_free_node[0], 2)

    # ================================================================== TEST 5
    # create: fails when head edge slots are full
    print("TEST 5: create fails — head edge slots full")
    m = NP_Graph_Memory(num_batches=1, num_nodes=6, max_edges_per_node=2, node_dim=2)
    m.next_free_node[0] = 1
    m.create(np.array([0]), np.array([[1.0, 0.0]]))  # head slot 0 used
    m.create(np.array([0]), np.array([[2.0, 0.0]]))  # head slot 1 used — head now full
    ok = m.create(np.array([0]), np.array([[3.0, 0.0]]))
    assert_equal("create fails when head full", ok, [False])
    assert_equal("node count unchanged", m.next_free_node[0], 3)

    # ================================================================== TEST 6
    # create: partial success across batches
    print("TEST 6: create partial success")
    m = NP_Graph_Memory(num_batches=2, num_nodes=2, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    m.create(np.array([0]), np.array([[1.0, 0.0]]))  # batch 0 node 1 created — now full
    ok = m.create(np.array([0, 1]), np.array([[9.0, 9.0], [2.0, 0.0]]))
    assert_equal("partial success", ok, [False, True])
    assert_allclose("batch 0 node 1 unchanged", m.nodes[0, 1], [1.0, 0.0])
    assert_allclose("batch 1 node 1 written", m.nodes[1, 1], [2.0, 0.0])
    assert_equal("batch 1 node1 stores head (head→node1 edge)", m.edges[1, 1, 0], 0)

    # ================================================================== TEST 7
    # write: writes value to head node
    print("TEST 7: write")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=2, node_dim=2)
    ok = m.write(np.array([0, 1]), np.array([[5.0, 6.0], [7.0, 8.0]]))
    assert_equal("write success", ok, [True, True])
    assert_allclose("written to head batch 0", m.nodes[0, 0], [5.0, 6.0])
    assert_allclose("written to head batch 1", m.nodes[1, 0], [7.0, 8.0])
    assert_equal("head unchanged batch 0", m.head[0], 0)
    assert_equal("head unchanged batch 1", m.head[1], 0)

    # ================================================================== TEST 8
    # move: follow a valid edge slot
    print("TEST 8: move valid")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[0] = 1
    m.create(np.array([0]), np.array([[1.0, 0.0]]))  # creates node 1; head(0)↔node1
    ok = m.move(np.array([0]), np.array([0]))  # follow slot 0 of head(0) -> node 1
    assert_equal("move success", ok, [True])
    assert_equal("head moved to node 1", m.head[0], 1)

    # ================================================================== TEST 9
    # move: traverse back via back-link
    print("TEST 9: move back via back-link")
    # head is now at node 1, which has back-link to node 0 at slot 0
    ok = m.move(np.array([0]), np.array([0]))
    assert_equal("move back success", ok, [True])
    assert_equal("head back at node 0", m.head[0], 0)

    # ================================================================== TEST 10
    # move: fails on unoccupied edge slot
    print("TEST 10: move invalid slot")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    m.create(np.array([0]), np.array([[1.0, 0.0]]))  # batch 0: slot 0 occupied
    ok = m.move(np.array([0, 1]), np.array([1, 0]))  # batch 0: slot 1 free; batch 1: slot 0 free
    assert_equal("move fails on free slots", ok, [False, False])
    assert_equal("head unchanged batch 0", m.head[0], 0)
    assert_equal("head unchanged batch 1", m.head[1], 0)

    # ================================================================== TEST 11
    # get_node_context: no edges — only head slot filled, rest zeroed
    print("TEST 11: get_node_context no edges")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=3)
    m.nodes[0, 0] = [1.0, 2.0, 3.0]
    ctx = m.get_node_context()
    assert_allclose("head content", ctx[0, 0], [1.0, 2.0, 3.0])
    assert_allclose("unused slot 0 zeroed", ctx[0, 1], [0.0, 0.0, 0.0])
    assert_allclose("unused slot 1 zeroed", ctx[0, 2], [0.0, 0.0, 0.0])

    # ================================================================== TEST 12
    # get_node_context: occupied slots show neighbor content
    print("TEST 12: get_node_context with edges")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.nodes[0, 0] = [10.0, 11.0]
    m.nodes[0, 1] = [20.0, 21.0]
    m.nodes[0, 2] = [30.0, 31.0]
    m.edges[0, 0, 0] = 1;  m.edges[0, 0, 1] = 2;  m.next_free_edge[0, 0] = 2
    ctx = m.get_node_context()
    assert_allclose("head", ctx[0, 0], [10.0, 11.0])
    assert_allclose("neighbor slot 0", ctx[0, 1], [20.0, 21.0])
    assert_allclose("neighbor slot 1", ctx[0, 2], [30.0, 31.0])

    # ================================================================== TEST 13
    # get_node_context: partial edges — free slot must be zeroed
    print("TEST 13: get_node_context partial edges masked")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.nodes[0, 0] = [1.0, 0.0];  m.nodes[0, 1] = [2.0, 0.0]
    m.edges[0, 0, 0] = 1;  m.next_free_edge[0, 0] = 1
    ctx = m.get_node_context()
    assert_allclose("occupied slot filled", ctx[0, 1], [2.0, 0.0])
    assert_allclose("free slot zeroed", ctx[0, 2], [0.0, 0.0])

    # ================================================================== TEST 14
    # link: adds bidirectional edge between two neighbors of head
    print("TEST 14: link basic bidirectional")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    # manually wire: head(0) ↔ node1, head(0) ↔ node2
    m.edges[0, 0, 0] = 1;  m.edges[0, 0, 1] = 2;  m.next_free_edge[0, 0] = 2
    m.edges[0, 1, 0] = 0;  m.next_free_edge[0, 1] = 1
    m.edges[0, 2, 0] = 0;  m.next_free_edge[0, 2] = 1
    assert_equal("total_used_edges before link", m.total_used_edges(), [2])
    ok = m.link(np.array([0]), np.array([0]), np.array([1]))  # link node1 <-> node2
    assert_equal("link success", ok, [True])
    assert_equal("node1 -> node2 added", m.edges[0, 1, 1], 2)
    assert_equal("node2 -> node1 added", m.edges[0, 2, 1], 1)
    assert_equal("node1 edge count", m.next_free_edge[0, 1], 2)
    assert_equal("node2 edge count", m.next_free_edge[0, 2], 2)
    assert_equal("total_used_edges after link", m.total_used_edges(), [3])

    # ================================================================== TEST 15
    # link: fails when destination node is full (new undirected requirement)
    print("TEST 15: link fails — dst node full")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=1, node_dim=2)
    # head(0) -> node1 (slot 0), node1 -> head (back-link, fills its only slot)
    m.edges[0, 0, 0] = 1;  m.next_free_edge[0, 0] = 1  # head full too
    m.edges[0, 1, 0] = 0;  m.next_free_edge[0, 1] = 1  # node1 full
    ok = m.link(np.array([0]), np.array([0]), np.array([0]))  # try node1 <-> node1 (self)
    assert_equal("link fails dst full", ok, [False])
    assert_equal("node1 edge count unchanged", m.next_free_edge[0, 1], 1)

    # ================================================================== TEST 16
    # link: fails when edge slot indices are not occupied on head
    print("TEST 16: link fails — invalid slot indices")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.edges[0, 0, 0] = 1;  m.next_free_edge[0, 0] = 1  # only slot 0 occupied
    ok = m.link(np.array([0]), np.array([0]), np.array([1]))  # edge_2=1 not occupied
    assert_equal("link fails invalid slot", ok, [False])

    # ================================================================== TEST 17
    # rotate: basic — remove head↔pivot, add src↔pivot (both bidirectional)
    print("TEST 17: rotate basic")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    # head(0) ↔ node1 (slot 0), head(0) ↔ node2 (slot 1), both bidirectional
    m.edges[0, 0, 0] = 1;  m.edges[0, 0, 1] = 2;  m.next_free_edge[0, 0] = 2
    m.edges[0, 1, 0] = 0;  m.next_free_edge[0, 1] = 1
    m.edges[0, 2, 0] = 0;  m.next_free_edge[0, 2] = 1
    assert_equal("total_used_edges before rotate", m.total_used_edges(), [2])
    ok = m.rotate(np.array([0]), np.array([0]), np.array([1]))  # pivot=node1, src=node2
    assert_equal("rotate success", ok, [True])
    # head: lost node1, still has node2 → 1 edge
    assert_equal("head edge count", m.next_free_edge[0, 0], 1)
    assert_equal("head slot 0 = node2 (swapped)", m.edges[0, 0, 0], 2)
    # node1: lost head, gained node2 → 1 edge
    assert_equal("node1 edge count", m.next_free_edge[0, 1], 1)
    assert_equal("node1 -> node2", m.edges[0, 1, 0], 2)
    # node2: still has head (slot 0), gained node1 → 2 edges
    assert_equal("node2 edge count", m.next_free_edge[0, 2], 2)
    assert_equal("total_used_edges unchanged after rotate", m.total_used_edges(), [2])

    # ================================================================== TEST 18
    # rotate: fails when src node has no free edge slot
    print("TEST 18: rotate fails — src node full")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.edges[0, 0, 0] = 1;  m.edges[0, 0, 1] = 2;  m.next_free_edge[0, 0] = 2
    m.edges[0, 1, 0] = 0;  m.next_free_edge[0, 1] = 1
    m.edges[0, 2, 0] = 0;  m.edges[0, 2, 1] = 3;  m.next_free_edge[0, 2] = 2  # node2 full
    ok = m.rotate(np.array([0]), np.array([0]), np.array([1]))  # pivot=node1, src=node2(full)
    assert_equal("rotate fails src full", ok, [False])
    assert_equal("head edge count unchanged", m.next_free_edge[0, 0], 2)
    assert_equal("node1 edge count unchanged", m.next_free_edge[0, 1], 1)
    assert_equal("node2 edge count unchanged", m.next_free_edge[0, 2], 2)

    # ================================================================== TEST 19
    # rotate: fails when edge slot indices are not occupied on head
    print("TEST 19: rotate fails — invalid slot indices")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.edges[0, 0, 0] = 1;  m.next_free_edge[0, 0] = 1  # only slot 0 occupied
    ok = m.rotate(np.array([0]), np.array([1]), np.array([0]))  # edge_1=1 not occupied
    assert_equal("rotate fails invalid edge_1", ok, [False])
    ok = m.rotate(np.array([0]), np.array([0]), np.array([1]))  # edge_2=1 not occupied
    assert_equal("rotate fails invalid edge_2", ok, [False])
    assert_equal("head edge count unchanged", m.next_free_edge[0, 0], 1)

    # ================================================================== TEST 20
    # rotate: partial success across batches
    print("TEST 20: rotate partial multi-batch")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    for b in range(2):
        m.edges[b, 0, 0] = 1;  m.edges[b, 0, 1] = 2;  m.next_free_edge[b, 0] = 2
        m.edges[b, 1, 0] = 0;  m.next_free_edge[b, 1] = 1
        m.edges[b, 2, 0] = 0;  m.next_free_edge[b, 2] = 1
    m.edges[1, 2, 1] = 3;  m.edges[1, 2, 2] = 3;  m.next_free_edge[1, 2] = 3  # batch 1 src full
    ok = m.rotate(np.array([0, 1]), np.array([0, 0]), np.array([1, 1]))
    assert_equal("rotate partial success", ok, [True, False])
    assert_equal("batch 0 head edge count", m.next_free_edge[0, 0], 1)
    assert_equal("batch 0 node1 -> node2", m.edges[0, 1, 0], 2)
    assert_equal("batch 1 head unchanged", m.next_free_edge[1, 0], 2)
    assert_equal("batch 1 node1 unchanged", m.next_free_edge[1, 1], 1)

    # ================================================================== TEST 21
    # reset: clears nodes, edges, counters, head for given batches
    print("TEST 21: reset basic")
    m = NP_Graph_Memory(num_batches=2, num_nodes=3, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    m.create(np.array([0, 1]), np.array([[1.0, 2.0], [3.0, 4.0]]))
    m.head[0] = 1
    m.reset(np.array([0]))
    assert_allclose("nodes zeroed", m.nodes[0], np.zeros((3, 2)))
    assert_equal("edges cleared", m.edges[0], np.full((3, 3), -1, dtype=np.int32))
    assert_equal("next_free_node reset", m.next_free_node[0], 1)
    assert_equal("next_free_edge reset", m.next_free_edge[0], [0, 0, 0])
    assert_equal("head reset", m.head[0], 0)
    # batch 1 untouched
    assert_equal("batch 1 node count untouched", m.next_free_node[1], 2)
    assert_allclose("batch 1 node untouched", m.nodes[1, 1], [3.0, 4.0])

    # ================================================================== TEST 22
    # execute: mixed operations dispatched correctly
    print("TEST 22: execute mixed operations")
    m = NP_Graph_Memory(num_batches=3, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    # batch 1: setup for MOVE
    m.edges[1, 0, 0] = 2;  m.edges[1, 2, 0] = 0;  m.next_free_edge[1, 0] = 1;  m.next_free_edge[1, 2] = 1
    # batch 2: something to reset
    m.nodes[2, 0] = [9.0, 9.0];  m.next_free_node[2] = 2
    write_value = np.array([[5.0, 6.0], [0.0, 0.0], [0.0, 0.0]])
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.CREATE, OP.MOVE, OP.RESET], write_value, np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert_equal("execute success", ok, [True, True, True])
    # batch 0: create -> node 1 written, bidirectional edge
    assert_allclose("create wrote node", m.nodes[0, 1], [5.0, 6.0])
    assert_equal("create forward edge", m.edges[0, 0, 0], 1)
    assert_equal("create back-link", m.edges[0, 1, 0], 0)
    # batch 1: move -> head moved to node 2
    assert_equal("move head", m.head[1], 2)
    # batch 2: reset -> cleared
    assert_equal("reset node count", m.next_free_node[2], 1)
    assert_allclose("reset node zeroed", m.nodes[2, 0], [0.0, 0.0])

    # ================================================================== TEST 23
    # execute: ROTATE dispatched correctly
    print("TEST 23: execute rotate")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.edges[0, 0, 0] = 1;  m.edges[0, 0, 1] = 2;  m.next_free_edge[0, 0] = 2
    m.edges[0, 1, 0] = 0;  m.next_free_edge[0, 1] = 1
    m.edges[0, 2, 0] = 0;  m.next_free_edge[0, 2] = 1
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.ROTATE], np.zeros((1, 2)), np.array([0]), np.array([1]))
    assert_equal("execute rotate success", ok, [True])
    assert_equal("execute rotate head edge count", m.next_free_edge[0, 0], 1)
    assert_equal("execute rotate node1 -> node2", m.edges[0, 1, 0], 2)
    assert_equal("execute rotate node2 has node1", m.next_free_edge[0, 2], 2)

    # ================================================================== TEST 24
    # execute: IDLE is a no-op, WRITE dispatched correctly
    print("TEST 24: execute IDLE and WRITE")
    m = NP_Graph_Memory(num_batches=3, num_nodes=4, max_edges_per_node=2, node_dim=2)
    write_value = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.WRITE, OP.IDLE, OP.WRITE], write_value, np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    assert_equal("WRITE+IDLE success", ok, [True, True, True])
    assert_allclose("batch 0 written", m.nodes[0, 0], [1.0, 0.0])
    assert_equal("batch 1 IDLE untouched", m.next_free_node[1], 1)
    assert_allclose("batch 2 written", m.nodes[2, 0], [2.0, 0.0])

    # ================================================================== TEST 25
    # WRITE|MOVE combo: both succeed — value written to old head, head moves
    print("TEST 25: WRITE|MOVE both succeed")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[0] = 1
    m.create(np.array([0]), np.array([[0.0, 0.0]]))  # creates node 1; head(0) ↔ node1
    OP = Graph_Memory_Operation_Type
    ok = m.execute(
        [OP.WRITE | OP.MOVE],
        np.array([[7.0, 8.0]]),
        np.array([0]),  # edge slot 0 on head -> node1
        np.zeros(1, dtype=int),
    )
    assert_equal("write|move success", ok, [True])
    assert_allclose("value written to old head (node 0)", m.nodes[0, 0], [7.0, 8.0])
    assert_equal("head moved to node 1", m.head[0], 1)

    # ================================================================== TEST 26
    # WRITE|MOVE combo: move fails (invalid slot) — write still happens, returns False
    print("TEST 26: WRITE|MOVE move fails — write still applied")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=3, node_dim=2)
    OP = Graph_Memory_Operation_Type
    ok = m.execute(
        [OP.WRITE | OP.MOVE],
        np.array([[5.0, 6.0]]),
        np.array([0]),  # slot 0 not occupied — no edges on head
        np.zeros(1, dtype=int),
    )
    assert_equal("write|move returns False when move fails", ok, [False])
    assert_allclose("write still applied to head", m.nodes[0, 0], [5.0, 6.0])
    assert_equal("head unchanged", m.head[0], 0)

    # ================================================================== TEST 27
    # WRITE|MOVE combo: partial success across batches
    print("TEST 27: WRITE|MOVE partial success")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    m.create(np.array([0]), np.array([[0.0, 0.0]]))  # batch 0: head(0) ↔ node1, slot 0 valid
    # batch 1: no edges, slot 0 invalid
    OP = Graph_Memory_Operation_Type
    ok = m.execute(
        [OP.WRITE | OP.MOVE, OP.WRITE | OP.MOVE],
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0, 0]),
        np.zeros(2, dtype=int),
    )
    assert_equal("partial success", ok, [True, False])
    assert_allclose("batch 0 value written to old head", m.nodes[0, 0], [1.0, 2.0])
    assert_equal("batch 0 head moved to node 1", m.head[0], 1)
    assert_allclose("batch 1 write still applied", m.nodes[1, 0], [3.0, 4.0])
    assert_equal("batch 1 head unchanged", m.head[1], 0)

    # ================================================================== TEST 28
    # execute: WRITE|MOVE mixed with other ops in same call
    print("TEST 28: WRITE|MOVE mixed with CREATE in execute")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    m.next_free_node[:] = 1
    m.create(np.array([0]), np.array([[0.0, 0.0]]))  # batch 0: head(0) ↔ node1
    OP = Graph_Memory_Operation_Type
    ok = m.execute(
        [OP.WRITE | OP.MOVE, OP.CREATE],
        np.array([[9.0, 9.0], [5.0, 5.0]]),
        np.array([0, 0]),
        np.zeros(2, dtype=int),
    )
    assert_equal("mixed ops success", ok, [True, True])
    assert_allclose("batch 0 written to old head", m.nodes[0, 0], [9.0, 9.0])
    assert_equal("batch 0 head moved to node 1", m.head[0], 1)
    assert_allclose("batch 1 new node written", m.nodes[1, 1], [5.0, 5.0])
    assert_equal("batch 1 new node back-link to head", m.edges[1, 1, 0], 0)

    print("\nAll tests passed.")
