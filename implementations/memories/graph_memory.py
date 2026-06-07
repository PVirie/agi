import numpy as np

try:
    from interfaces.memory import Graph_Memory, Graph_Memory_Operation_Type
except ImportError:
    # Define dummy classes for testing without the full interface
    from enum import Flag, auto

    class Graph_Memory_Operation_Type(Flag):
        IDLE = 0
        CREATE = auto()
        WRITE_THEN_MOVE = auto()
        LINK = auto()
        RESET = auto()

    class Graph_Memory:
        def write_then_move(self, batch_indices, write_value, next_edge):
            pass

        def create(self, batch_indices, write_value):
            pass

        def link(self, batch_indices, edge_1, edge_2):
            pass

        def reset(self, batch_indices):
            pass


class NP_Graph_Memory(Graph_Memory):

    def __init__(self, num_batches, num_nodes, max_edges_per_node, node_dim):

        self.C = 1 + max_edges_per_node

        self.nodes = np.zeros((num_batches, num_nodes, node_dim), dtype=np.float32)
        self.edges = np.zeros((num_batches, num_nodes, max_edges_per_node), dtype=np.int32) # store the destination node index for each edge slot, 0 means no edge

        self.next_free_node = np.zeros((num_batches,), dtype=np.int32)  # track the next free node index for each batch
        self.next_free_edge = np.zeros((num_batches, num_nodes), dtype=np.int32)  # track the next free edge index for each node in each batch

        self.head = np.zeros((num_batches,), dtype=np.int32)  # track the head node index for each batch


    def total_used_nodes(self):
        return self.next_free_node


    def total_used_edges(self):
        return np.sum(self.next_free_edge, axis=1)  # sum over nodes to get total edges used per batch
    

    def get_node_context(self):
        # fetch content at the head and gather all connected nodes' content, using numpy advanced indexing and broadcasting
        batch_size = self.nodes.shape[0]
        node_dim = self.nodes.shape[2]
        max_edges_per_node = self.edges.shape[2]
        context = np.zeros((batch_size, self.C, node_dim), dtype=np.float32)
        batch_idx = np.arange(batch_size)
        context[:, 0, :] = self.nodes[batch_idx, self.head, :]
        
        # gather connected nodes' content with numpy advanced indexing and broadcasting
        neighbor_indices = self.edges[batch_idx, self.head, :]  # (batch_size, max_edges_per_node)
        context[:, 1:, :] = self.nodes[batch_idx[:, None], neighbor_indices, :]  # (batch_size, max_edges_per_node, node_dim)

        # zero out free (unused) edge slots
        used_edges = self.next_free_edge[batch_idx, self.head]  # (batch_size,)
        slot_indices = np.arange(max_edges_per_node)  # (max_edges_per_node,)
        valid_mask = slot_indices[None, :] < used_edges[:, None]  # (batch_size, max_edges_per_node)
        context[:, 1:, :] *= valid_mask[:, :, None]

        return context


    def write_then_move(self, batch_indices, write_value, next_edge):
        batch_indices = np.asarray(batch_indices)
        self.nodes[batch_indices, self.head[batch_indices], :] = write_value  # write to head node
        # move head to next node based on next_edge
        next_node = self.edges[batch_indices, self.head[batch_indices], next_edge]  # get next node index from edges
        self.head[batch_indices] = next_node  # move head to next node
        return np.ones(len(batch_indices), dtype=bool)


    def create(self, batch_indices, write_value):
        # check both: free node slot AND free edge slot on head
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
            # auto-link from head to new node
            head_indices = self.head[write_batches]
            link_slots = self.next_free_edge[write_batches, head_indices]
            self.edges[write_batches, head_indices, link_slots] = new_node_indices
            self.next_free_edge[write_batches, head_indices] += 1

        return success


    def link(self, batch_indices, edge_1, edge_2):
        # edge_1 and edge_2 are edge slot indices on the head node
        # resolve them to actual node indices via the head's edge list
        max_edges = self.edges.shape[2]
        batch_indices = np.asarray(batch_indices)
        edge_1 = np.asarray(edge_1)
        edge_2 = np.asarray(edge_2)

        # get the node indices that edge_1 and edge_2 of the head point to
        src_nodes = self.edges[batch_indices, self.head[batch_indices], edge_1]  # (len(batch_indices),)
        dst_nodes = self.edges[batch_indices, self.head[batch_indices], edge_2]  # (len(batch_indices),)

        # first check whether there is a free edge slot on src_nodes, if not, do nothing
        free_edges = self.next_free_edge[batch_indices, src_nodes]
        success = free_edges < max_edges

        write_batches = batch_indices[success]
        write_src = src_nodes[success]
        write_dst = dst_nodes[success]
        if len(write_batches) > 0:
            slots = self.next_free_edge[write_batches, write_src]
            self.edges[write_batches, write_src, slots] = write_dst
            self.next_free_edge[write_batches, write_src] += 1

        return success
    

    def reset(self, batch_indices):
        batch_indices = np.asarray(batch_indices)
        self.nodes[batch_indices, :, :] = 0.0
        self.edges[batch_indices, :, :] = 0
        self.next_free_node[batch_indices] = 0
        self.next_free_edge[batch_indices, :] = 0
        self.head[batch_indices] = 0
        return np.ones(len(batch_indices), dtype=bool)


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
            elif op == Graph_Memory_Operation_Type.WRITE_THEN_MOVE:
                op_success = self.write_then_move(op_indices, write_value[op_indices], edge_1[op_indices])
            elif op == Graph_Memory_Operation_Type.RESET:
                op_success = self.reset(op_indices)
            else:
                op_success = np.ones(len(op_indices), dtype=bool)  # IDLE: always successful
            success[op_indices] = op_success
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
    # total_used_nodes / total_used_edges start at zero
    print("TEST 1: initial state")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    assert_equal("total_used_nodes", m.total_used_nodes(), [0, 0])
    assert_equal("total_used_edges", m.total_used_edges(), [0, 0])

    # ================================================================== TEST 2
    # create: basic write, return value, counter increment, auto-link from head
    print("TEST 2: create basic")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=3, node_dim=2)
    ok = m.create(np.array([0, 1]), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert_equal("create success", ok, [True, True])
    assert_equal("next_free_node after create", m.next_free_node, [1, 1])
    assert_allclose("node 0 batch 0", m.nodes[0, 0], [1.0, 2.0])
    assert_allclose("node 0 batch 1", m.nodes[1, 0], [3.0, 4.0])
    # head (node 0) should now have an edge to the new node (node 0 itself — same index before increment)
    assert_equal("auto-link edge written batch 0", m.edges[0, 0, 0], 0)
    assert_equal("auto-link edge written batch 1", m.edges[1, 0, 0], 0)
    assert_equal("auto-link edge count batch 0", m.next_free_edge[0, 0], 1)
    assert_equal("auto-link edge count batch 1", m.next_free_edge[1, 0], 1)

    # ================================================================== TEST 3
    # create: overflow — no more nodes available
    print("TEST 3: create overflow")
    m = NP_Graph_Memory(num_batches=1, num_nodes=2, max_edges_per_node=2, node_dim=2)
    m.create(np.array([0]), np.array([[1.0, 0.0]]))
    m.create(np.array([0]), np.array([[2.0, 0.0]]))
    ok = m.create(np.array([0]), np.array([[9.0, 9.0]]))  # should fail
    assert_equal("create overflow fails", ok, [False])
    assert_equal("node count stays at 2", m.next_free_node, [2])
    assert_allclose("no overwrite", m.nodes[0, 1], [2.0, 0.0])
    # auto-link: head moved to node 1 after first write_then_move isn't called here;
    # just verify the edges written by the two successful creates
    assert_equal("first create auto-link", m.edges[0, 0, 0], 0)  # head(0) -> node 0
    assert_equal("second create auto-link", m.edges[0, 0, 1], 1)  # head(0) -> node 1

    # ================================================================== TEST 4
    # create: partial success across batches
    print("TEST 4: create partial")
    m = NP_Graph_Memory(num_batches=2, num_nodes=1, max_edges_per_node=2, node_dim=2)
    m.create(np.array([0]), np.array([[5.0, 6.0]]))   # fill batch 0
    ok = m.create(np.array([0, 1]), np.array([[9.0, 9.0], [7.0, 8.0]]))
    assert_equal("partial success", ok, [False, True])
    assert_allclose("batch 0 unchanged", m.nodes[0, 0], [5.0, 6.0])
    assert_allclose("batch 1 written", m.nodes[1, 0], [7.0, 8.0])
    assert_equal("batch 1 auto-link written", m.edges[1, 0, 0], 0)
    assert_equal("batch 1 auto-link count", m.next_free_edge[1, 0], 1)

    # ================================================================== TEST 5
    # get_node_context: no edges — only head slot filled, rest zeroed
    print("TEST 5: get_node_context no edges")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=3)
    m.nodes[0, 0] = [1.0, 2.0, 3.0]
    # head stays at 0, no edges linked
    ctx = m.get_node_context()   # shape (1, 3, 3)
    assert_allclose("head content", ctx[0, 0], [1.0, 2.0, 3.0])
    assert_allclose("unused edge slot 0 zeroed", ctx[0, 1], [0.0, 0.0, 0.0])
    assert_allclose("unused edge slot 1 zeroed", ctx[0, 2], [0.0, 0.0, 0.0])

    # ================================================================== TEST 6
    # get_node_context: with edges — only occupied slots return content
    print("TEST 6: get_node_context with edges")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.nodes[0, 0] = [10.0, 11.0]
    m.nodes[0, 1] = [20.0, 21.0]
    m.nodes[0, 2] = [30.0, 31.0]
    # manually wire: node 0 -> node 1 (slot 0), node 0 -> node 2 (slot 1)
    m.edges[0, 0, 0] = 1
    m.edges[0, 0, 1] = 2
    m.next_free_edge[0, 0] = 2
    ctx = m.get_node_context()
    assert_allclose("head", ctx[0, 0], [10.0, 11.0])
    assert_allclose("neighbor slot 0", ctx[0, 1], [20.0, 21.0])
    assert_allclose("neighbor slot 1", ctx[0, 2], [30.0, 31.0])

    # ================================================================== TEST 7
    # get_node_context: partial edges — second slot unused, must be zeroed
    print("TEST 7: get_node_context partial edges masked")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.nodes[0, 0] = [1.0, 0.0]
    m.nodes[0, 1] = [2.0, 0.0]
    m.nodes[0, 2] = [9.0, 9.0]  # node 2 exists but not linked
    m.edges[0, 0, 0] = 1
    m.next_free_edge[0, 0] = 1  # only slot 0 used; slot 1 free
    ctx = m.get_node_context()
    assert_allclose("occupied slot filled", ctx[0, 1], [2.0, 0.0])
    assert_allclose("free slot zeroed", ctx[0, 2], [0.0, 0.0])

    # ================================================================== TEST 8
    # write_then_move: writes to head then advances head along an edge
    print("TEST 8: write_then_move")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.edges[0, 0, 0] = 2  # node 0 edge-slot 0 -> node 2
    m.write_then_move(np.array([0]), np.array([[5.0, 6.0]]), np.array([0]))
    assert_allclose("written to old head", m.nodes[0, 0], [5.0, 6.0])
    assert_equal("head moved to node 2", m.head, [2])

    # ================================================================== TEST 9
    # link: connects two nodes visible from head
    print("TEST 9: link basic")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    # head (node 0) has two edges: slot 0 -> node 1, slot 1 -> node 2
    m.edges[0, 0, 0] = 1
    m.edges[0, 0, 1] = 2
    m.next_free_edge[0, 0] = 2
    # link edge_1=0 (node 1) -> edge_2=1 (node 2), i.e. add an edge from node 1 to node 2
    ok = m.link(np.array([0]), np.array([0]), np.array([1]))
    assert_equal("link success", ok, [True])
    assert_equal("edge written on node 1", m.edges[0, 1, 0], 2)
    assert_equal("node 1 edge count incremented", m.next_free_edge[0, 1], 1)

    # ================================================================== TEST 10
    # link: overflow — src node has no free edge slots
    print("TEST 10: link overflow")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=1, node_dim=2)
    m.edges[0, 0, 0] = 1  # head -> node 1 (slot 0)
    m.next_free_edge[0, 0] = 1
    m.edges[0, 1, 0] = 3  # node 1 already has its single edge slot filled
    m.next_free_edge[0, 1] = 1
    # try to link from node 1 (edge_1=0) to anything — node 1 is full
    ok = m.link(np.array([0]), np.array([0]), np.array([0]))
    assert_equal("link overflow fails", ok, [False])
    assert_equal("edge count on node 1 unchanged", m.next_free_edge[0, 1], 1)

    # ================================================================== TEST 11
    # multi-batch link: partial success
    print("TEST 11: link partial multi-batch")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=1, node_dim=2)
    # batch 0: head (node 0) edge slot 0 -> node 1; node 1 edge slot free
    m.edges[0, 0, 0] = 1
    m.next_free_edge[0, 0] = 1
    # batch 1: head (node 0) edge slot 0 -> node 2; node 2 already full
    m.edges[1, 0, 0] = 2
    m.next_free_edge[1, 0] = 1
    m.edges[1, 2, 0] = 3
    m.next_free_edge[1, 2] = 1
    ok = m.link(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]))
    assert_equal("partial link success", ok, [True, False])
    assert_equal("batch 0 edge written", m.edges[0, 1, 0], 1)
    assert_equal("batch 1 unchanged", m.next_free_edge[1, 2], 1)

    # ================================================================== TEST 11b
    # create: fails when head edge slots are full (even if node slot is free)
    print("TEST 11b: create fails when head edge full")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=1, node_dim=2)
    # fill the single edge slot on head (node 0)
    m.edges[0, 0, 0] = 3
    m.next_free_edge[0, 0] = 1
    ok = m.create(np.array([0]), np.array([[5.0, 5.0]]))
    assert_equal("create fails when head full", ok, [False])
    assert_equal("node count unchanged", m.next_free_node[0], 0)
    assert_equal("head edge slot unchanged", m.edges[0, 0, 0], 3)
    assert_equal("head edge count unchanged", m.next_free_edge[0, 0], 1)

    # ================================================================== TEST 12
    # reset: clears nodes, edges, counters, and head for given batch indices
    print("TEST 12: reset basic")
    m = NP_Graph_Memory(num_batches=2, num_nodes=3, max_edges_per_node=2, node_dim=2)
    m.create(np.array([0, 1]), np.array([[1.0, 2.0], [3.0, 4.0]]))
    m.edges[0, 0, 0] = 1
    m.next_free_edge[0, 0] = 1
    m.head[0] = 1
    m.reset(np.array([0]))
    assert_allclose("nodes zeroed after reset", m.nodes[0], np.zeros((3, 2)))
    assert_equal("edges zeroed after reset", m.edges[0], np.zeros((3, 2), dtype=np.int32))
    assert_equal("next_free_node reset", m.next_free_node[0], 0)
    assert_equal("next_free_edge reset", m.next_free_edge[0], [0, 0, 0])
    assert_equal("head reset", m.head[0], 0)
    # batch 1 untouched
    assert_equal("other batch next_free_node untouched", m.next_free_node[1], 1)
    assert_allclose("other batch node untouched", m.nodes[1, 0], [3.0, 4.0])

    # ================================================================== TEST 13
    # reset: multiple batches at once
    print("TEST 13: reset multiple batches")
    m = NP_Graph_Memory(num_batches=3, num_nodes=2, max_edges_per_node=1, node_dim=2)
    m.create(np.array([0, 1, 2]), np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    m.reset(np.array([0, 2]))
    assert_equal("batch 0 cleared", m.next_free_node[0], 0)
    assert_equal("batch 2 cleared", m.next_free_node[2], 0)
    assert_equal("batch 1 untouched", m.next_free_node[1], 1)

    # ================================================================== TEST 14
    # execute: each batch runs a different operation
    print("TEST 14: execute mixed operations")
    m = NP_Graph_Memory(num_batches=3, num_nodes=4, max_edges_per_node=2, node_dim=2)
    # pre-populate batch 1 so write_then_move has an edge to follow
    m.edges[1, 0, 0] = 2
    # pre-populate batch 2 with a node so reset has something to clear
    m.nodes[2, 0] = [9.0, 9.0]
    m.next_free_node[2] = 1
    write_value = np.array([[5.0, 6.0], [7.0, 8.0], [0.0, 0.0]])
    edge_1 = np.array([0, 0, 0])
    edge_2 = np.array([0, 0, 0])
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.CREATE, OP.WRITE_THEN_MOVE, OP.RESET], write_value, edge_1, edge_2)
    assert_equal("execute success order", ok, [True, True, True])
    # batch 0: create -> node 0 written, counter = 1, auto-link head(0)->node 0
    assert_allclose("execute create wrote node", m.nodes[0, 0], [5.0, 6.0])
    assert_equal("execute create counter", m.next_free_node[0], 1)
    assert_equal("execute create auto-link", m.edges[0, 0, 0], 0)
    # batch 1: write_then_move -> node 0 written, head moves to node 2
    assert_allclose("execute write_then_move wrote node", m.nodes[1, 0], [7.0, 8.0])
    assert_equal("execute write_then_move head", m.head[1], 2)
    # batch 2: reset -> everything cleared
    assert_equal("execute reset counter", m.next_free_node[2], 0)
    assert_allclose("execute reset node zeroed", m.nodes[2, 0], [0.0, 0.0])

    # ================================================================== TEST 15
    # execute: all same operation (create) across all batches
    print("TEST 15: execute all-create")
    m = NP_Graph_Memory(num_batches=2, num_nodes=4, max_edges_per_node=2, node_dim=2)
    write_value = np.array([[1.0, 2.0], [3.0, 4.0]])
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.CREATE, OP.CREATE], write_value, np.zeros(2, dtype=int), np.zeros(2, dtype=int))
    assert_equal("all-create success", ok, [True, True])
    assert_equal("all-create counters", m.next_free_node, [1, 1])
    assert_allclose("all-create batch 0", m.nodes[0, 0], [1.0, 2.0])
    assert_allclose("all-create batch 1", m.nodes[1, 0], [3.0, 4.0])
    assert_equal("all-create auto-link batch 0", m.edges[0, 0, 0], 0)
    assert_equal("all-create auto-link batch 1", m.edges[1, 0, 0], 0)

    # ================================================================== TEST 16
    # execute: link operation dispatched correctly
    print("TEST 16: execute link")
    m = NP_Graph_Memory(num_batches=1, num_nodes=4, max_edges_per_node=2, node_dim=2)
    m.edges[0, 0, 0] = 1  # head slot 0 -> node 1
    m.edges[0, 0, 1] = 2  # head slot 1 -> node 2
    m.next_free_edge[0, 0] = 2
    write_value = np.zeros((1, 2))
    edge_1 = np.array([0])  # src = node 1
    edge_2 = np.array([1])  # dst = node 2
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.LINK], write_value, edge_1, edge_2)
    assert_equal("execute link success", ok, [True])
    assert_equal("execute link edge written", m.edges[0, 1, 0], 2)
    assert_equal("execute link counter incremented", m.next_free_edge[0, 1], 1)

    # ================================================================== TEST 17
    # execute: IDLE is a no-op but still returns True; mixed IDLE + CREATE preserves order
    print("TEST 17: execute IDLE and order")
    m = NP_Graph_Memory(num_batches=3, num_nodes=4, max_edges_per_node=2, node_dim=2)
    write_value = np.array([[1.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    OP = Graph_Memory_Operation_Type
    ok = m.execute([OP.CREATE, OP.IDLE, OP.CREATE], write_value, np.zeros(3, dtype=int), np.zeros(3, dtype=int))
    assert_equal("IDLE+CREATE order", ok, [True, True, True])
    assert_allclose("batch 0 created", m.nodes[0, 0], [1.0, 0.0])
    assert_equal("batch 1 idle untouched", m.next_free_node[1], 0)
    assert_allclose("batch 2 created", m.nodes[2, 0], [2.0, 0.0])

    print("\nAll tests passed.")
