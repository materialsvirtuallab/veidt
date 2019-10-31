import unittest
from veidt.mcts.motifs import *
from veidt.mcts.reward import gbr_reward
import pandas as pd
import os

file_path = os.path.dirname(__file__)

class TestState(unittest.TestCase):
    def testInit(self):
        fids = [0, 1, 2, 3]
        avail_fids = [4, 5]
        state = State(fids=fids,
                      avail_fids=avail_fids,
                      current_run=1)
        self.assertEqual(state.fids, fids)
        self.assertEqual(state.avail_fids, avail_fids)
        self.assertEqual(state.current_run, 1)

    def testNextState(self):
        fids = [0, 1, 2, 3]
        avail_fids = [4]
        state = State(fids=fids,
                      avail_fids=avail_fids,
                      current_run=1)
        next_state = state.next_state()
        self.assertEqual(next_state.current_run, 2)
        self.assertTrue(next_state.fids == list(range(4)) or \
                        next_state.fids == list(range(5)))
        self.assertTrue(next_state.avail_fids == [4] or \
                        next_state.avail_fids == [])

    def testTerminal(self):
        state = State(fids=[1, 2, 3],
                      avail_fids=[],
                      current_run=1)
        self.assertTrue(state.is_terminal(3))
        state = State(fids=[1, 2, 3],
                      avail_fids=[1],
                      current_run=1)
        self.assertTrue(state.is_terminal(4) == False)


class TestNode(unittest.TestCase):
    def testInit(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3, 4, 5],
                      current_run=1)
        node = Node(state)
        self.assertTrue(node.visits == 1)
        self.assertTrue(node.reward == -node.MAX_MAE)
        self.assertTrue(node.children == [])
        self.assertTrue(node.parent == None)
        self.assertTrue(node.is_root == False)
        self.assertTrue(node.state.fids == [0, 1, 2])
        self.assertTrue(node.state.avail_fids == [3, 4, 5])
        self.assertTrue(node.state.current_run == 1)

    def testAddChild(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3],
                      current_run=1)
        child_state1 = State(fids=[0, 1, 2, 3],
                             avail_fids=[],
                             current_run=2)
        child_state2 = State(fids=[0, 1, 2, ],
                             avail_fids=[3],
                             current_run=2)
        node = Node(state)
        node.add_child(child_state1)
        node.add_child(child_state2)
        self.assertTrue(node.is_fully_expanded)
        self.assertTrue(node.children[0].state.fids == [0, 1, 2, 3])
        self.assertTrue(node.children[1].state.fids == [0, 1, 2])

    def testUpdate(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3],
                      current_run=1)
        node = Node(state)
        node.update(1)
        self.assertTrue(node.reward == -node.MAX_MAE + 1)
        self.assertTrue(node.visits == 2)

    def testRepr(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3],
                      current_run=1)
        node = Node(state)
        self.assertTrue(str(node) == 'visits: 1, reward: 0.000, Fid:  0,1,2')


class TestTreePolicy(unittest.TestCase):
    def testExpand(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3],
                      current_run=1)
        node = Node(state)
        child = EXPAND(node)
        self.assertTrue(child.state.fids == [0, 1, 2, 3])
        self.assertTrue(node.children[0].state.fids == [0, 1, 2, 3])
        self.assertTrue(node.children[1].state.fids == [0, 1, 2])
        self.assertTrue(node.is_fully_expanded)

    def testBestChild(self):
        node = Node(State([], [], 1))
        node.reward = -9
        node.visits = 10
        child1 = State([0], [], 1)
        child2 = State([1], [], 1)
        node.add_child(child1)
        node.add_child(child2)
        node.children[0].reward = -2.5
        node.children[0].visits = 5
        node.children[1].reward = 0
        node.children[1].visits = 1
        bestchild = BESTCHILD(node, -0.9)
        self.assertTrue(bestchild.state.fids == [0])
        bestchild = BESTCHILD(node, -0.4)
        self.assertTrue(bestchild.state.fids == [1])

    def testTreePolicy(self):
        node = Node(State([0], [1], 1))
        node.reward = -9
        node.visits = 10
        child1 = State([0, 1], [], 1)
        child2 = State([0], [1], 1)
        node.add_child(child1)
        node.add_child(child2)
        node.children[0].reward = -2.5
        node.children[0].visits = 5
        node.children[1].reward = 0
        node.children[1].visits = 1
        frontnode = TREEPOLICY(node, -0.9, k=6)
        self.assertTrue(str(frontnode) == 'visits: 5, reward: -2.500, Fid:  0,1')
        frontnode = TREEPOLICY(node, -0.4, k=6)
        # Now the bestchild of node is child2, but child2 is not terminal, one of its children is
        self.assertTrue(str(frontnode.parent) == 'visits: 1, reward: 0.000, Fid:  0')


class TestDefaultPolicy(unittest.TestCase):
    def testReward(self):
        data = pd.read_csv(os.path.join(file_path, "train.dat"), sep=' ')
        features = data.values[:, 2:]
        properties = data.property
        state = State(list(range(5)), [5], 1)
        setup = SetUp(features,
                      properties,
                      k=6,
                      scalar=-0.9)
        r6 = REWARD(state, setup)
        gbr_r6 = gbr_reward(features, properties)
        self.assertAlmostEqual(r6, gbr_r6, 3)

        state = State(list(range(3)), [], 1)
        setup = SetUp(features,
                      properties,
                      k=3,
                      scalar=-0.9)
        r3 = REWARD(state, setup)
        gbr_r3 = gbr_reward(features[:, :3], properties)
        self.assertTrue(r3 < gbr_r6)
        self.assertAlmostEqual(r3, gbr_r3, 3)

    def testDefaultPolicy(self):
        data = pd.read_csv(os.path.join(file_path, "train.dat"), sep=' ')
        features = data.values[:, 2:]
        properties = data.property
        setup = SetUp(features,
                      properties,
                      k=6,
                      scalar=-0.9)
        state = State([], list(range(6)), 1)
        gbr_r = gbr_reward(features, properties)
        r = DEFAULTPOLICY(state, setup)
        self.assertTrue(r <= gbr_r)


class TestBackUp(unittest.TestCase):
    def testBackup(self):
        state = State(fids=[0, 1, 2],
                      avail_fids=[3],
                      current_run=1)
        node = Node(state, is_root=True)
        node.visits = 2
        node.reward = -0.1
        frontnode = TREEPOLICY(node, scalar=1, k=6)
        BACKUP(frontnode, reward=1)
        self.assertTrue(node.visits == 3)
        self.assertTrue(node.reward == 0.9)
        self.assertTrue(frontnode.visits == 2)
        self.assertTrue(frontnode.reward == 1)


if __name__ == '__main__':
    unittest.main()
