import numpy as np
import random
from sklearn.feature_selection import SelectKBest, f_regression
from veidt.mcts.reward import gbr_reward
import math


class SetUp():
    def __init__(self, features, properties, k, scalar, subsis=10):
        self.features = features
        self.properties = properties
        self.k = k
        self.scalar = scalar
        self.subsis = subsis


class State():
    MAX_RUNS = 100

    def __init__(self, fids, avail_fids, current_run):
        self.fids = fids
        self.avail_fids = avail_fids
        self.current_run = current_run

    def next_state(self):
        nextfid = random.choice(self.avail_fids)
        new_avail_fids = [i for i in self.avail_fids if i != nextfid]
        if random.uniform(0, 1) < 0.5:
            next = State(self.fids + [nextfid],
                         new_avail_fids,
                         self.current_run + 1)
        else:
            next = State(self.fids,
                         new_avail_fids,
                         self.current_run + 1)
        return next

    def is_terminal(self, k):
        if len(self.fids) == k or len(self.avail_fids) == 0:
            return True
        return False


class Node():
    MAX_MAE = 0

    def __init__(self, state, parent=None, is_root=False):
        self.visits = 1
        self.reward = -self.MAX_MAE
        self.state = state
        self.children = []
        self.parent = parent
        self.is_root = is_root

    def add_child(self, child_state):
        child = Node(child_state, parent=self)
        self.children.append(child)

    def update(self, reward):
        self.reward += reward  # TODO: take cumulative reward or max
        self.visits += 1

    @property
    def is_fully_expanded(self):
        if len(self.children) == 2:
            return True
        return False

    def __repr__(self):
        s = "visits: %d, reward: %.3f, Fid: " % (self.visits, self.reward)
        s = s + ' ' + ','.join([str(i) for i in self.state.fids])
        return s


def TREEPOLICY(node, scalar, k):
    while not node.state.is_terminal(k=k):
        if not node.is_fully_expanded:
            return EXPAND(node)
        else:
            node = BESTCHILD(node, scalar)
    return node


def EXPAND(node):
    nextfid = random.choice(node.state.avail_fids)
    nextstate1 = State(fids=node.state.fids + [nextfid],
                       avail_fids=[i for i in node.state.avail_fids if i != nextfid],
                       current_run=node.state.current_run + 1)
    nextstate2 = State(fids=node.state.fids,
                       avail_fids=[i for i in node.state.avail_fids if i != nextfid],
                       current_run=node.state.current_run + 1)
    node.add_child(nextstate1)
    node.add_child(nextstate2)

    return node.children[0]


def BESTCHILD(parent, C):
    bestchildren = []
    bestscore = -np.inf
    for c in parent.children:
        exploid = c.reward / c.visits
        explore = math.sqrt(2.0 * math.log(parent.visits) / float(c.visits))
        score = exploid + C * explore
        if score == bestscore:
            bestchildren.append(c)
        elif score > bestscore:
            bestchildren = [c]
            bestscore = score
    bestchild = random.choice(bestchildren)
    return bestchild


def DEFAULTPOLICY(state, setup):
    # last state's reward
    while not state.is_terminal(k=setup.k):
        state = state.next_state()
    return REWARD(state, setup)


def REWARD(state, setup):
    # num_fea = features.shape[1]
    # fidRemain = [i for i in range(num_fea) if i not in fidSubset]
    fidSubset = state.fids
    fidRemain = state.avail_fids
    if len(fidRemain) > setup.subsis:
        featureRemain_rand = random.choices(fidRemain, k=setup.subsis)
    else:
        featureRemain_rand = fidRemain
    featureSubset = setup.features[:, fidSubset + featureRemain_rand]
    filter_method = SelectKBest(f_regression, k=setup.k).fit_transform
    try:
        Fcandidate = filter_method(featureSubset, setup.properties)
    except ValueError:
        # k > len(Fcandidate)
        if featureSubset.shape[1]:
            Fcandidate = featureSubset

    simulation_reward = gbr_reward(Fcandidate, setup.properties)
    return simulation_reward


def BACKUP(node, reward):
    while node is not None:
        node.visits += 1
        # node.reward = reward if reward > node.reward else node.reward
        node.reward += reward
        node = node.parent


def UCTSEARCH(budget, root, setup):
    # budget: how many simulations to run in each UCTSearch
    for iter in range(budget):
        frontnode = TREEPOLICY(root, setup.scalar, setup.k)
        reward = DEFAULTPOLICY(frontnode.state, setup)
        BACKUP(frontnode, reward)
    return BESTCHILD(root, 0)
