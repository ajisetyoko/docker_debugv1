# @Author: ajisetyoko <simslab-cs>
# @Date:   2020-01-12T00:07:19+08:00
# @Email:  aji.setyoko.second@gmail.com
# @Last modified by:   simslab-cs
# @Last modified time: 2020-03-03T16:04:19+08:00



import numpy as np

class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,
                                                                         11),
                             (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                             (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'sign_lang_body':
            self.num_node = 15
            self_link = [(i,i) for i in range(self.num_node)]
            neighbor_link = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),
                            (1,8),(8,9),(9,10),(0,11),(0,12),(11,13),(12,14)]
            self.edge = self_link+neighbor_link 
            self.center = 0
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'sign_lang':
            self.num_node = 137
            self_link    = [(i, i) for i in range(self.num_node)]
            #Face _original index
            outer_link   = [(i,i+1) for i in range(16)] #1
            mouth1_link  = [(i,i+1) for i in range(48,59)] + [(59,48)] #2
            mouth2_link  = [(i,i+1) for i in range(60,67)] + [(67,60)] #3
            mstache_link = [(i,i+1) for i in range(31,35)] #4
            nose_link    = [(27,28),(28,29),(29,30)] #5
            eyebrw_link1 = [(i,i+1) for i in range(17,21)] #6
            eyebrw_link2 = [(i,i+1) for i in range(22,26)] #7
            eye_link1    = [(i,i+1) for i in range(36,41)] + [(41,36)] #8
            eye_link2    = [(i,i+1) for i in range(42,47)] + [(47,42)] #9
            #Hand _original index
            thumb_link   = [(0,1),(1,2),(2,3),(3,4)]
            index_link   = [(0,5),(5,6),(6,7),(7,8)]
            midle_link   = [(0,9),(9,10),(10,11),(11,12)]
            ring_link    = [(0,13),(13,14),(14,15),(15,16)]
            litle_link   = [(0,17),(17,18),(18,19),(19,20)]
            #Face and hand real representation
            face_link_ori = outer_link + mouth1_link + mouth2_link + mstache_link + nose_link + eyebrw_link1 + eyebrw_link2 + eye_link1 + eye_link2
            hand_right_ori= thumb_link + index_link + midle_link + ring_link + litle_link
            hand_left_ori = thumb_link + index_link + midle_link + ring_link + litle_link
            face_link  = [(i+25,j+25) for (i,j) in face_link_ori]
            hand_left  = [(i+95,j+95) for (i,j) in hand_left_ori]
            hand_right = [(i+116,j+116) for (i,j) in hand_right_ori]
            #Pose Body link
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            # face_link     = [(i,i) for i in range(70)]
            # self.edge = face_link_ori + face_link # Train Face only
            self.edge = self_link + neighbor_link + face_link + hand_left + hand_right
            self.center = 30 # For Face
            # self.center = 0
        elif layout == 'ntu-rgb+d2':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                              (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                              (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                              (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                              (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout=='costum1':
            self.num_node = 18#25
            self_link =[]
            for j in range(self.num_node):
                for i in range(self.num_node):
                    self_link.append((i,j))
            self.edge = self_link
            self.center = 21-1
        elif layout =='sbu':
            self.num_node = 15
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(0,1),(1,2),(1,3),(3,4),(4,5),(8,7),(7,6),
                            (6,1),(14,13),(13,12),(12,2),(11,10),
                            (10,9),(9,2),(9,3),(12,6)]
            # neighbor_link = [(0,1),(1,2),(2,3),(3,4),(4,5),(8,7),(7,6),
            #                 (6,2),(14,13),(13,12),(12,6),(12,2),(11,10),
            #                 (10,9),(9,3),(9,2)]
            self.edge = self_link + neighbor_link
            self.center = 2
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'pam':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
