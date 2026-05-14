import numpy as np
import math

# seed random number generator
np.random.seed(0)


class VNMobSim:

    def __init__(self, mobility_model):
        self.m_model = mobility_model

        # Getting the required parameters
        if self.m_model == 'Freeway':
            print('Pleas provide the required information')
            self.num_lanes = int(input('Number of Lanes: '))
            self.lane_length = int(input('Length of the lanes in meters: '))
            self.lane_gap = int(input('Lane Gap: '))
            self.nv_ps = int(input('Maximum number of vehicles entering a lane per time slot: '))
            self.v_max = float(input('Maximum Speed in meter/sec: '))
            self.v_min = float(input('Minimum Speed in meter/sec: '))

            while self.v_min > self.v_max:
                print('Minimum Speed must be less than or equal to Maximum Speed')
                self.v_min = float(input('Please re-enter Minimum Speed(meter/sec) correctly: '))

            self.m_max = int(input('Maximum Queue Size of Vehicles: '))
            self.cl_max = int(input('Maximum Communication Queue Size of Vehicles: '))
            self.del_t = float(input('Simulation time step duration: '))
            self.dec_t = float(input('Decision period duration: '))
            self.sig_psi = float(input('Enter shadow fading variance: '))
            self.Td = float(input('Enter time deadline of tasks: '))
            self.userIndex = 1.0
            self.userUtil = 0
            self.Os = 0.01
            self.Is = 7.312
            self.r_cell = 5
            self.r_dd = 16
            self.done = 0

            # Initializing the first node
            self.index_count = 1
            self.index_count_thresh = 2000
            self.v_pos = np.concatenate(
                ([np.random.rand(1) * self.v_max * self.del_t],
                 [np.random.randint(self.num_lanes, size=1) + 1],
                 [[self.index_count]],
                 np.zeros((1, 1), dtype=int),
                 np.zeros((1, self.m_max)),
                 np.zeros((1, self.m_max)),
                 np.zeros((1, self.m_max)),
                 np.zeros((1, 1), dtype=int),
                 np.zeros((1, self.cl_max)),
                 np.zeros((1, self.cl_max)),
                 np.zeros((1, self.cl_max)),
                 np.zeros((1, self.cl_max)),
                 np.zeros((1, self.cl_max)),
                 np.zeros((1, 1))), axis=1)

    # Setting the User Node
    def set_user_node_manual(self, user_idx):
        self.userIndex = user_idx

    # Sets the last node in the node list as the user node
    def set_user_node_auto(self):
        self.userIndex = self.v_pos[-1, 2]

    # Updating the processing rates
    def update_processing_rates(self, lu, li):
        self.v_pos[:, -1] = (self.del_t/self.dec_t)*np.random.poisson(li, self.v_pos.shape[0])
        user_row = np.where(self.v_pos[:, 2] == self.userIndex)
        self.v_pos[user_row[0][0], -1] = (self.del_t/self.dec_t)*np.random.poisson(lu, 1)

    def task_enqueue(self, v_idx, t_rem, util):
        v_row = np.where(self.v_pos[:, 2] == v_idx)
        if self.v_pos[v_row, 3] < self.m_max:
            self.v_pos[v_row, 3] += 1
            self.v_pos[v_row, (3 + int(self.v_pos[v_row, 3]))] = 1
            self.v_pos[v_row, (3 + self.m_max + int(self.v_pos[v_row, 3]))] = t_rem
            self.v_pos[v_row, (3 + (2*self.m_max) + int(self.v_pos[v_row, 3]))] = util

    def comm_enqueue(self, v_idx, data_size, t_rem, is_v2v_broken, dest_idx, util):
        v_row = np.where(self.v_pos[:, 2] == v_idx)
        if self.v_pos[v_row, (4 + (3*self.m_max))] < self.cl_max:
            self.v_pos[v_row, (4 + (3 * self.m_max))] += 1
            self.v_pos[v_row, (4 + int(self.v_pos[v_row, (4 + (3 * self.m_max))]) + (3 * self.m_max))] = data_size
            self.v_pos[v_row, (4 + int(self.v_pos[v_row, (4 + (3 * self.m_max))]) + (3 * self.m_max)) + self.cl_max] = \
                t_rem
            self.v_pos[v_row, (4 + int(self.v_pos[v_row, (4 + (3 * self.m_max))]) + (3 * self.m_max)) +
                       (2 * self.cl_max)] = is_v2v_broken
            self.v_pos[v_row, (4 + int(self.v_pos[v_row, (4 + (3 * self.m_max))]) + (3 * self.m_max)) +
                       (3 * self.cl_max)] = dest_idx
            self.v_pos[v_row, (4 + int(self.v_pos[v_row, (4 + (3 * self.m_max))]) + (3 * self.m_max)) +
                       (4 * self.cl_max)] = util

    def task_dequeue(self, v_idx):
        v_row = np.where(self.v_pos[:, 2] == v_idx)
        if self.v_pos[v_row, 3] > 0:
            self.v_pos[v_row, 3] -= 1
            print(self.v_pos[v_row, 4:(4+self.m_max)])
            print(self.v_pos[v_row, 3])
            self.v_pos[v_row, 4:(4+self.m_max)] = np.concatenate((self.v_pos[v_row, 5:(4+self.m_max)], [0]))
            self.v_pos[v_row, (4 + self.m_max):(4 + (2*self.m_max))] = \
                np.concatenate((self.v_pos[v_row, (5 + self.m_max):(4 + (2*self.m_max))], [0]))
            self.v_pos[v_row, (4 + (2*self.m_max)):(4 + (3*self.m_max))] = \
                np.concatenate((self.v_pos[v_row, (5 + (2*self.m_max)):(4 + (3*self.m_max))], [0]))

    def comm_dequeue(self, v_idx, t_idx):
        t_idx = int(t_idx)
        v_row = np.where(self.v_pos[:, 2] == v_idx)
        if self.v_pos[v_row, (4 + (3*self.m_max))] >= t_idx:
            self.v_pos[v_row, (4 + (3 * self.m_max))] -= 1
            self.v_pos[v_row, ((4 + (3 * self.m_max)) + t_idx):(5 + (3 * self.m_max) + self.cl_max)] = \
                np.concatenate((self.v_pos[v_row, ((5 + (3 * self.m_max)) + t_idx):(5 + (3 * self.m_max)
                                                                                    + self.cl_max)], [0]))
            self.v_pos[v_row, ((4 + (3 * self.m_max)) + self.cl_max + t_idx):(5 + (3 * self.m_max)
                                                                                + (2 * self.cl_max))] = \
                np.concatenate((self.v_pos[v_row, ((5 + (3 * self.m_max)) + self.cl_max + t_idx):(5 + (3 * self.m_max)
                                                                             + (2 * self.cl_max))], [0]))
            self.v_pos[v_row, ((4 + (3 * self.m_max)) + (2 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                    + (3 * self.cl_max))] = \
                np.concatenate(
                    (self.v_pos[v_row, ((5 + (3 * self.m_max)) + (2 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                             + (3 * self.cl_max))],
                     [0]))
            self.v_pos[v_row, ((4 + (3 * self.m_max)) + (3 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                    + (4 * self.cl_max))] = \
                np.concatenate(
                    (self.v_pos[v_row, ((5 + (3 * self.m_max)) + (3 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                             + (4 * self.cl_max))],
                     [0]))
            self.v_pos[v_row, ((4 + (3 * self.m_max)) + (4 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                    + (5 * self.cl_max))] = \
                np.concatenate(
                    (self.v_pos[v_row, ((5 + (3 * self.m_max)) + (4 * self.cl_max) + t_idx):(5 + (3 * self.m_max)
                                                                                             + (5 * self.cl_max))],
                     [0]))

    # Returns the number of dequeues required for vehicle with position in v_pos given by v_idx
    def process_tasks(self, v_idx):
        i = 0
        v_idx = int(v_idx)
        p = self.v_pos[v_idx, -1]
        while p > 0 and i < self.v_pos[v_idx, 3]:
            temp = self.v_pos[v_idx, 4+i]
            self.v_pos[v_idx, 4 + i] = max(self.v_pos[v_idx, 4+i] - p, 0)
            p -= min(temp, p)
            if self.v_pos[v_idx, 4 + i] == 0:
                i += 1
        return i

    def update_task_queues_sim_step(self):

        # Nodes with non-empty task queues
        vwt = np.where(self.v_pos[:, 3] > 0)[0]

        for i in vwt:

            # Updating the remaining times of node i
            self.v_pos[i, (4 + self.m_max):(5 + self.m_max + int(self.v_pos[i, 3]))] -= self.del_t

            num_dq = self.process_tasks(i)
            for j in range(num_dq):
                if self.v_pos[i, 2] == self.userIndex:
                    if self.v_pos[i, (4 + self.m_max + j)] > 0:
                        self.userUtil += self.v_pos[i, 4 + (2 * self.m_max) + j]
                    self.task_dequeue(self.v_pos[i, 2])
                else:
                    self.comm_enqueue(self.v_pos[i, 2], self.Os, self.v_pos[i, (4 + self.m_max + j)],
                                      1, self.userIndex, self.v_pos[i, 4 + (2 * self.m_max) + j])
                    self.task_dequeue(self.v_pos[i, 2])

    # Returns the index of comm tasks that must be dequeued for the vehicle with position in v_pos given by v_idx
    def processed_comm_tasks(self, v_idx):
        re_idx = []
        for i in range(int(self.v_pos[v_idx, 4+(3*self.m_max)])):
            if self.v_pos[v_idx, 5+(3*self.m_max)+i] > 0:
                re_idx.append((i+1))
        return np.array(re_idx)

    # Returns the physical distance between two vehicles
    def phy_dist(self, v1_idx, v2_idx):
        v1_row = np.where(self.v_pos[:, 2] == v1_idx)
        v2_row = np.where(self.v_pos[:, 2] == v2_idx)
        return (((self.v_pos[v1_row, 0] - self.v_pos[v2_row, 0]) ** 2) + (
                    (self.lane_gap * (self.v_pos[v1_row, 1] - self.v_pos[v2_row, 1])) ** 2)) * 0.5

    # Returns 1 if the link is broken for a given tx-rx pair at a distance x
    def link_state_update(self, x):

        c_sig = 1.5912 * np.exp((self.sig_psi**2)/18.86)
        n_parts = 3000
        a = np.ones((1, n_parts)) * 10
        b = np.linspace(-90, 0.1, n_parts) * 0.1
        b.reshape(1, n_parts)

        if x < 50:
            m = 3
        elif (x >= 50) and (x < 150):
            m = 2
        else:
            m = 1

        c = np.power(a, (m*b))
        d = np.power(a, b)*(-m)*(x**3)/c_sig
        e = np.exp(d)
        f = np.multiply(c, e)
        f = f.flatten()
        g = ((((m*(x**3)/c_sig)**m)/math.factorial(int(m-1)))*0.23)*np.trapz(f, dx=(90.1/(n_parts-1)))
        return np.random.choice(np.array([0, 1]), 1, replace=True, p=np.concatenate((g, 1-g), axis=1).flatten())

    def update_comm_queues_sim_step(self):

        # Nodes with non-empty communication queues
        vwt = np.where(self.v_pos[:, 4+(3*self.m_max)] > 0)[0]

        for i in vwt:

            # Updating the remaining data sizes
            link_mat = np.zeros((2, int(self.v_pos[i, 4 + (3 * self.m_max)])))
            link_mat[0, :] = self.v_pos[i, (5+(3*self.m_max)+(2*self.cl_max)):
                                           (5+(3*self.m_max)+(2*self.cl_max)+int(self.v_pos[i, 4+(3*self.m_max)]))]
            link_mat[1, :] = 1 - link_mat[0, :]
            rate_vec = np.zeros((1, 2))
            rate_vec[0, 0] = self.r_cell
            rate_vec[0, 1] = self.r_dd
            self.v_pos[i, (5+(3*self.m_max)):(5+(3*self.m_max)+int(self.v_pos[i, 4+(3*self.m_max)]))] -= \
                (np.dot(rate_vec, link_mat).flatten())

            # Updating the remaining times
            self.v_pos[i, (5 + (3 * self.m_max) + self.cl_max):(
                        5 + (3 * self.m_max) + self.cl_max + int(self.v_pos[i, 4 + (3 * self.m_max)]))] -= self.del_t

            # Dequeueing the finished tasks
            # Task list to be dequeued
            dt_list = self.processed_comm_tasks(i)
            for j in dt_list:
                if (int(self.v_pos[i, 4 + (3 * self.m_max) + (3 * self.cl_max) + j]) == -2) and (
                        self.v_pos[i, 4 + (3 * self.m_max) + self.cl_max + j] > 0):
                    self.userUtil += self.v_pos[i, 4+(3*self.m_max)+(4*self.cl_max)+j]
                    self.comm_dequeue(self.v_pos[i, 2], j)
                if self.v_pos[i, 4 + (3 * self.m_max) + (3 * self.cl_max) + j] != -2:
                    self.task_enqueue(self.v_pos[i, 4 + (3 * self.m_max) + (3 * self.cl_max) + j],
                                      self.v_pos[i, 4 + (3 * self.m_max) + self.cl_max + j],
                                      self.v_pos[i, 4 + (3 * self.m_max) + (4 * self.cl_max) + j])

            # Updating the V2V link status
            for j in range(int(self.v_pos[i, 4+(3*self.m_max)])):
                if int(self.v_pos[i, 5+(3*self.m_max)+(2*self.cl_max)+j]) == 0:
                    iv_dist = self.phy_dist(self.v_pos[i, 2], self.v_pos[i, 5+(3*self.m_max)+(3*self.cl_max)+j])
                    self.v_pos[i, 5+(3*self.m_max)+(2*self.cl_max)+j] = self.link_state_update(iv_dist)

    # Updating the position of current nodes (Freeway Model)
    def fw_curr_update(self):
        # Current number of vehicles
        curr_nodes = self.v_pos.shape[0]
        # Update the location of the current vehicles if any
        if curr_nodes > 0:
            self.v_pos[:, 0] += (((self.v_max - self.v_min) * np.random.rand(curr_nodes)) + self.v_min) * self.del_t

    # Adding new nodes (Freeway Model)
    def fw_add_new_nodes(self):
        na = np.random.randint(self.nv_ps + 1, size=self.num_lanes)
        for i in range(self.num_lanes):
            nv = np.ones((na[i], (6 + (3 * self.m_max) + (5 * self.cl_max))))
            nv[:, 0] = np.random.rand(na[i]) * self.v_max * self.del_t                          # X-position
            nv[:, 1] *= (i + 1)                                                                 # Lane Number
            nv[:, 2] = (np.arange(na[i], dtype=float) + 1 + self.index_count)                   # Vehicle Index Number
            nv[:, 3] = 0                                                        # Current number of tasks in task queue
            nv[:, 4:(4+self.m_max)] = np.zeros((na[i], self.m_max))                             # Remaining Portions
            nv[:, (4 + self.m_max):(4 + (2*self.m_max))] = np.zeros((na[i], self.m_max))    # Remaining Time
            nv[:, (4 + (2*self.m_max)):(4 + (3*self.m_max))] = np.zeros((na[i], self.m_max))   # Utility if successful
            nv[:, (4 + (3*self.m_max))] = 0                         # Current size of communication queue
            # Data Size Remaining (Parallel Communication assumed)
            nv[:, (5 + (3*self.m_max)):(5 + (3*self.m_max) + self.cl_max)] = np.zeros((na[i], self.cl_max))

            # Time Remaining (For elements of Communication task queue)
            nv[:, (5 + (3*self.m_max) + self.cl_max):(5 + (3*self.m_max) + (2*self.cl_max))] = \
                np.zeros((na[i], self.cl_max))

            # Is V2V broken (1 if broken)
            nv[:, (5 + (3 * self.m_max) + (2 * self.cl_max)):(5 + (3 * self.m_max) + (3 * self.cl_max))] = \
                np.zeros((na[i], self.cl_max))

            # Destination Node Index
            nv[:, (5 + (3 * self.m_max) + (3 * self.cl_max)):(5 + (3 * self.m_max) + (4 * self.cl_max))] = \
                np.zeros((na[i], self.cl_max))

            # Utility gained if successful
            nv[:, (5 + (3 * self.m_max) + (4 * self.cl_max)):(5 + (3 * self.m_max) + (5 * self.cl_max))] = \
                np.zeros((na[i], self.cl_max))

            # Latest processing rate per simulation time slot
            nv[:, (5 + (3 * self.m_max) + (5 * self.cl_max))] = 0

            if self.index_count >= self.index_count_thresh:
                self.index_count = 1
            else:
                self.index_count += na[i]

            self.v_pos = np.append(self.v_pos, nv, axis=0)

    # Drop the nodes that have exceeded bounds (Freeway Model)
    def fw_drop_nodes(self):
        curr_nodes = self.v_pos.shape[0]
        i = 0
        while i < curr_nodes:
            if self.v_pos[i, 0] > self.lane_length:
                self.v_pos = np.delete(self.v_pos, i, 0)
                curr_nodes = curr_nodes - 1
            else:
                i = i + 1

    # Add new nodes, update position of current nodes, delete nodes if needed
    def step(self, s_type, action):
        if self.m_model == 'Freeway':
            if s_type == 'SimStep':

                # Updating the position of current nodes
                self.fw_curr_update()

                # Adding new nodes
                self.fw_add_new_nodes()

                # Updating the task queues
                self.update_task_queues_sim_step()

                # Updating the communication queues
                self.update_comm_queues_sim_step()

                # Drop the nodes that have exceeded bounds
                self.fw_drop_nodes()

            if s_type == 'DecStep':

                # Adding tasks to appropriate nodes
                for i in range(np.shape(action)[0]):

                    # Sending processed results only
                    if action[i, 1] == 1:
                        self.task_enqueue(action[i, 0], self.Td, action[i, 2])

                    # Sending raw sensor data only
                    if action[i, 1] == 2:
                        self.comm_enqueue(action[i, 0], self.Is, self.Td, 0, self.userIndex, action[i, 2])

                # Updating the position of current nodes
                self.fw_curr_update()

                # Adding new nodes
                self.fw_add_new_nodes()

                # Updating the task queues
                self.update_task_queues_sim_step()

                # Updating the communication queues
                self.update_comm_queues_sim_step()

                # Drop the nodes that have exceeded bounds
                self.fw_drop_nodes()

                # Updating the processing rate of nodes (lu, li)
                self.update_processing_rates(2, 1)


s1 = VNMobSim('Freeway')
for z in range(1000):
    if z < 5:
        s1.step('SimStep', np.array([[0]]))
    else:
        if z % 5 == 0:
            s1.step('DecStep', np.array([[s1.v_pos[1, 2], 2, 100], [s1.v_pos[1, 2], 1, 200]]))
        else:
            s1.step('SimStep', np.array([[0]]))
print(s1.v_pos)
