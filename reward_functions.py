import numpy as np

class RewardFunctions:

    def __init__(self, which_agent, x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, 
                frontleg_index, frontshin_index, frontfoot_index, xvel_index, orientation_index):
        self.which_agent = which_agent
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index 
        self.yaw_index = yaw_index 
        self.joint1_index = joint1_index 
        self.joint2_index = joint2_index 
        self.frontleg_index = frontleg_index
        self.frontshin_index = frontshin_index 
        self.frontfoot_index = frontfoot_index 
        self.xvel_index = xvel_index 
        self.orientation_index = orientation_index 

    def get_reward_func(self, follow_trajectories, desired_states, horiz_penalty_factor, 
                        forward_encouragement_factor, heading_penalty_factor):

        #init vars
        self.desired_states= desired_states
        self.horiz_penalty_factor = horiz_penalty_factor
        self.forward_encouragement_factor = forward_encouragement_factor
        self.heading_penalty_factor = heading_penalty_factor

        if(follow_trajectories):
            if(self.which_agent==1):
                reward_func= self.ant_follow_traj
            if(self.which_agent==2):
                reward_func= self.swimmer_follow_traj
            if(self.which_agent==4):
                reward_func= self.cheetah_follow_traj
        else:
            if(self.which_agent==1):
                reward_func= self.ant_forward
            if(self.which_agent==2):
                reward_func= self.swimmer_forward
            if(self.which_agent==4):
                reward_func= self.cheetah_forward
            if(self.which_agent==6):
                reward_func= self.hopper_forward
        return reward_func

######################################################################################################################
    def ant_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                        curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        #penalize horiz dist away from trajectory
        scores[min_perp_dist<1] += (min_perp_dist*self.horiz_penalty_factor)[min_perp_dist<1]
        scores[min_perp_dist>=1] += (min_perp_dist*10*self.horiz_penalty_factor)[min_perp_dist>=1]

        #encourage moving-forward
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0]
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1]

        #prevent height from going too high or too low
        scores[pt[:,self.z_index]>0.67] += (self.heading_penalty_factor*40 + 0*pt[:,self.z_index])[pt[:,self.z_index]>0.67]
        scores[pt[:,self.z_index]<0.3] += (self.heading_penalty_factor*40 + 0*pt[:,self.z_index])[pt[:,self.z_index]<0.3]

        return scores, done_forever

    def swimmer_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                            curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        #penalize horiz dist away from trajectory
        scores += min_perp_dist*self.horiz_penalty_factor

        #encourage moving-forward and penalize not-moving-forward
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0]
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1]

        #angle that (desired traj) line segment makes WRT horizontal
        curr_line_start = self.desired_states[curr_seg]
        curr_line_end = self.desired_states[curr_seg+1]
        angle = np.arctan2(curr_line_end[:,1]-curr_line_start[:,1], curr_line_end[:,0]-curr_line_start[:,0]) 
            # ^ -pi to pi

        #penalize heading away from that angle
        diff = np.abs(pt[:,self.yaw_index]-angle)
        diff[diff>np.pi]=(2*np.pi-diff)[diff>np.pi] 
            #^ if the calculation takes you the long way around the circle, 
            #take the shorter value instead as the difference
        my_range = np.pi/3.0
        scores[diff<my_range] += (self.heading_penalty_factor*diff)[diff<my_range]
        scores[diff>=my_range] += 20

        #dont bend in too much
        first_joint = np.abs(pt[:,self.joint1_index])
        second_joint = np.abs(pt[:,self.joint2_index])
        limit = np.pi/3
        scores[limit<first_joint] += 2
        scores[limit<second_joint] += 2

        return scores, done_forever

    def cheetah_follow_traj(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                            curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        #penalize horiz dist away from trajectory
        scores += min_perp_dist*self.horiz_penalty_factor

        #encourage moving-forward
        scores[moved_to_next==0] -= self.forward_encouragement_factor*(curr_forward - prev_forward)[moved_to_next==0]
        scores[moved_to_next==1] -= self.forward_encouragement_factor*(curr_forward)[moved_to_next==1]

        #dont move front shin back so far that you tilt forward
        front_leg = pt[:,self.frontleg_index]
        my_range = 0.2
        scores[front_leg>=my_range] += self.heading_penalty_factor

        front_shin = pt[:,self.frontshin_index]
        my_range = 0
        scores[front_shin>=my_range] += self.heading_penalty_factor

        front_foot = pt[:,self.frontfoot_index]
        my_range = 0
        scores[front_foot>=my_range] += self.heading_penalty_factor

        return scores, done_forever

######################################################################################################################
    def ant_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        #watch the height
        done_forever[pt[:,self.z_index] > 1] = 1
        done_forever[pt[:,self.z_index] < 0.3] = 1

        #action
        scaling= 150.0
        if(pt_number==all_samples.shape[1]):
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number-1,:][done_forever==0]/scaling), axis=1)
        else:
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number,:][done_forever==0]/scaling), axis=1)

        #velocity
        scores[done_forever==0] -= pt[:,self.xvel_index][done_forever==0]

        #survival
        scores[done_forever==0] -= 0.5 # used to be 0.05

        return scores, done_forever

    def swimmer_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                        curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        ########### GYM

        '''if(pt_number==all_samples.shape[1]):
            reward_ctrl = 0.0001 * np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1)
        else:
            reward_ctrl = 0.0001 * np.sum(np.square(all_samples[:,pt_number,:]), axis=1)
        reward_fwd = (pt[:,self.x_index]-prev_pt[:,self.x_index]) / 0.01'''

        ########### RLLAB

        scaling=50.0
        if(pt_number==all_samples.shape[1]):
            reward_ctrl = 0.5 * np.sum(np.square(all_samples[:,pt_number-1,:]/scaling), axis=1)
        else:
            reward_ctrl = 0.5 * np.sum(np.square(all_samples[:,pt_number,:]/scaling), axis=1)
        reward_fwd = pt[:,self.xvel_index]

        #########################

        scores += -reward_fwd + reward_ctrl
        return scores, done_forever

    def cheetah_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward,
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        ########### GYM

        '''#action
        if(pt_number==all_samples.shape[1]):
            scores += 0.1*np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1)
        else:
            scores += 0.1*np.sum(np.square(all_samples[:,pt_number,:]), axis=1)

        #velocity
        scores -= (pt[:,self.x_index]-prev_pt[:,self.x_index]) / 0.01'''

        ########### RLLAB

        #action
        if(pt_number==all_samples.shape[1]):
            scores += 0.05*np.sum(np.square(all_samples[:,pt_number-1,:]), axis=1)
        else:
            scores += 0.05*np.sum(np.square(all_samples[:,pt_number,:]), axis=1)

        #velocity
        scores -= pt[:,self.xvel_index]

        return scores, done_forever

    def hopper_forward(self, pt, prev_pt, scores, min_perp_dist, curr_forward, prev_forward, 
                    curr_seg, moved_to_next, done_forever, all_samples, pt_number):

        scaling=200.0

        #dont tilt orientation out of range
        orientation = pt[:,self.orientation_index]
        done_forever[np.abs(orientation)>= 0.3] = 1

        #dont fall to ground
        done_forever[pt[:,self.z_index] <= 0.7] = 1

        #action
        if(pt_number==all_samples.shape[1]):
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number-1,:][done_forever==0]/scaling), axis=1)
        else:
            scores[done_forever==0] += 0.005*np.sum(np.square(all_samples[:,pt_number,:][done_forever==0])/scaling, axis=1)

        #velocity
        scores[done_forever==0] -= pt[:,self.xvel_index][done_forever==0]

        #survival
        scores[done_forever==0] -= 1

        return scores, done_forever