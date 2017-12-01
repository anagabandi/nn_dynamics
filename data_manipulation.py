import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math
import matplotlib.pyplot as plt
import copy

def get_indices(which_agent):
    x_index = -7
    y_index = -7
    z_index = -7 
    yaw_index = -7
    joint1_index = -7 
    joint2_index = -7 
    frontleg_index = -7
    frontshin_index = -7
    frontfoot_index = -7
    xvel_index = -7
    orientation_index = -7

    if(which_agent==0): #pointmass
        x_index= 0
        y_index= 1
    elif(which_agent==1): #ant
        x_index= 29
        y_index= 30
        z_index = 31
        xvel_index = 38
    elif(which_agent==2): #swimmer
        x_index= 10
        y_index= 11
        yaw_index = 2
        joint1_index = 3
        joint2_index = 4
        xvel_index = 13
    elif(which_agent==3): #reacher
        x_index= 6
        y_index= 7
    elif(which_agent==4): #cheetah
        x_index= 18
        y_index= 20
        frontleg_index = 6
        frontshin_index = 7
        frontfoot_index = 8
        xvel_index = 21
    elif(which_agent==5): #roach (not mujoco)
        x_index= 0
        y_index= 1
    elif(which_agent==6): #hopper
        x_index = 11
        y_index = 13
        z_index = 0
        xvel_index = 14
        orientation_index = 1
    elif(which_agent==7): #walker
        x_index = 18
        y_index = 20

    return x_index, y_index, z_index, yaw_index, joint1_index, joint2_index, frontleg_index, \
            frontshin_index, frontfoot_index, xvel_index, orientation_index

def generate_training_data_inputs(states0, controls0):
    # init vars
    states=np.copy(states0)
    controls=np.copy(controls0)
    new_states=[]
    new_controls=[]

    # remove the last entry in each rollout (because that entry doesn't have an associated "output")
    for i in range(len(states)):
        curr_item = states[i]
        length = curr_item.shape[0]
        new_states.append(curr_item[0:length-1,:])

        curr_item = controls[i]
        length = curr_item.shape[0]
        new_controls.append(curr_item[0:length-1,:])
   
    #turn the list of rollouts into just one large array of data
    dataX= np.concatenate(new_states, axis=0)
    dataY= np.concatenate(new_controls, axis=0)
    return dataX, dataY

def generate_training_data_outputs(states, which_agent):
    #for each rollout, the output corresponding to each (s_i) is (s_i+1 - s_i)
    differences=[]
    for states_in_single_rollout in states:
        output = states_in_single_rollout[1:states_in_single_rollout.shape[0],:] \
                -states_in_single_rollout[0:states_in_single_rollout.shape[0]-1,:]
        differences.append(output)
    output = np.concatenate(differences, axis=0)
    return output

def from_observation_to_usablestate(states, which_agent, just_one):

    #######################################
    ######### POINTMASS ###################
    #######################################

    #0: x
    #1: y
    #2: vx
    #3: vy
    if(which_agent==0):
        return states

    #######################################
    ######### ANT #########################
    #######################################

    #we use the following observation as input to NN (41 things)
        #0 to 14... 15 joint positions
        #15 to 28... 14 joint velocities
        #29 to 31... 3 body com pos
        #32 to 37... 6 cos and sin of 3 body angles (from 9 rotation mat)
        #38 to 40... body com vel

    #returned by env.step
        #0 to 14 = positions
            #j0 x position
            #j1 y position
            #j2 z position
            #3 ?
            #4 5 body flip
            #6 body rotate
            #7 leg yaw ccw, 8 leg bend down
            #9, 10
            #11, 12
            #13,14 
        #15 to 28 = velocities
        #29 to 37 = rotation matrix (9)
        #38 to 40 = com positions
        #41 to 43 = com velocities

    if(which_agent==1):
        if(just_one):
            curr_item = np.copy(states)
            joint_pos = curr_item[0:15]
            joint_vel = curr_item[15:29]
            body_pos = curr_item[38:41]
            body_rpy = to_euler(curr_item[29:38], just_one) #9 vals of rot mat --> 6 vals (cos sin of rpy)
            body_vel = curr_item[41:44]
            full_item = np.concatenate((joint_pos, joint_vel, body_pos, body_rpy, body_vel), axis=0)
            return full_item

        else:
            new_states=[]
            for i in range(len(states)): #for each rollout
                curr_item = np.copy(states[i])

                joint_pos = curr_item[:,0:15]
                joint_vel = curr_item[:,15:29]
                body_pos = curr_item[:,38:41]
                body_rpy = to_euler(curr_item[:,29:38], just_one) #9 vals of rot mat --> 6 vals (cos sin of rpy)
                body_vel = curr_item[:,41:44]
                
                full_item = np.concatenate((joint_pos, joint_vel, body_pos, body_rpy, body_vel), axis=1)
                new_states.append(full_item)
            return new_states


    #######################################
    ######### SWIMMER #####################
    #######################################

    #total = 16
        #0 slider x... 1 slider y.... 2 heading
        #3,4 the two hinge joint pos
        #5,6 slider x/y vel
        #7 heading vel
        #8,9 the two hinge joint vel
        #10,11,12 cm x and y and z pos
        #13,14,15 cm x and y and z vel
    if(which_agent==2):
        return states

    #######################################
    ######### REACHER #####################
    #######################################

    #total = 11
        # 2-- cos(theta) of the 2 angles
        # 2-- sin(theta) of the 2 angles
        # 2-- goal pos -------------------(ignore this)
        # 2-- vel of the 2 angles
        # 3-- fingertip cm
    if(which_agent==3):
        if(just_one):
            curr_item = np.copy(states)
            keep_1 = curr_item[0:4]
            keep_2 = curr_item[6:11]
            full_item = np.concatenate((keep_1, keep_2), axis=0)
            return full_item

        else:
            new_states=[]
            for i in range(len(states)): #for each rollout
                curr_item = np.copy(states[i])
                keep1 = curr_item[:,0:4]
                keep2 = curr_item[:,6:11]
                full_item = np.concatenate((keep1, keep2), axis=1)
                new_states.append(full_item)
            return new_states

    #######################################
    ######### HALF CHEETAH ################
    #######################################

    #STATE when you pass in something to reset env: (33)
    #    rootx, rootz, rooty
    #    bthigh, bshin, bfoot
    #    fthigh, fshin, ffoot
    #    rootx, rootz, rooty --vel
    #    bthigh, bshin, bfoot --vel
    #    fthigh, fshin, ffoot --vel
    # self.model.data.qacc (9)
    # self.model.data.ctrl (6)
    #OBSERVATION: (24) 
    #    0: rootx (forward/backward)
    #    1: rootz (up/down)
    #    2: rooty (angle of body)
    #    3: bthigh (+ is move back)
    #    4: bshin
    #    5: bfoot
    #    6: fthigh
    #    7: fshin
    #    8: ffoot
    #    9: root x vel 
    #    10: root z vel 
    #    11: root y vel 
    #    12: bthigh vel
    #    13: bshin vel
    #    14: bfoot vel
    #    15: fthigh vel 
    #    16: fshin vel 
    #    17: ffoot vel 
    #com x
    #com y
    #com z
    #com vx
    #com vy
    #com vz

    if(which_agent==4):
        return states

    #######################################
    ######### ROACH (personal env) ########
    #######################################

        # x,y,z com position
        # orientation com
        # cos of 2 motor positions
        # sin of 2 motor positions
        # com velocity
        # orientation angular vel
        # 2 motor vel
    
    elif(which_agent==5):
        if(just_one):
            curr_item = np.copy(states)
            keep_1 = curr_item[0:6]
            two = np.cos(curr_item[6:8])
            three = np.sin(curr_item[6:8])
            keep_4 = curr_item[8:16]
            full_item = np.concatenate((keep_1, two, three, keep_4), axis=0)
            return full_item

        else:
            new_states=[]
            for i in range(len(states)): #for each rollout
                curr_item = np.copy(states[i])
                keep1 = curr_item[:,0:6]
                two = np.cos(curr_item[:,6:8])
                three = np.sin(curr_item[:,6:8])
                keep4 = curr_item[:,8:16]
                full_item = np.concatenate((keep1, two, three, keep4), axis=1)
                new_states.append(full_item)
            return new_states

    #######################################
    ######### HOPPER ######################
    #######################################

    #observation: 17 things
        #5 joints-- j0 (height), j2, j3, j4, j5
        #6 velocities
        #3 com pos
        #3 com vel
    #state: 21 things
        #6 joint pos
        #6 joint vel
        #6 qacc
        #3 ctrl

    if(which_agent==6):
        return states

    #######################################
    ######### WALKER ######################
    #######################################
    
    #observation: 24 things
        #9 joint pos
        #9 velocities
        #3 com pos
        #3 com vel

    if(which_agent==7):
        return states


def to_euler(rot_mat, just_one):
    if(just_one):
        r=np.arctan2(rot_mat[3], rot_mat[1])
        p=np.arctan2(-rot_mat[6], np.sqrt(rot_mat[7]*rot_mat[7]+rot_mat[8]*rot_mat[8]))
        y=np.arctan2(rot_mat[7], rot_mat[8])

        return np.array([np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)])

    else:
        r=np.arctan2(rot_mat[:,3], rot_mat[:,1])
        r=np.concatenate((np.expand_dims(np.cos(r), axis=1), np.expand_dims(np.sin(r), axis=1)), axis=1)

        p=np.arctan2(-rot_mat[:,6], np.sqrt(rot_mat[:,7]*rot_mat[:,7]+rot_mat[:,8]*rot_mat[:,8]))
        p=np.concatenate((np.expand_dims(np.cos(p), axis=1), np.expand_dims(np.sin(p), axis=1)), axis=1)

        y=np.arctan2(rot_mat[:,7], rot_mat[:,8])
        y=np.concatenate((np.expand_dims(np.cos(y), axis=1), np.expand_dims(np.sin(y), axis=1)), axis=1)

        return np.concatenate((r,p,y), axis=1)
