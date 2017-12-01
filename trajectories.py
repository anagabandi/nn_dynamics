import numpy as np

def make_trajectory(shape, starting_state_NN, x_index, y_index, which_agent):

    curr_x = np.copy(starting_state_NN[x_index])
    curr_y = np.copy(starting_state_NN[y_index])

    my_list = []

    if(shape=="left_turn"):
        if(which_agent==1):
            my_list.append(np.array([curr_x, curr_y]))
            my_list.append(np.array([curr_x+2, curr_y]))
            my_list.append(np.array([curr_x+4, curr_y]))
            my_list.append(np.array([curr_x+6, curr_y]))
            my_list.append(np.array([curr_x+6, curr_y+2]))
            my_list.append(np.array([curr_x+6, curr_y+3]))
            my_list.append(np.array([curr_x+6, curr_y+4]))
            my_list.append(np.array([curr_x+6, curr_y+5]))
            my_list.append(np.array([curr_x+6, curr_y+6]))
            my_list.append(np.array([curr_x+6, curr_y+7]))
        else:
            my_list.append(np.array([curr_x, curr_y]))
            my_list.append(np.array([curr_x+1, curr_y]))
            my_list.append(np.array([curr_x+2, curr_y]))
            my_list.append(np.array([curr_x+3, curr_y]))
            my_list.append(np.array([curr_x+4, curr_y+1]))
            my_list.append(np.array([curr_x+4, curr_y+2]))
            my_list.append(np.array([curr_x+4, curr_y+3]))
            my_list.append(np.array([curr_x+4, curr_y+4]))

    if(shape=="right_turn"):
        if(which_agent==1):
            my_list.append(np.array([curr_x, curr_y]))
            my_list.append(np.array([curr_x, curr_y+1]))
            my_list.append(np.array([curr_x, curr_y+2]))
            my_list.append(np.array([curr_x, curr_y+3]))
            my_list.append(np.array([curr_x, curr_y+4]))
            my_list.append(np.array([curr_x+2, curr_y+4]))
            my_list.append(np.array([curr_x+3, curr_y+4]))
            my_list.append(np.array([curr_x+4, curr_y+4]))
            my_list.append(np.array([curr_x+6, curr_y+4]))
            my_list.append(np.array([curr_x+7, curr_y+4]))
        else:
            my_list.append(np.array([curr_x, curr_y]))
            my_list.append(np.array([curr_x, curr_y+1]))
            my_list.append(np.array([curr_x, curr_y+2]))
            my_list.append(np.array([curr_x+2, curr_y+3]))
            my_list.append(np.array([curr_x+3, curr_y+3]))
            my_list.append(np.array([curr_x+4, curr_y+3]))
            my_list.append(np.array([curr_x+5, curr_y+3]))
            my_list.append(np.array([curr_x+6, curr_y+3]))
            my_list.append(np.array([curr_x+7, curr_y+3]))
            my_list.append(np.array([curr_x+8, curr_y+3]))

    if(shape=="u_turn"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y]))
        my_list.append(np.array([curr_x+4, curr_y+1]))
        my_list.append(np.array([curr_x+4, curr_y+2]))
        my_list.append(np.array([curr_x+2, curr_y+2]))
        my_list.append(np.array([curr_x+1, curr_y+2]))
        my_list.append(np.array([curr_x, curr_y+2]))

    if(shape=="straight"):
        i=0
        num_pts = 40
        while(i < num_pts):
            my_list.append(np.array([curr_x+i, curr_y]))
            i+=0.7

    if(shape=="backward"):
        i=0
        num_pts = 40
        while(i < num_pts):
            my_list.append(np.array([curr_x-i, curr_y]))
            i+=0.5

    if(shape=="forward_backward"):
        my_list.append(np.array([curr_x, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+3, curr_y]))
        my_list.append(np.array([curr_x+2, curr_y]))
        my_list.append(np.array([curr_x+1, curr_y]))
        my_list.append(np.array([curr_x+0, curr_y]))
        my_list.append(np.array([curr_x-1, curr_y]))
        my_list.append(np.array([curr_x-2, curr_y]))

    if(shape=="circle"):
        num_pts = 10
        radius=2.0
        speed=-np.pi/8.0
        for i in range(num_pts):
            curr_x= radius*np.cos(speed*i)-radius
            curr_y= radius*np.sin(speed*i)
            my_list.append(np.array([curr_x, curr_y]))

    return np.array(my_list)

def get_trajfollow_params(which_agent, desired_traj_type):

    desired_snake_headingInit= 0
    horiz_penalty_factor= 0
    forward_encouragement_factor= 0
    heading_penalty_factor= 0

    if(which_agent==1):
        if(desired_traj_type=="right_turn"):
            horiz_penalty_factor= 3
            forward_encouragement_factor= 50
            heading_penalty_factor= 100
        if(desired_traj_type=="left_turn"):
            horiz_penalty_factor= 4
            forward_encouragement_factor= 85
            heading_penalty_factor= 100
        if(desired_traj_type=="straight"):
            horiz_penalty_factor= 3.5
            forward_encouragement_factor= 85
            heading_penalty_factor= 100
        if(desired_traj_type=="u_turn"):
            horiz_penalty_factor= 3
            forward_encouragement_factor= 50
            heading_penalty_factor= 100

    if(which_agent==2):
        if(desired_traj_type=="right_turn"):
            desired_snake_headingInit= np.pi/2.0
            horiz_penalty_factor= 0.1 
            forward_encouragement_factor= 250
            heading_penalty_factor= 0.9
        if(desired_traj_type=="left_turn"):
            horiz_penalty_factor= 0.7
            forward_encouragement_factor= 200
            heading_penalty_factor= 0.9
        if(desired_traj_type=="straight"):
            horiz_penalty_factor= 4
            forward_encouragement_factor= 500
            heading_penalty_factor= 2

    if(which_agent==4):
        if(desired_traj_type=="backward"):
            horiz_penalty_factor= 0
            forward_encouragement_factor= 20
            heading_penalty_factor= 10
        if(desired_traj_type=="forward_backward"):
            horiz_penalty_factor= 0
            forward_encouragement_factor= 20
            heading_penalty_factor= 10
        if(desired_traj_type=="straight"):
            horiz_penalty_factor= 0
            forward_encouragement_factor= 40
            heading_penalty_factor= 10

    return horiz_penalty_factor, forward_encouragement_factor, heading_penalty_factor, desired_snake_headingInit