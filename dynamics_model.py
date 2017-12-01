
import numpy as np
import numpy.random as npr
import tensorflow as tf
import time
import math

from feedforward_network import feedforward_network


class Dyn_Model:

    def __init__(self, inputSize, outputSize, sess, learning_rate, batchsize, which_agent, x_index, y_index, 
                num_fc_layers, depth_fc_layers, mean_x, mean_y, mean_z, std_x, std_y, std_z, tf_datatype, print_minimal):

        #init vars
        self.sess = sess
        self.batchsize = batchsize
        self.which_agent = which_agent
        self.x_index = x_index
        self.y_index = y_index
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.mean_x = mean_x
        self.mean_y = mean_y
        self.mean_z = mean_z
        self.std_x = std_x
        self.std_y = std_y
        self.std_z = std_z
        self.print_minimal = print_minimal

        #placeholders
        self.x_ = tf.placeholder(tf_datatype, shape=[None, self.inputSize], name='x') #inputs
        self.z_ = tf.placeholder(tf_datatype, shape=[None, self.outputSize], name='z') #labels

        #forward pass
        self.curr_nn_output = feedforward_network(self.x_, self.inputSize, self.outputSize, 
                                                num_fc_layers, depth_fc_layers, tf_datatype)

        #loss
        self.mse_ = tf.reduce_mean(tf.square(self.z_ - self.curr_nn_output))

        # Compute gradients and update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.theta = tf.trainable_variables()
        self.gv = [(g,v) for g,v in
                    self.opt.compute_gradients(self.mse_, self.theta)
                    if g is not None]
        self.train_step = self.opt.apply_gradients(self.gv)

    def train(self, dataX, dataZ, dataX_new, dataZ_new, nEpoch, save_dir, fraction_use_new):

        #init vars
        start = time.time()
        training_loss_list = []
        range_of_indeces = np.arange(dataX.shape[0])
        nData_old = dataX.shape[0]
        num_new_pts = dataX_new.shape[0]

        #how much of new data to use per batch
        if(num_new_pts<(self.batchsize*fraction_use_new)):
            batchsize_new_pts = num_new_pts #use all of the new ones
        else:
            batchsize_new_pts = int(self.batchsize*fraction_use_new)

        #how much of old data to use per batch
        batchsize_old_pts = int(self.batchsize- batchsize_new_pts)

        #training loop
        for i in range(nEpoch):
            
            #reset to 0
            avg_loss=0
            num_batches=0

            #randomly order indeces (equivalent to shuffling dataX and dataZ)
            old_indeces = npr.choice(range_of_indeces, size=(dataX.shape[0],), replace=False)
            #train from both old and new dataset
            if(batchsize_old_pts>0): 

                #get through the full old dataset
                for batch in range(int(math.floor(nData_old / batchsize_old_pts))):

                    #randomly sample points from new dataset
                    if(num_new_pts==0):
                        dataX_new_batch = dataX_new
                        dataZ_new_batch = dataZ_new
                    else:
                        new_indeces = npr.randint(0,dataX_new.shape[0], (batchsize_new_pts,))
                        dataX_new_batch = dataX_new[new_indeces, :]
                        dataZ_new_batch = dataZ_new[new_indeces, :]

                    #walk through the randomly reordered "old data"
                    dataX_old_batch = dataX[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    dataZ_old_batch = dataZ[old_indeces[batch*batchsize_old_pts:(batch+1)*batchsize_old_pts], :]
                    
                    #combine the old and new data
                    dataX_batch = np.concatenate((dataX_old_batch, dataX_new_batch))
                    dataZ_batch = np.concatenate((dataZ_old_batch, dataZ_new_batch))

                    #one iteration of feedforward training
                    _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
                    training_loss_list.append(loss)
                    avg_loss+= loss
                    num_batches+=1

            #train completely from new set
            else: 
                for batch in range(int(math.floor(num_new_pts / batchsize_new_pts))):

                    #walk through the shuffled new data
                    dataX_batch = dataX_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]
                    dataZ_batch = dataZ_new[batch*batchsize_new_pts:(batch+1)*batchsize_new_pts, :]

                    #one iteration of feedforward training
                    _, loss, output, true_output = self.sess.run([self.train_step, self.mse_, self.curr_nn_output, self.z_], 
                                                                feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

                    training_loss_list.append(loss)
                    avg_loss+= loss
                    num_batches+=1

                #shuffle new dataset after an epoch (if training only on it)
                p = npr.permutation(dataX_new.shape[0])
                dataX_new = dataX_new[p]
                dataZ_new = dataZ_new[p]

            #save losses after an epoch
            np.save(save_dir + '/training_losses.npy', training_loss_list)
            if(not(self.print_minimal)):
                if((i%10)==0):
                    print("\n=== Epoch {} ===".format(i))
                    print ("loss: ", avg_loss/num_batches)
        
        if(not(self.print_minimal)):
            print ("Training set size: ", (nData_old + dataX_new.shape[0]))
            print("Training duration: {:0.2f} s".format(time.time()-start))

        #get loss of curr model on old dataset
        avg_old_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(nData_old / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ[batch*self.batchsize:(batch+1)*self.batchsize, :]
            #one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_old_loss+= loss
            iters_in_batch+=1
        old_loss =  avg_old_loss/iters_in_batch

        #get loss of curr model on new dataset
        avg_new_loss=0
        iters_in_batch=0
        for batch in range(int(math.floor(dataX_new.shape[0] / self.batchsize))):
            # Batch the training data
            dataX_batch = dataX_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = dataZ_new[batch*self.batchsize:(batch+1)*self.batchsize, :]
            #one iteration of feedforward training
            loss, _ = self.sess.run([self.mse_, self.curr_nn_output], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})
            avg_new_loss+= loss
            iters_in_batch+=1
        if(iters_in_batch==0):
            new_loss=0
        else:
            new_loss =  avg_new_loss/iters_in_batch

        #done
        return (avg_loss/num_batches), old_loss, new_loss

    def run_validation(self, inputs, outputs):

        #init vars
        nData = inputs.shape[0]
        avg_loss=0
        iters_in_batch=0

        for batch in range(int(math.floor(nData / self.batchsize))):
            # Batch the training data
            dataX_batch = inputs[batch*self.batchsize:(batch+1)*self.batchsize, :]
            dataZ_batch = outputs[batch*self.batchsize:(batch+1)*self.batchsize, :]

            #one iteration of feedforward training
            z_predictions, loss = self.sess.run([self.curr_nn_output, self.mse_], feed_dict={self.x_: dataX_batch, self.z_: dataZ_batch})

            avg_loss+= loss
            iters_in_batch+=1

        #avg loss + all predictions
        print ("Validation set size: ", nData)
        print ("Validation set's total loss: ", avg_loss/iters_in_batch)

        return (avg_loss/iters_in_batch)

    #multistep prediction using the learned dynamics model at each step
    def do_forward_sim(self, forwardsim_x_true, forwardsim_y, many_in_parallel, env_inp, which_agent):

        #init vars
        state_list = []

        if(many_in_parallel):
            #init vars
            N= forwardsim_y.shape[0]
            horizon = forwardsim_y.shape[1]
            array_stdz = np.tile(np.expand_dims(self.std_z, axis=0),(N,1))
            array_meanz = np.tile(np.expand_dims(self.mean_z, axis=0),(N,1))
            array_stdy = np.tile(np.expand_dims(self.std_y, axis=0),(N,1))
            array_meany = np.tile(np.expand_dims(self.mean_y, axis=0),(N,1))
            array_stdx = np.tile(np.expand_dims(self.std_x, axis=0),(N,1))
            array_meanx = np.tile(np.expand_dims(self.mean_x, axis=0),(N,1))

            if(len(forwardsim_x_true)==2):
                #N starting states, one for each of the simultaneous sims
                curr_states=np.tile(forwardsim_x_true[0], (N,1))
            else:
                curr_states=np.copy(forwardsim_x_true)

            #advance all N sims, one timestep at a time
            for timestep in range(horizon):

                #keep track of states for all N sims
                state_list.append(np.copy(curr_states))

                #make [N x (state,action)] array to pass into NN
                states_preprocessed = np.nan_to_num(np.divide((curr_states-array_meanx), array_stdx))
                actions_preprocessed = np.nan_to_num(np.divide((forwardsim_y[:,timestep,:]-array_meany), array_stdy))
                inputs_list= np.concatenate((states_preprocessed, actions_preprocessed), axis=1)

                #run the N sims all at once
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_list}) 
                state_differences = np.multiply(model_output[0],array_stdz)+array_meanz

                #update the state info
                curr_states = curr_states + state_differences

            #return a list of length = horizon+1... each one has N entries, where each entry is (13,)
            state_list.append(np.copy(curr_states))
        else:
            curr_state = np.copy(forwardsim_x_true[0]) #curr state is of dim NN input

            for curr_control in forwardsim_y:

                state_list.append(np.copy(curr_state))
                curr_control = np.expand_dims(curr_control, axis=0)

                #subtract mean and divide by standard deviation
                curr_state_preprocessed = curr_state - self.mean_x
                curr_state_preprocessed = np.nan_to_num(curr_state_preprocessed/self.std_x)
                curr_control_preprocessed = curr_control - self.mean_y
                curr_control_preprocessed = np.nan_to_num(curr_control_preprocessed/self.std_y)
                inputs_preprocessed = np.expand_dims(np.append(curr_state_preprocessed, curr_control_preprocessed), axis=0)

                #run through NN to get prediction
                model_output = self.sess.run([self.curr_nn_output], feed_dict={self.x_: inputs_preprocessed}) 

                #multiply by std and add mean back in
                state_differences= (model_output[0][0]*self.std_z)+self.mean_z

                #update the state info
                next_state = curr_state + state_differences

                #copy the state info
                curr_state= np.copy(next_state)

            state_list.append(np.copy(curr_state))
              
        return state_list