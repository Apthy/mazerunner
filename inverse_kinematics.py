import numpy as np
import torch
import matplotlib.pyplot as plt
#makes a number of angles and returns them

'''TEMP'''
def perceptron_output(x, wmat1, wmat2):
    x = np.array(x)
    w1 = wmat1
    w2 = wmat2
    xBias = np.ones((x.shape[0]+1,x.shape[1]))
    xBias[0:2, :] = x#adding bias
    output = np.zeros((x.shape[0],x.shape[1]))
    for i in range(x.shape[1]):
        net2 = np.dot(w1,xBias[:,i])

        a2 = 1/(1+np.exp(-net2))#sigmoid
        a2_aug = np.append(a2, 1)

        output[:,i] = np.dot(w2, a2_aug)
    return output



def perceptron(x, training,iterations,alpha, midnode):
    start_node, samples = x.shape
    output_node = training.shape[0]
    x=np.array(x)
    training = training
    xBias = np.ones((x.shape[0]+1,x.shape[1]))
    xBias[0:2,:] =x
    #torch.cat((x,torch.ones(samples).unsqueeze(0)),0)#bias added
    w1 = np.random.randn(midnode, start_node+1)
    w2 = np.random.randn(output_node, midnode + 1)

    eAvg=np.zeros((2,iterations))

    e=np.zeros((2,samples))
    for j in range(iterations):
        print(f'{100*j/iterations}% of iterations complete')
        for i in range(samples):
            #test = xBias[:,i]
            net2 = np.expand_dims(np.dot(w1,xBias[:,i]), axis=1)
            a2 = 1/(1+np.exp(-net2))
            #print(a2.shape[1]) only 1 dim
            a2_aug = np.expand_dims(np.append(a2,1), axis=1 )


            # sum of output now
            output =np.dot(w2,a2_aug)

            #backprop
            #print(f'training:{training[:,i]} and output{output}')
            e[:,i] = -(training[:,i]-np.squeeze(output,axis=1))
            delta3 =np.expand_dims(e[:,i],axis=1)

            w2noBias = w2[:,0:-1]#cut off the last row of the matrix
            delta2 = np.dot(w2noBias.T, delta3)
            delta2 = delta2 * a2 * (1 - a2)
            #delta3 = np.expand_dims(e[:, i], axis=1)
            #delta2 = np.expand_dims(delta2,axis=1)
            dedw1 = np.dot(delta2,np.expand_dims(xBias[:,i].T,axis=1).T)
            dedw2 = np.dot(delta3,a2_aug.T)
            w2 = w2-np.dot(alpha, dedw2)
            w1 = w1 - np.dot(alpha, dedw1)
        eAvg[:,j] = np.mean(e,1)
    return w1,w2,eAvg

def perceptronTorch(x, training):
    iterations = 100
    alpha = .2
    midnode = 50
    start_node, samples = x.shape
    output_node = training.shape[0]
    #X=torch.zeros((2,samples))
    x=torch.tensor(x)
    #x=np.array(x)
    training = torch.tensor(training)
    xBias = torch.ones((x.shape[0]+1,x.shape[1]))
    xBias[0:2,:] =x
    #torch.cat((x,torch.ones(samples).unsqueeze(0)),0)#bias added
    w1 = np.random.rand(midnode, start_node+1)
    w2 = np.random.rand(output_node, midnode + 1)
    eAvg=np.zeros((2,iterations))

    e=np.zeros((2,samples))
    for j in range(iterations):
        print(f'{100*j/iterations}% of iterations complete')
        for i in range(samples):
            test = xBias[:,i]
            net2 = np.dot(w1,xBias[:,i])
            a2 = 1/(1+np.exp(-net2))
            #print(a2.shape[1]) only 1 dim
            a2_aug = np.append(a2,1)


            # sum of output now
            output =np.dot(w2,a2_aug)

            #backprop
            #print(f'training:{training[:,i]} and output{output}')
            e[:,i] = -(training[:,i]-output)
            delta3 =e[:,i]

            w2noBias = w2[:,0:-1]#cut off the last row of the matrix
            delta2 = np.dot(w2noBias.T, delta3)
            delta3 = np.expand_dims(e[:, i], axis=1)
            delta2= delta2*a2*(1-a2)
            delta2 = np.expand_dims(delta2,axis=1)
            dedw1 = np.dot(delta2,np.expand_dims(xBias[:,i].T,axis=1).T)
            dedw2 = np.dot(delta3,np.expand_dims(a2_aug,axis=1).T)
            w2 = w2-np.dot(alpha, dedw2)
            w1 = w1 - np.dot(alpha, dedw1)
        eAvg[:,j] = np.mean(e,1)

    return w1,w2,eAvg

'''TEMP'''

def generate_data(samples, max_angle):
    rand_angles = torch.rand([2,samples])*max_angle
    #rand_angles2 = torch.rand([2, samples] * 180)
    return rand_angles

def forward_kinematics(angles, len1, len2, originx, originy):
    #outputs the position of a 2 joint arm when given the angles and length of the arms
    #it also calculates the midpoints to reduce the amount of iterations i have to do as we will be looping through the same thing
    limit = angles.shape[1]#get length
    outputx = np.zeros(limit)
    outputy = np.zeros(limit)
    mid_pointx = np.zeros(limit)
    mid_pointy = np.zeros(limit)
    for i in range(limit):
        mid_pointx[i] = originx + len1*np.cos(angles[0,i])
        mid_pointy[i] = originy + len1 * np.sin(angles[0, i])
        outputx[i] = originx+len1*np.cos(angles[0,i])+len2*np.cos(angles[0,i]+angles[1,i])
        outputy[i] = originy + len1 * np.sin(angles[0, i]) + len2 * np.sin(angles[0, i] + angles[1, i])
    return outputx, outputy, mid_pointx, mid_pointy
def forward_kinematics1(angles, len1, len2, originx, originy):
    #outputs the position of a 2 joint arm when given the angles and length of the arms
    #it also calculates the midpoints to reduce the amount of iterations i have to do as we will be looping through the same thing
    limit = 1
    outputx = np.zeros(limit)
    outputy = np.zeros(limit)
    mid_pointx = np.zeros(limit)
    mid_pointy = np.zeros(limit)

    mid_pointx[0] = originx + len1*np.cos(angles[0])
    mid_pointy[0] = originy + len1 * np.sin(angles[0])
    outputx[0] = originx+len1*np.cos(angles[0])+len2*np.cos(angles[0]+angles[1])
    outputy[0] = originy + len1 * np.sin(angles[0]) + len2 * np.sin(angles[0] + angles[1])
    return outputx, outputy, mid_pointx, mid_pointy

max_angle = np.pi
samples = 5000
angles = generate_data(samples, max_angle)

length1 = 0.5
length2 = 0.5
originX = 0
originY = 0
F_K_x, F_K_y, MPx, MPy = forward_kinematics(angles, length1, length2, originX, originY)
plt.scatter(F_K_x,F_K_y)
plt.show()
print('finished, now generating true scaled data')



angles = angles.numpy()
for i in range(samples):
    outx, outy = 5, 5
    while ((-0.8>=outx)or(outx>=-0.1)) or ((-0.3>=outy)or(outy>=0.4)):
        angles[:,i] = generate_data(2, max_angle)[0].numpy()
        outx,outy,_,_ =forward_kinematics1(angles[:, i], length1, length2, originX, originY)
    F_K_x[i]=outx
    F_K_y[i]=outy


plt.scatter(F_K_x,F_K_y)
plt.show()
print('plot')

alpha = .1
mid_node = 50
its = 50
#Ploted the points that a 2 joint arm can move
input_per = np.stack((F_K_x, F_K_y))
#debug
weight_mat_1 , weight_mat2, errors = perceptron(input_per, angles, 1, alpha, mid_node)
output_angles = perceptron_output(input_per, weight_mat_1, weight_mat2)
#F_K_x, F_K_y,_,_ =forwardkinematics(outputAngles,length1,length2,originX,originY)
plt.scatter(F_K_x,F_K_y,c='orange')
plt.scatter(F_K_x,F_K_y,c='blue')
plt.show()
#debug
weight_mat_1 , weight_mat2, errors = perceptron(input_per, angles, its, alpha, mid_node)
plt.plot(range(errors.shape[1]),errors[0,:])
plt.plot(range(errors.shape[1]),errors[1,:])
plt.title('a graph to show the loss')
plt.show()
output_angles = perceptron_output(input_per, weight_mat_1, weight_mat2)
F_K_x, F_K_y,_,_ =forward_kinematics(output_angles, length1, length2, originX, originY)
plt.scatter(input_per[0], input_per[1], c='orange')
plt.scatter(F_K_x,F_K_y,c='blue')
plt.show()
print('plot')
#Ploted the points that a 2 joint arm can move predicted by perceptron

