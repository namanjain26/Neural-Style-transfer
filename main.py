#importing libraries
import numpy as np
import scipy.misc
import scipy.io
import tensorflow as tf
from nst_utils import *

#input for content and style image 
content_image = scipy.misc.imread("images/dd-neckarfront.jpg")
style_image = scipy.misc.imread("images/Femme.jpg")
style_image= scipy.misc.imresize(style_image,content_image.shape)

content_image = reshape_and_normalize_image(content_image)
style_image = reshape_and_normalize_image(style_image)
generated_image = generate_noise_image(content_image)                           


# selecting layers for content and style  
layer_content = 'conv4_2' 
layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
layers_style_weights = [0.2,0.2,0.2,0.2,0.2]


#loading pretrained VGG-19 model
model= load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled =  tf.transpose(tf.reshape(a_C, [-1]))
    a_G_unrolled =  tf.transpose(tf.reshape(a_G, [-1]))
    J_content = tf.reduce_sum((a_C_unrolled - a_G_unrolled)**2) / (4 * n_H * n_W * n_C)
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))    
    return GA

def compute_layer_style_cost(a_S, a_G):                                    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S,[n_H*n_W, n_C])
    a_G = tf.reshape(a_G, [n_H*n_W, n_C])
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))
    J_style_layer = tf.reduce_sum((GS - GG)**2) / (4 * n_C**2 * (n_W * n_H)**2)
    return J_style_layer


STYLE_LAYERS = [('conv1_1', 0.2),('conv2_1', 0.2),('conv3_1', 0.2),('conv4_1', 0.2),('conv5_1', 0.2)]
def compute_style_cost(model, STYLE_LAYERS):                                  
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):                    # total cost function
    J = alpha*J_content+beta*J_style
    return J

#starting session 
sess = tf.InteractiveSession()
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = compute_content_cost(a_C, a_G)                                  #computing content cost


sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)                           #computing style cost


J = total_cost(J_content, J_style,  alpha = 10, beta = 40)                  #computing total cost


optimizer = tf.train.AdamOptimizer(2.0)                                     #running AdamOptimizer
train_step = optimizer.minimize(J)


def model_nn(sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        _ = sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%20 == 0:                                                    # after every 20 iterations image will be generated 
            Jt, Jc, Js = sess.run([J, J_content, J_style])               # and cost will be calculated
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output3/" + str(i) + ".png", generated_image)     
    
    save_image('output3/generated_image.jpg', generated_image)             # Final image will be generated and 
                                                                           # will be called as "generated_image.png"
    return generated_image

model_nn(sess, generated_image)    
