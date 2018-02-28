# In[1]:

import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

get_ipython().magic('matplotlib inline')


# In[2]:

def load_data(data_dir):
      
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
       
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels



ROOT_PATH = "C:/Users/grcks/Desktop/r&D/traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)


# In[3]:

print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))


# In[4]:

def display_images_and_labels(images, labels):
    
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()

display_images_and_labels(images, labels)



# In[5]:

def display_label_images(images, label):
    
    limit = 24  
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  
        plt.axis('off')
        i += 1
        plt.imshow(image)
    plt.show()

display_label_images(images, 32)


# In[6]:

for image in images[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


# In[7]:

# Resize images
images32 = [skimage.transform.resize(image, (32, 32),mode= 'reflect')
                for image in images]
display_images_and_labels(images32, labels)


# In[8]:

for image in images32[:5]:
    print("shape: {0}, min: {1}, max: {2}".format(image.shape, image.min(), image.max()))


# #  Model 3: with multi layer FC using leaky relu

# In[9]:

labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)


# In[10]:

def lrelu(x):
    return tf.maximum(0.01*x,x)


# In[11]:


graph = tf.Graph()


with graph.as_default():
   
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

   
    images_flat = tf.contrib.layers.flatten(images_ph)

    
    logits1 = tf.contrib.layers.fully_connected(images_flat, 200, lrelu)
    logits2 = tf.contrib.layers.fully_connected(logits1, 100, lrelu)
    logits = tf.contrib.layers.fully_connected(logits2, 62, lrelu)
    

    
    predicted_labels = tf.argmax(logits, 1)

   
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

   
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

   
    init = tf.global_variables_initializer()

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", predicted_labels)


# ## Training

# In[12]:


session = tf.Session(graph=graph)


_ = session.run([init])


# In[14]:

for i in range(201):
    _, loss_value = session.run([train, loss], 
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)


# ## Using the Model
# 
# 

# In[15]:


sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]


predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)


# In[16]:


fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Actual:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
sig_out=predicted[2]
print(sig_out)


# ## Evaluation
# 
# 

# In[17]:


test_images, test_labels = load_data(test_data_dir)


# In[18]:


test_images32 = [skimage.transform.resize(image, (32, 32),mode= 'reflect')
                 for image in test_images]
display_images_and_labels(test_images32, test_labels)


# In[19]:


predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: test_images32})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.3f}".format(accuracy))