
import os
import random
from sign_model_3_out import *
from EEG_python_out import * 
import timeit 

# In[3]:

print('The model trained with the below Traffic Signs ')

display_images_and_labels(images, labels)

# In[4]:

print('Evaluation of model with some Sample samples randomly selected ')
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = session.run([predicted_labels], 
                        feed_dict={images_ph: sample_images})[0]
print(sample_labels)
print(predicted)

# In[5]:

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
sig_out=predicted[0]
sig_out_sample=predicted[i]
print(sig_out)
print(sig_out_sample)

display_label_images(images, sig_out)

# In[7]:

print('Driver curent EEG status')
print(x_test.sample(n=1))

eeg_out_sample=clf.predict(x_test.sample(n=1)) 


if eeg_out_sample[0]==0:
    print('Driver in active state')
else: 
    print('Driver in drowse  State')



# In[8]:

a=[10,33,56,21,22,30,40,55,28,39,45]
print(sig_out)

if eeg_out_sample[0]==1 and any ( [ x == sig_out for x in a ] ) :
    print ("take tha Action to Stop Vehicle ")
elif eeg_out_sample[0]==1 :
     print ("take tha Actions to Activate driver:start alarem ")
    
else:
    print("good status ",eeg_out[0] )