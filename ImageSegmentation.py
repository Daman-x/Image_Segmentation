#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[24]:


img = cv2.imread("C:\\Users\\Daman\\Desktop\\img.jpg")
img.shape


# In[31]:


x,y,z = img.shape


# In[26]:


y


# In[27]:


plt.imshow(img)


# In[28]:


img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# In[29]:


plt.imshow(img)


# In[30]:


scaling = 50
width = int((img.shape[1] * scaling)/100)
height = int((img.shape[0] * scaling)/100)
img = cv2.resize(img,(width,height))


# In[32]:


img.shape


# In[33]:


img =img.reshape((-1,3))


# In[34]:


img.shape


# In[35]:


dominant_color = 5


# In[36]:


model = KMeans(dominant_color)


# In[37]:


model.fit(img)


# In[38]:


new_img = np.zeros((x*y,z),dtype="uint8")


# In[39]:


#plt.imshow(new_img)
new_img.shape


# In[40]:


color = model.cluster_centers_


# In[41]:


color = np.array(color,dtype = 'uint8')
color


# In[42]:


for i in range(img.shape[0]):
    new_img[i] = color[model.labels_[i]]


# In[43]:


new_img = np.reshape(new_img,(x,y,z))


# In[44]:


plt.imshow(new_img)


# In[45]:


new_img = cv2.cvtColor(new_img,cv2.COLOR_RGB2BGR)
cv2.imwrite("C:\\Users\\Daman\\Desktop\\image.jpg",new_img)


# In[ ]:




