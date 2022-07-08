#!/usr/bin/env python
# coding: utf-8

# Box plot representation of the different methods' accuracy

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


cnn_array = np.array([0.33603239, 0.44534413, 0.34817814, 0.34008097, 0.33198381,
       0.37246964, 0.31174089, 0.36032389, 0.3562753 , 0.35222672])
       
hog_sift_svm = np.array([0.34579439252336447,
 0.4046511627906977,
 0.41013824884792627,
 0.39814814814814814,
 0.3778801843317972,
 0.37962962962962965,
 0.39814814814814814,
 0.40825688073394495,
 0.42990654205607476,
 0.3581395348837209])
       
hog_svm = np.array([0.6088709677419355,
 0.6774193548387096,
 0.6411290322580645,
 0.5967741935483871,
 0.6370967741935484,
 0.625,
 0.6370967741935484,
 0.6370967741935484,
 0.6008064516129032,
 0.6129032258064516])
       
hog_cnn = np.array([0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,
 0.340080971659919,0.345])
      
surf_svm = np.array([0.3043097734,
       0.3204425768,
       0.3178380286,
       0.3670166946,
       0.3386626807,
       0.328019204 ,
       0.3267683208,
       0.3404907134,
       0.3185157099,
       0.3096389469])
       

cnn_array = cnn_array.reshape(-1,1)
hog_sift_svm = hog_sift_svm.reshape(-1,1)
hog_svm = hog_svm.reshape(-1,1)
hog_cnn = hog_cnn.reshape(-1,1)
surf_svm = surf_svm.reshape(-1,1)



NO_OF_BINS = 5
accuracy_matrix = np.hstack((cnn_array,hog_sift_svm, hog_svm, hog_cnn, surf_svm))

     
labelList = ['CNN', 'HOG+SIFT+SVM', 'HOG+SVM', 'HOG+CNN', 'SURF+SVM']
    

fig = plt.figure(1,figsize=(6.5, 4.5))
ax = fig.add_subplot(111)
plt.boxplot(accuracy_matrix,whis=5)
ax.set_xticklabels(labelList)
ax.set_ylabel('Engagement detection accuracy', fontsize=14)
ax.set_xlabel('Methods', fontsize=14)
ax.grid(True)
plt.tight_layout()
plt.savefig('./result.pdf', format='pdf')
plt.show()

