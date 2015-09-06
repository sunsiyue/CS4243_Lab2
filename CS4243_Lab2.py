

import cv2
import numpy as np
from matplotlib import pyplot as plt #for plotting cdf



pic_name = "editpic2.jpg"

pic_original = cv2.imread(pic_name,0) #read image

hist,bins = np.histogram(pic_original,256,[0,256])
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
pic_editted = cdf[pic_original]
hist1,bins1 = np.histogram(pic_editted,256,[0,256])

#cv2.imwrite("edit" + pic_name, pic_editted) #write image

##########plot the graph
cdf_normalized = cdf * hist1.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'g')
plt.hist(pic_editted.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf'), loc = 'upper right')
plt.title('editpic2')
plt.show()


