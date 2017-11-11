from matplotlib import pyplot as plt
import numpy as np

tpr = [0.665,
0.67,
0.672,
0.68,
0.686,
0.69,
0.865,
0.894,
0.932,
0.872,
0.914,
0.95]

tpr.sort(reverse=True)

fpr = [0.334,
0.333,
0.328,
0.323,
0.314,
0.31,
0.135,
0.11,
0.07,
0.135,
0.11,
0.07]
fpr.sort(reverse=False)
plt.figure()
lw = 4
plt.plot(fpr, tpr, color='darkblue',
         lw=lw)#, #label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.xlim([0.0, 0.4])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined Receiver operating characteristic curve : A + B + C + D')
plt.legend(loc="lower right")
plt.show()