import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

red_patch = mpatches.Patch(color='red', label='No Hot Encoding')
blue_patch = mpatches.Patch(color='blue', label='One Hot Encoding')

no_ohe = [43.87755102040816, 64.75667189952904, 94.5839874411303, 79.90580847723704, 96.15384615384616, 81.71114599686028, 96.546310832]
ohe = [72.52747252747253, 64.99215070643642, 94.74097331240189, 60.28257456828885, 96.15384615384616, 87.67660910518053, 96.3108320251]
# Plot number of features VS. cross-validation scores
plt.figure()
plt.legend(handles=[red_patch, blue_patch], loc=4)

plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.bar(range(1, len(no_ohe) + 1), ohe, align='center')
plt.bar(range(1, len(no_ohe) + 1), no_ohe, align='center', color='red', width=0.6)
#plt.bar(range(1, len(ohe) + 1), ohe)
ax = plt.gca()
ax.set_xticklabels(["","SVM","GNB","ID3","KNN","RF","LR","Ensemble"])
plt.show()
#range(1, len(no_ohe) + 1)