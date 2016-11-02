import csv
from sklearn import svm
from sklearn import tree

with open('decision_tree.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		Training_X = []
		Training_Y = []
		Test_X = []
		Test_Y = []
		for row in reader:
			if rownum == 0:
				header = row
				
			else:
				colnum = 0
				newlist = []
				for col in row:
					if header[colnum] == 'gname':
						y = col
					else:
						newlist.append(col)
					colnum += 1
				if rownum % 10 == 0:
					Test_X.append(newlist)
					Test_Y.append(y)
				else:
					Training_X.append(newlist)
					Training_Y.append(y)

			rownum += 1
		db.close()

clf = tree.DecisionTreeClassifier()
clf.fit(Training_X, Training_Y)

Predict_Y = clf.predict(Test_X)

colnum = 0
correct = 0
false = 0
for prediction in Predict_Y:
	if prediction == Test_Y[colnum]:
		correct += 1
	else:
		false += 1
	colnum += 1

print correct, false
