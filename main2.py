import csv
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# Binomial Naive Bayes - Bag of words

# This function takes a csv file and prepares a new csv file with column names
# for example
# extract_columns_by_name('gname_indexed_known', ['eventid', 'gname', 'gname_index'], 'output', True)
# gname_indexed_known file will be opened. ['eventid', 'gname', 'gname_index'] named columns will be written to output
# flag is a switch. If it is sent as False, function would remove those columns specified in names
def extract_columns_by_name(infile, names, outfile, flag):
	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				records.append(header)
			else:
				colnum = 0
				newlist = []
				while colnum < len(header):
					col = row[colnum]
					if (header[colnum] in names and flag is True) or (header[colnum] not in names and flag is False):
						newlist.append(col)
					colnum += 1
				records.append(newlist)
		db.close()

	with open(outfile + '.csv', 'wb') as csvout:
		writer = csv.writer(csvout, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		new_header = []
		for col in header:
			if (col in names and flag is True) or (col not in names and flag is False):
				new_header.append(col)

		records[0] = new_header
		for row in records:
			writer.writerow(row)

# Pretty much the same function as upper one. This function, instead of names, takes column numbers as parameters 
def extract_columns_by_number(infile, numbers, outfile):
	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []
		names = []
		for row in reader:
			if rownum == 0:
				colnum = 1
				newlist = []
				for col in row:
					if colnum in numbers:
						names.append(col)
					colnum += 1
				rownum += 1
				records.append(row)
			else:
				colnum = 1
				newlist = []
				for col in row:
					if colnum in numbers:
						newlist.append(col)
					colnum += 1
				records.append(newlist)
		db.close()

	with open(outfile + '.csv', 'wb') as csvout:
	    writer = csv.writer(csvout, delimiter=',',
	                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    records[0] = names
	    for row in records:
	    	writer.writerow(row)

# An interesting function. This is a simple filtering function for row and column values
# Lets say, we only want the Unknown incidents. 
# extract_rows_by_key_value('db', {'gname':'Unknown'}, False, 'knowndb')
# If we send True parameter instead of False, it would only select the rows with gname = Unknown values.
def extract_rows_by_key_value(infile, dict_values, filt, outfile):
	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []
		header = []
		number = 0
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				records.append(header)
			else:
				flag = True
				colnum = 0
				for col in row:
					current_key = header[colnum]
					if current_key in dict_values.keys():
						if str(dict_values.get(current_key)) == col:
							if filt is False:
								flag = False
								break
						else:
							if filt is True:
								flag = False
								break
					colnum += 1
				if flag:
					records.append(row)
		db.close()

	with open(outfile + '.csv', 'wb') as csvout:
	    writer = csv.writer(csvout, delimiter=',',
	                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    for row in records:
	    	writer.writerow(row)

# This function gives integer numbers to features.
def index_features(infile, features, outfile):
	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []
		header = []
		indices = {}
		for feature in features:
			indices[feature] = {}
		number = 0
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				records.append(header)
			else:
				fallback = 0
				colnum = 0
				while True:
					col = row[colnum]
					current_key = header[colnum - fallback]
					if current_key in features:
						if col in indices.get(current_key).keys():
							new_index = indices.get(current_key)[col]
							row.insert(colnum + 1, new_index)
						else:
							new_index = len(indices.get(current_key).keys())
							indices.get(current_key)[col] = new_index
							row.insert(colnum + 1, new_index)
						colnum += 2
						fallback += 1
					else:
						colnum += 1

					if colnum >= len(row):
						break
				records.append(row)
				
		colnum = 0
		while True:
			current_key = header[colnum]
			if current_key in features:
				header.insert(colnum + 1, current_key + '_index')
				colnum += 2
			else:
				colnum += 1

			if colnum == len(header):
						break
		db.close()
		records[0] = header

	with open(outfile + '.csv', 'wb') as csvout:
	    writer = csv.writer(csvout, delimiter=',',
	                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    for row in records:
	    	writer.writerow(row)
def threshold_year(infile, threshold_lower, threshold_upper, outfile):

	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []
		names = []
		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				colnum = 0
				while colnum < len(header):
					if header[colnum] == 'iyear':
						year_index = colnum
						break
					colnum += 1
				records.append(header)
			elif int(row[year_index]) >= threshold_lower and int(row[year_index]) <= threshold_upper:
				records.append(row)

	with open(outfile + '.csv', 'wb') as csvout:
	    writer = csv.writer(csvout, delimiter=',',
	                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    for row in records:
	    	writer.writerow(row)
	db.close()

def take_most_groups(infile, number_of_groups, outfile):
	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		gnames = {}

		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				colnum = 0
				while colnum < len(header):
					if header[colnum] == 'gname':
						gname_index = colnum
						break
					colnum += 1
			else:
				if header[gname_index] == 'gname':
					if row[gname_index] not in gnames.keys():
						gnames[ row[gname_index] ] = 1
					else:
						gnames[ row[gname_index] ] += 1
				rownum +=1
		total = 0
		for key in gnames.keys():
			total += gnames[key]

		most_frequent_groups = []
		for a_gname in sorted(gnames, key=gnames.get, reverse=True):
			if len(most_frequent_groups) > number_of_groups:
				break
			else:
				most_frequent_groups.append(a_gname)
		db.close()

	with open(infile + '.csv', 'rb') as db:
		reader = csv.reader(db)
		rownum = 0
		records = []

		for row in reader:
			if rownum == 0:
				header = row
				rownum += 1
				colnum = 0
				while colnum < len(header):
					if header[colnum] == 'gname':
						gname_index = colnum
						break
					colnum += 1
				records.append(header)

			elif row[gname_index] in most_frequent_groups:
				records.append(row)


	with open(outfile + '.csv', 'wb') as csvout:
	    writer = csv.writer(csvout, delimiter=',',
	                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
	    for row in records:
	    	writer.writerow(row)
	db.close()

def accuracy(Test_Y, Predict_Y):
	colnum = 0
	correct = 0.
	false = 0.

	for prediction in Predict_Y:

		if prediction == Test_Y[colnum]:
			correct += 1
		else:
			false += 1

		colnum += 1

	return (correct / (correct+false))*100

####### START OF PROGRAM ##############

indexed_features = ['iyear', 'extended', 'country', 'region', 'specificity', 'vicinity', 'crit1', 'crit2', 'crit3', 'doubtterr', 'alternative', 'multiple', 'success', 'suicide', 'attacktype1', 'attacktype2', 'attacktype3', 'targtype1', 'targsubtype1', 'natlty1', 'targtype2', 'targsubtype2', 'natlty2', 'targtype3', 'targsubtype3', 'natlty3', 'gname', 'guncertain1', 'guncertain2', 'guncertain3', 'claimed', 'claimmode', 'claim2', 'claimmode2', 'claim3', 'claimmode3', 'compclaim', 'weaptype1', 'weapsubtype1', 'weaptype2', 'weapsubtype2', 'weaptype3', 'weapsubtype3', 'weaptype4', 'weapsubtype4', 'property', 'propextent', 'ishostkid', 'ransom', 'hostkidoutcome', 'INT_LOG', 'INT_IDEO', 'INT_MISC', 'INT_ANY']

new_input = raw_input("Do you wanna take new inputs")
if new_input == '1':

	features_in = raw_input("Write new features(Default ones are: imonth,longitude,latitude,attacktype1,targtype1,weaptype1): ")

	if features_in == "":
		features_in = "imonth,longitude,latitude,attacktype1,targtype1,weaptype1"

	features = features_in.split(",")
	empty_features = {}
	for feature in features:
		empty_features[feature] = ''

	features.append('iyear')
	features.append('gname')

	number_of_groups = raw_input("Select most frequent group number(Default 10): ")
	if number_of_groups == "":
		number_of_groups = 10
	else:
		number_of_groups = int(number_of_groups)

	#extract_columns_by_name('db', features, 'cleandb', True)
	extract_rows_by_key_value('db', empty_features, False, 'cleandb')
	extract_rows_by_key_value('cleandb', {'gname':'Unknown'}, False, 'cleandb')
	extract_rows_by_key_value('cleandb', {'gname':'Other'}, False, 'cleandb')
	extract_rows_by_key_value('cleandb', {'gname':'Individual'}, False, 'cleandb')

	extract_columns_by_name('cleandb', features, 'cleandb', True)
	threshold_year('cleandb', 2000, 2011, 'cleandb')
	take_most_groups('cleandb', number_of_groups, 'cleandb')
	
	threshold_year('cleandb', 2000, 2010, 'cleandb_train')
	extract_columns_by_name('cleandb_train', ['iyear'], 'cleandb_train', False)
	threshold_year('cleandb', 2011, 2011 , 'cleandb_test')
	extract_columns_by_name('cleandb_test', ['iyear'], 'cleandb_test', False)

with open('cleandb_train.csv', 'rb') as db:
	reader = csv.reader(db)
	rownum = 0
	Training_H = []
	Training_X = []
	Training_Y = []
	for row in reader:
		if rownum == 0:
			header = row
			Training_H = header
		else:
			colnum = 0
			newlist = []
			for col in row:
				if header[colnum] == 'gname':
					y = col
				else:
					"""
					try:
						if header[colnum] in indexed_features and float(col) < 0:
							newlist.append(1000)
						else:
							newlist.append(float(col))
					except:
						newlist.append(float(1001))
						"""
					newlist.append(float(col))
				colnum += 1

			Training_X.append(newlist)
			Training_Y.append(y)

		rownum += 1
	db.close()

Training_H.remove('gname')

with open('cleandb_test.csv', 'rb') as db:
	reader = csv.reader(db)
	rownum = 0
	Test_H = []
	Test_X = []
	Test_Y = []
	for row in reader:
		if rownum == 0:
			header = row
			Test_H = header
		else:
			colnum = 0
			newlist = []
			for col in row:
				if header[colnum] == 'gname':
					y = col
				else:
					"""
					try:
						if header[colnum] in indexed_features and float(col) < 0:
							newlist.append(1000)
						else:
							newlist.append(float(col))
					except:
						newlist.append(float(1001))
						"""
					newlist.append(float(col))
				colnum += 1

			Test_X.append(newlist)
			Test_Y.append(y)

		rownum += 1
	db.close()

Test_H.remove('gname')

Training_H_Mask = []
i = 0
for feature in Training_H:
	
	if feature in indexed_features:
		Training_H_Mask.append(i)

	i+=1


encoder = preprocessing.OneHotEncoder(categorical_features = Training_H_Mask, n_values='auto', handle_unknown='ignore')
#encoder.fit(Training_X)
"""
try:
	Training_X_Encoded = encoder.transform(Training_X).toarray()
	Test_X_Encoded = encoder.transform(Test_X).toarray()
except:
	Training_X_Encoded = np.ndarray.tolist(encoder.transform(Training_X))
	Test_X_Encoded = np.ndarray.tolist(encoder.transform(Test_X))
"""
"""
Training_X_Encoded = encoder.transform(Training_X).toarray()
Test_X_Encoded = encoder.transform(Test_X).toarray()
"""

Training_X_Encoded = Training_X
Test_X_Encoded = Test_X

print Training_X[0]
print Training_X_Encoded[0]

classifier = int(raw_input(
"""
Enter number corresponding to classifier you would like to use:
1. Support Vector Machines
2. Gaussian Naive Bayes
3. ID3 Decision Tree
4. KNN
5. Random Forest
6. Ensemble of all 
"""))

if classifier == 1: # Support vector machines
	clf = SVC()
elif classifier == 2: # Gaussian Naive Bayes
	clf = GaussianNB()
elif classifier == 3:
	clf = tree.DecisionTreeClassifier(criterion = "entropy")
elif classifier == 4:
	clf = KNeighborsClassifier(n_neighbors = 5, algorithm = 'kd_tree')
elif classifier == 5:
	clf = RandomForestClassifier(n_estimators = 20, criterion = "entropy")
elif classifier == 6:
	clf_SVC = SVC()
	clf_GNB = GaussianNB()
	clf_ID3 = tree.DecisionTreeClassifier(criterion = "entropy")
	clf_KNN = KNeighborsClassifier(n_neighbors = 5, algorithm = 'kd_tree')
	clf_RFC = RandomForestClassifier(n_estimators = 20, criterion = "entropy")

	clf_SVC.fit(Training_X_Encoded, Training_Y)
	clf_GNB.fit(Training_X_Encoded, Training_Y)
	clf_ID3.fit(Training_X_Encoded, Training_Y)
	clf_KNN.fit(Training_X_Encoded, Training_Y)
	clf_RFC.fit(Training_X_Encoded, Training_Y)

	Predict_Y_SVC = clf_SVC.predict(Test_X_Encoded)
	Predict_Y_GNB = clf_GNB.predict(Test_X_Encoded)
	Predict_Y_ID3 = clf_ID3.predict(Test_X_Encoded)
	Predict_Y_KNN = clf_KNN.predict(Test_X_Encoded)
	Predict_Y_RFC = clf_RFC.predict(Test_X_Encoded)

	accuracy_SVC = accuracy(Test_Y, Predict_Y_SVC)
	accuracy_GNB = accuracy(Test_Y, Predict_Y_GNB)
	accuracy_ID3 = accuracy(Test_Y, Predict_Y_ID3)
	accuracy_KNN = accuracy(Test_Y, Predict_Y_KNN)
	accuracy_RFC = accuracy(Test_Y, Predict_Y_RFC)
	accuracies = [accuracy_SVC, accuracy_GNB, accuracy_ID3, accuracy_KNN, accuracy_RFC]
	print accuracies

	Predict_Y = []

	for index in range(len(Test_Y)):
		votes = [Predict_Y_SVC[index], Predict_Y_GNB[index], Predict_Y_ID3[index], Predict_Y_KNN[index], Predict_Y_RFC[index]]
		vote_counter = {}
		for sub_index in range(len(votes)):
			vote = votes[sub_index]
			if vote in vote_counter:
				vote_counter[vote] += accuracies[sub_index]
			else:
				vote_counter[vote] = accuracies[sub_index]

		most_votes = sorted(vote_counter, key = vote_counter.get, reverse = True)
		most_vote = most_votes[0]
		Predict_Y.append(most_vote) 


if classifier != 6:
	clf.fit(Training_X_Encoded, Training_Y)
	if classifier == 3:
		print clf.feature_importances_
	Predict_Y = clf.predict(Test_X_Encoded)

print accuracy(Test_Y, Predict_Y)