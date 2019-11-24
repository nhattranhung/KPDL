import re
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score

# read_File
fileTrain = open("D:/KPDL/data_train/agedetector_group_train.v1.0.txt", "r", encoding="utf-8")
num_lines_Train = sum(
    1 for lineTrain in open('D:/KPDL/data_train/agedetector_group_train.v1.0.txt', "r", encoding="utf-8"))

# Dictionary
label_Train = {}
label_Train['__label__18-24'] = {}
label_Train['__label__25-34'] = {}
label_Train['__label__35-44'] = {}
label_Train['__label__45-54'] = {}
label_Train['__label__55+'] = {}

st = ["", "", "", "", ""]  # String_label

tmp = ""
train_list = ['__label__18-24', '__label__25-34', '__label__35-44', '__label__45-54', '__label__55+']
for i in range(0, num_lines_Train):
    line = fileTrain.readline()
    line = line.split(" ", 1)
    tmp += line[1]
    if (re.search("18-24$", line[0])):
        st[0] += line[1]
    elif (re.search("25-34$", line[0])):
        st[1] += line[1]
    elif (re.search("35-44$", line[0])):
        st[2] += line[1]
    elif (re.search("45-54$", line[0])):
        st[3] += line[1]
    elif (re.search("55[+]$", line[0])):
        st[4] += line[1]

tmp = re.sub("\n", " ", tmp)
tmp = tmp.split(" ")
tmp = set(tmp)

for i in label_Train:
    for j in tmp:
        label_Train[i][j] = 0

for i in range(0, num_lines_Train):
    line = fileTrain.readline()
    line = line.split(" ", 1)

for i in range(0, len(st)):
    st[i] = re.sub("\n", " ", st[i])
    st[i] = st[i].split(" ")
    st[i] = list(st[i])
    for j in range(0, len(st[i])):
        if st[i][j] == "":
            del st[i][j]

index = 0
for i in label_Train:
    for j in st[index]:
        label_Train[i][j] += 1
    if index < 4:
        index += 1
    else:
        break

dtTrain = pd.DataFrame.from_dict(label_Train, orient='index')
dtTrain.to_csv(r'D:/KPDL/dt.csv', index=True)

###############################################################################################
num_lines_Test = sum(1 for lineTest in open('D:/KPDL/data_train/2.txt', "r", encoding="utf-8"))

dtTest = pd.DataFrame(columns=tmp)

with open("D:/KPDL/data_train/2.txt", "r", encoding="utf-8") as fileTest:
    for i in range(0, num_lines_Test):
        line = fileTest.readline()
        line = re.sub("\n", " ", line)
        line = line.split(" ")
        for j in range(0, len(line)):
            if line[j] == "":
                del line[j]
        vector = {}
        for j in tmp:
            vector[j] = 0
        for j in range(0, len(line)):
            if line[j] in tmp:
                vector[line[j]] += 1
        dtTest = dtTest.append(vector, ignore_index=True)

dtTest.to_csv(r'D:/KPDL/test.csv', index=False)

df_new = pd.read_csv('D:/KPDL/dt.csv', low_memory=False).drop(['Unnamed: 0'], axis=1)

train_x = df_new
train_y = train_list

test_x = dtTest

model = MultinomialNB()

# fit the model with the training data
model.fit(train_x, train_y)

# predict the target on the train dataset
predict_train = model.predict(train_x)
print('Target on train data', predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(train_y, predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(test_x)
print('Target on test data', predict_test)

fileW = open("D:/KPDL/data.TXT", "w", encoding="utf-8")
fileW.write(predict_test)

fileTrain.close()
fileTest.close()
