
# coding: utf-8

# In[1]:


import numpy as np
import re
import pandas as pd
from math import exp, log
import matplotlib.pyplot as plt

DATA = 150
ALPHA = 0.01
EPOCH = 100

x1 = np.empty(DATA)
x2 = np.empty(DATA)
x3 = np.empty(DATA)
x4 = np.empty(DATA)
kelas = np.empty(DATA, dtype='S30')

fakta1 = np.empty(DATA, dtype='int32')
fakta2 = np.empty(DATA, dtype='int32')
fakta3 = np.empty(DATA, dtype='int32')

fakta1[0:50] = 1
fakta1[50:150] = 0
fakta2[0:50] = 0
fakta2[50:100] = 1
fakta2[100:150] = 0
fakta3[0:100] = 0
fakta3[100:150] = 1

#print(fakta1)
#print(fakta2)
#print(fakta3)

# In[2]:

def get_data(dt,grp):
	f = open('iris.txt','r')
	f1 = f.readlines()
	p = re.compile(r'([\d\.]+),([\d\.]+),([\d\.]+),([\d\.]+),([\w\-]+)')
	for i in range(DATA):
		dt[i] = p.match(f1[i]).group(grp)
	f.close()
	return dt

# In[5]:

def signoid(x):
    return 1 / (1 + exp(-x))

def softmax(x):
    return exp(x)

def table_count(data):
    return data.iloc[:,0].size

def theta_bias_initiation():
    theta1 = np.array([[0.88, 0.36, 0.03, 0.57], 
                       [0.02, 0.93, 0.54, 0.32],
                       [0.57, 0.03, 0.99, 0.88]])
    bias1 = np.array([0.39,0.93,0.27])

    theta2 = np.array([[0.82, 0.3, 0.64], 
                       [0.65, 0.2, 0.52],
                       [0.7, 0.55, 0.18]])
    bias2 = np.array([0.94,0.56,0.55])

    return [theta1, bias1, theta2, bias2]


# In[6]:


def back_prop(training,params):
    theta1 = np.array(params[0])
    bias1 = np.array(params[1])
    theta2 = np.array(params[2])
    bias2 = np.array(params[3])
    
    error = 0
    
    #print( table_count(training)

    for i in range(table_count(training)):
        #print( theta1
        #print( theta2
    
        h11 = sum(training.iloc[i,4:8]*theta1[0,0:4]) + bias1[0]
        h12 = sum(training.iloc[i,4:8]*theta1[1,0:4]) + bias1[1]
        h13 = sum(training.iloc[i,4:8]*theta1[2,0:4]) + bias1[2]
        #print( h11
        #print( h12
        #print( h13
        s11 = signoid(h11)
        s12 = signoid(h12)
        s13 = signoid(h13)
        #print( s11
        #print( s12
        #print( s13
    
        h21 = s11 * theta2[0][0] + s12 * theta2[0][1] + s13 * theta2[0][2] + bias2[0]
        h22 = s11 * theta2[1][0] + s12 * theta2[1][1] + s13 * theta2[1][2] + bias2[1]
        h23 = s11 * theta2[2][0] + s12 * theta2[2][1] + s13 * theta2[2][2] + bias2[2]

        #print( h21
        #print( h22
        #print( h23

        s_total = softmax(h21) + softmax(h22) + softmax(h23)
        s21 = softmax(h21) / s_total
        s22 = softmax(h22) / s_total
        s23 = softmax(h23) / s_total

        #print(s21+s22+s23)

        #print( s21
        #print( s22
        #print( s23

        error = error + -(training.iloc[i,0] * log(s21) + training.iloc[i,1] * log(s22) + training.iloc[i,2] * log(s23))
        #print(error)
    
        #error1 = error1 + (s21 - training.iloc[i,0])**2
        #error2 = error2 + (s22 - training.iloc[i,1])**2
        #error3 = error3 + (s23 - training.iloc[i,2])**2
        #print( error1
        #print( error2
        #print( error3
    
        tau21 = s21 - training.iloc[i,0]
        dw21 = tau21 * s11
        dw24 = tau21 * s12
        dw27 = tau21 * s13
        #print( tau21, dw21, dw24, dw27
    
        tau22 = s22 - training.iloc[i,1]
        dw22 = tau22 * s11
        dw25 = tau22 * s12
        dw28 = tau22 * s13
        #print( tau22, dw22, dw24
        
        tau23 = s23 - training.iloc[i,2]
        dw23 = tau23 * s11
        dw26 = tau23 * s12
        dw29 = tau23 * s13
        #print( tau22, dw22, dw24
    
        tau11 = (tau21 * theta2[0,0] + tau22 * theta2[1,0] + tau23 * theta2[2,0]) * (1-s11) * s11
        dw11 = tau11 * training.iloc[i,4]
        dw14 = tau11 * training.iloc[i,5]
        dw17 = tau11 * training.iloc[i,6]
        dw110 = tau11 * training.iloc[i,7]
        #print( tau11, dw11, dw14, dw17, dw110

        tau12 = (tau21 * theta2[0,1] + tau22 * theta2[1,1] + tau23 * theta2[2,1]) *(1-s12)*s12
        dw12 = tau12 * training.iloc[i,4]
        dw15 = tau12 * training.iloc[i,5]
        dw18 = tau12 * training.iloc[i,6]
        dw111 = tau12 * training.iloc[i,7]
        #print( tau12, dw12, dw15, dw18, dw111
        
        tau13 = (tau21 * theta2[0,2] + tau22 * theta2[1,2] + tau23 * theta2[2,2]) * (1-s13) * s13
        dw13 = tau13 * training.iloc[i,4]
        dw16 = tau13 * training.iloc[i,5]
        dw19 = tau13 * training.iloc[i,6]
        dw112 = tau13 * training.iloc[i,7]
        #print( tau13, dw13, dw16, dw19, dw112
    
        dw2 = np.array([[dw21, dw24, dw27],
                        [dw22, dw25, dw28],
                        [dw23, dw26, dw29]])
        #print( dw2
        dbias2 = np.array([tau21, tau22, tau23])
        #print( dbias2
        dw1 = np.array([[dw11, dw14, dw17, dw110],
                        [dw12, dw15, dw18, dw111],
                        [dw13, dw16, dw19, dw112]])
        #print( dw1
        dbias1 = np.array([tau11, tau12, tau13])
        #print( dbias1
        
        theta1 = theta1 - ALPHA * dw1
        bias1 = bias1 - ALPHA * dbias1
        theta2 = theta2 - ALPHA * dw2
        bias2 = bias2 - ALPHA * dbias2
        
    
    error = error/table_count(training)
    #print(error)
    #error = np.mean(error)
    #print( error
    
    return np.array([[error],[theta1, bias1, theta2, bias2]])


# In[7]:


#def predict(signoid):
#    if(signoid < 0.5):
#        return 0
#    else:
#        return 1
    
# In[8]:


def checking_accuration(data1,data2):
    checking = np.zeros(table_count(data1), dtype=bool)
    #print (data1)
    #print (data2)
    #print( checking
    for i in range(table_count(data1)):
        checking[i] = np.array_equal(data1.iloc[i,:], data2.iloc[i,:])
        #print (data1.iloc[i,:])
        #print (data2.iloc[i,:])
        #print (checking[i])
    
    return checking

def predict(table):
	prediction = pd.DataFrame(columns=['prediksi 1','prediksi 2','prediksi 3'])
	for i in range(table_count(table)):
		value = np.max(table.iloc[i])
		temp_array = np.zeros(3, dtype='int32')
		#print np.where(table.iloc[i] == value)
		temp_array[np.where(table.iloc[i] == value)[0][0]] = 1
		dictio = {'prediksi 1':temp_array[0],
				  'prediksi 2':temp_array[1],
				  'prediksi 3':temp_array[2]}
		prediction = prediction.append(dictio, ignore_index=True)

	return prediction

# In[9]:





def test(testing, params):
    theta1 = np.array(params[0])
    bias1 = np.array(params[1])
    theta2 = np.array(params[2])
    bias2 = np.array(params[3])
    
    prediksi1 = np.zeros(table_count(testing))
    prediksi2 = np.zeros(table_count(testing))
    prediksi3 = np.zeros(table_count(testing))

    error = 0
    
    #print( table_count(testing)
    
    for i in range(table_count(testing)):
        #print( theta1
        #print( theta2
    
        h11 = sum(testing.iloc[i,4:8]*theta1[0,0:4]) + bias1[0]
        h12 = sum(testing.iloc[i,4:8]*theta1[1,0:4]) + bias1[1]
        h13 = sum(testing.iloc[i,4:8]*theta1[2,0:4]) + bias1[2]
        #print( h11
        #print( h12
        #print( h13
        s11 = signoid(h11)
        s12 = signoid(h12)
        s13 = signoid(h13)
        #print( s11
        #print( s12
        #print( s13
    
        h21 = s11 * theta2[0][0] + s12 * theta2[0][1] + s13 * theta2[0][2] + bias2[0]
        h22 = s11 * theta2[1][0] + s12 * theta2[1][1] + s13 * theta2[1][2] + bias2[1]
        h23 = s11 * theta2[2][0] + s12 * theta2[2][1] + s13 * theta2[2][2] + bias2[2]

        #print( h21
        #print( h22
        #print( h23
    
        s_total = softmax(h21) + softmax(h22) + softmax(h23)
        s21 = softmax(h21) / s_total
        s22 = softmax(h22) / s_total
        s23 = softmax(h23) / s_total
        
        #print( s21
        #print( s22
        #print( s23
        
        prediksi1[i] = s21
        prediksi2[i] = s22
        prediksi3[i] = s23
    
        error = error + -(testing.iloc[i,0] * log(s21) + testing.iloc[i,1] * log(s22) + testing.iloc[i,2] * log(s23))
        #print( error1
        #print( error2
        #print( error3
        
    error = error/table_count(testing)
    #print(error)
   
    #print( prediksi1
    #print( prediksi2
    #print( prediksi3
    
    #raw_predict
    predict_table = pd.DataFrame({'prediksi 1':prediksi1,
                                  'prediksi 2':prediksi2,
                                  'prediksi 3':prediksi3})
    
    #testing = pd.concat([testing, predict_table], axis=1)
    
    #print(predict_table)
    predict_table = predict(predict_table)
    text.append(str(predict_table)+"\n")
    #print(predict_table)

    conditional = checking_accuration(testing.iloc[:,0:3], predict_table)
    #print(conditional)
    unique, count = np.unique(conditional, return_counts=True)
    #print(unique, count)
    c = np.where(unique == True)
    #print(c[0].size)
    if(c[0].size != 0):   
        akurasi = (float(count[c[0][0]]) / table_count(testing)) * 100
        #print("hallo akurasi"+str(akurasi)+str(count[c[0][0]]/table_count(testing)))
        text.append("Akurasi "+str(akurasi)+"\n")
    else:
        akurasi = 0
        text.append("Akurasi "+str(akurasi)+"\n")
    
    return np.array([error,akurasi])


# In[ ]:

f = open('output-softmax.txt','w')
text = []

x1 = get_data(x1,1)
x2 = get_data(x2,2)
x3 = get_data(x3,3)
x4 = get_data(x4,4)
kelas = get_data(kelas,5)

df = pd.DataFrame({'x1':x1,
                   'x2':x2,
                   'x3':x3,
                   'x4':x4,
                   'kelas':kelas,
                   'fakta 1':fakta1,
                   'fakta 2':fakta2,
                   'fakta 3':fakta3})
#print(df)


error_training_epoch = np.empty(EPOCH)
error_testing_epoch = np.empty(EPOCH)
akurasi_epoch = np.empty(EPOCH)

error_training = 0
error_testing = 0
akurasi = 0

text.append("Epoch 1\n")
text.append("=======\n")

testing = df.iloc[0:30]
training = df.iloc[30:150]
#print( testing
text.append(str(testing)+"\n")
#print( training

#print( theta_bias_initiation()
training_result = back_prop(training, theta_bias_initiation())
error_training = error_training + training_result[0][0]
#print( training_result
testing_result = test(testing,training_result[1])
error_testing = error_testing + testing_result[0]
akurasi = akurasi + testing_result[1]

#print( error_training
#print( error_testing
#print( akurasi

for i in range(EPOCH):
    testing = df.iloc[30:60]
    training = df.iloc[0:30]
    training = training.append(df.iloc[60:150])

    text.append(str(testing)+"\n")
    #print(testing)
    #print( training
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    #print( training_result
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    #print( error_training
    #print( error_testing
    #print(akurasi)

    testing = df.iloc[60:90]
    training = df.iloc[0:60]
    training = training.append(df.iloc[90:150])

    text.append(str(testing)+"\n")
    #print(testing)
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    #print( training_result
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    #print( error_training
    #print( error_testing
    #print(akurasi)

    testing = df.iloc[90:120]
    training = df.iloc[0:90]
    training = training.append(df.iloc[120:150])

    text.append(str(testing)+"\n")
    #print(testing)
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    #print( training_result
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    #print( error_training
    #print( error_testing
    #print( akurasi

    testing = df.iloc[120:150]
    training = df.iloc[0:120]
    #training = training.append(df.iloc[90:150])
    text.append(str(testing)+"\n")
    #print(testing)
    
    training_result = back_prop(training,training_result[1])
    error_training = error_training + training_result[0][0]
    #print( training_result
    testing_result = test(testing,training_result[1])
    error_testing = error_testing + testing_result[0]
    akurasi = akurasi + testing_result[1]

    #print( error_training
    #print( error_testing
    #print(akurasi)


    mean_error_training = error_training/5
    mean_error_testing = error_testing/5
    mean_akurasi = akurasi/5
    error_training_epoch[i] = mean_error_training
    error_testing_epoch[i] = mean_error_testing
    akurasi_epoch[i] = mean_akurasi
    
    if(i<EPOCH-1):
        text.append("Epoch "+str(i+1)+"\n")
        text.append("=======")
        #print("Epoch", str(EPOCH+1))
        error_training = 0
        error_testing = 0
        akurasi = 0

        testing = df.iloc[0:30]
        training = df.iloc[30:150]
        text.append(str(testing)+"\n")
        #print(testing)
        #print( training

        #print( theta_bias_initiation()
        training_result = back_prop(training, training_result[1])
        error_training = error_training + training_result[0][0]
        #print( training_result
        testing_result = test(testing,training_result[1])
        error_testing = error_testing + testing_result[0]
        akurasi = akurasi + testing_result[1]

        #print( error_training
        #print( error_testing
        #print(akurasi)

print("Error Training tiap Epoch")
print(error_training_epoch)
print("Error Testing tiap Epoch")
print(error_testing_epoch)
print("Akurasi tiap Epoch")
print(akurasi_epoch)

#print( testing_result

f.writelines(text)
f.close

plt.plot(error_training_epoch,'r',error_testing_epoch,'g')
#plt.scatter(np.arange(100),akurasi_epoch)
plt.show()



# In[ ]:





# In[ ]:





# In[189]:





# In[190]:





# In[191]:





# In[192]:





# In[193]:





# In[194]:




