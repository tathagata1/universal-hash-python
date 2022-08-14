# %% [markdown]
# Tathagata Mookherjee
# ROLL NUMBER - M21AI619
# CSL7630: Algorithms for Big Data, Assignment 1
# IIT JODHPUR

import warnings
warnings.filterwarnings("ignore")
import random as rnd

number_list=[]

#generating random numbers
for i in range(500000):
    number_list.append(rnd.randint(1, 10000000))


# Question 1: Remove Duplicate and Build Statistics (15 Marks)
print('Question 1: Remove Duplicate and Build Statistics (15 Marks)')

import numpy as np
from scipy import stats
import pandas as pd

K=np.unique(number_list)

print("length of original number list: "+str(len(number_list)))
print("length of unique number list: "+str(len(K)))

print("Statistics:")

var_stats=stats.describe(K)
var_stats_df = pd.DataFrame([var_stats], columns=var_stats._fields)
print (var_stats_df)

# Question 2: Dot Product Hash Family (15 Marks)
print('Question 2: Dot Product Hash Family (15 Marks)')

#function to change the bases on any int wrt any other int
def changeBase(n, b):
    if n == 0:
        return [0]
    str_no = ''
    while n:
        str_no=str(int(n % b))+str_no
        n //= b
    return int(str_no)

def doDotProductHash(m, K):
    r=rnd.randint(2, 5)
    U=m**r
    dot_list=[]
    for i in range(len(K)):
        Km=changeBase(K[i], m)
        A=rnd.randint(1, (U-1))
        Ak=changeBase(A, m)
        dot_list.append(int((Km*Ak) % m))
    return dot_list

print('changeBase and doDotProductHash compiled')

# Question 3: Linear Combination Hash Family (15 Marks)
print('Question 3: Linear Combination Hash Family (15 Marks)')
#function to check if a number is prime or not
def isPrime(num):
    if num > 1:
        for i in range(2, int(num/2)+1):
            if (num % i) == 0:
                return False
        else:
            return True
    else:
        return False

def doLinearCombinationHash(m, K):
    r=rnd.randint(2, 5)
    U=m**r
    TU=2*U

    primeFlag=False
    while(primeFlag == False):
        p=rnd.randint(U+1, TU-1)
        primeFlag=isPrime(p)    
    a=rnd.randint(1, p-1)
    b=rnd.randint(0, p-1)
    lin_list=[]
    for i in range(len(K)):
        X1=(a*K[i])+b
        X2=X1 % p
        X3=X2 % m
        lin_list.append(X3)
    return lin_list

print('isPrime and doLinearCombinationHash compiled')

# Question 4: Testing the Hash Functions (10 Marks)
print('Question 4: Testing the Hash Functions (10 Marks)')

from collections import Counter

#provide the upper limit for the prime number M. set >10
#M is guaranteed to be smaller than U due to the implementation of the functions themselves***
in_M_min=10
in_M_max=200

#set your input number list
in_list=K

primeFlag=False
while(primeFlag == False):
    var_prime=rnd.randint(in_M_min, in_M_max)
    primeFlag=isPrime(var_prime)

print('prime number used: '+str(var_prime))

var_linear_colis = Counter(doLinearCombinationHash(var_prime, in_list))
var_dot_colis = Counter(doDotProductHash(var_prime, in_list))

tot_count=len(in_list)
lin_count=len(var_linear_colis)
dot_count=len(var_dot_colis)

counter=0
lin_sum_values=0
for i in list(var_linear_colis.values()):
    if(i!=1):
        counter+=1
        lin_sum_values+=i

lin_avg_values=lin_sum_values/counter

counter=0
dot_sum_values=0
for i in list(var_dot_colis.values()):
    if(i!=1):
        counter+=1
        dot_sum_values+=i
dot_avg_values=dot_sum_values/counter

#print collisions
print('count of data: '+str(tot_count))
print('')
print('LINEAR HASH')
print('average collisions: '+str(lin_avg_values))
print('number of hashes: '+str(lin_count))
print(var_linear_colis)
print()
print('')
print('DOT HASH')
print('average collisions: '+str(dot_avg_values))
print('number of hashes: '+str(dot_count))
print(var_dot_colis)

# OBSERVATIONS
# 1. If the data count is at 10, linear hash tends to perform better than dot hash as it maintains similar or lesser number of average collisions while having creating hashes. Sample data below,
# 
# LINEAR HASH: average collisions: 2.0 - number of hashes: 8
# DOT HASH: average collisions: 2.0 - number of hashes: 9
# 
# 2. Increasing the data count to 100 shows similar performance where linear as well as dot hash is able to maintain low average collisions of 5-6.
# 
# average collisions: 5.2631578947368425 - number of hashes: 19
# 
# 3. Increasing to 10,000 or more shows makes the average collisions range from between 400+ to 3000+ for both DOT and LINEAR hashing.
# 
# Hence we can see that for smaller datasets linear hashing is better, but for larger datasets both hashing algorithms perform similarly.

# Question 5: Quick Select: Find kth largest element (20 Marks)
print('Question 5: Quick Select: Find kth largest element (20 Marks)')

import tracemalloc
import time

tracemalloc.start()
start_time = time.time()

n = len(K)

def quicksort(arr, l, r):
	x = arr[r]
	i = l
	for j in range(l, r):
		if arr[j] <= x:
			arr[i], arr[j] = arr[j], arr[i]
			i += 1
	arr[i], arr[r] = arr[r], arr[i]
	return i

def kthLargest(arr, l, r, k):
	if (k > 0 and k <= r - l + 1):
		index = quicksort(arr, l, r)
		if (index - l == k - 1):
			return arr[index]
		if (index - l > k - 1):
			return kthLargest(arr, l, index - 1, k)
		return kthLargest(arr, index + 1, r,
							k - index + l - 1)


#input your kth number here
k = 20

print(str(k)+"-th largest element is "+str(kthLargest(K, 0, n - 1, n-k)))

stat=tracemalloc.take_snapshot().statistics('traceback')[0]
print("memory usage: %s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
print("time consumed is %s seconds" % (time.time() - start_time))

tracemalloc.stop()

# Question 6: Lazy Select (25 Marks)
print('Question 6: Lazy Select (25 Marks)')

import tracemalloc
import time

tracemalloc.start()
start_time = time.time()

#declaring initial vars

counter=0
#S=K
S=[14,15,16,17,12,14,15,16,17,12,14,15,16,17,12,14,15,16,17,12,12,11,19,20,13,12,11,10,9,8,7,6,5,4,3,2,1,114,15,16,17,12,14,15,16,17,12,14,15,16,17,12,14,15,16,17,12,12,11,19,20,13,12,11,10,9,8,7,6,5,4,3,2,1,1,1]

#length of input S
n=len(S)

#position of item to find
k=int(np.floor(n/2))-1
#k=9

S_sort=np.sort(S)
print('actual '+str(k)+'-th smallest element is '+str(S_sort[k]))

#size of R
R_len=int(np.floor(n**(3/4)))
#print('R_len = '+str(R_len))
foundFlag=False
while (foundFlag==False):
    P=[]
    R=[]
    counter+=1
    #randomly picking values from S and appending to R
    for i in range(R_len):
        R.append(S[rnd.randint(0, n-1)])    
    
    #sorting R in ascending order
    R_sort = np.sort(R)
    #print('R_sort = '+str(R_sort))
    #position of Ra, did -1 because index starts from 0
    ra=int(np.floor(k*n**(-1/4)-np.sqrt(n)))
    #position of Rb, did -1 because index starts from 0
    rb=int(np.ceil(k*n**(-1/4)+np.sqrt(n)))-1
    #item from position of Ra in R
    ra_item=R_sort[ra]
    #item from position of Rb in R
    rb_item=R_sort[rb]
    # position of ra_item in S
    sa_pos=list(np.where(S_sort == ra_item)[0])[0]
    # position of rb_item in S
    sb_pos=list(np.where(S_sort == rb_item)[0])[0]
    
    if (k>sa_pos and k<sb_pos):
        for i in range(n):
            if (S_sort[i] >= ra_item and S_sort[i] <= rb_item):
                P.append(S_sort[i])
        if(np.linalg.norm(P) <= (4*R_len)):
            print('algorithmic '+str(k)+"-th smallest element is "+str(P[k-sa_pos]))
            print('Iterations consumed: '+str(counter))
            foundFlag=True
                
stat=tracemalloc.take_snapshot().statistics('traceback')[0]
print("memory usage: %s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
print("time consumed is %s seconds" % (time.time() - start_time))

tracemalloc.stop()
