import matplotlib.pyplot as plt
import numpy as np

mat = []
row = []

with open('./data', 'r') as f:
    for i in f:
        i = i.strip()
        data = i.split('\t')        
        for j in data:
           row.append(int(j))
        mat.append(row.copy())
        row.clear()

i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(1001):
   i1.append(mat[i][0])
   i2.append(mat[i][1])
   i3.append(mat[i][2])
   i4.append(mat[i][3])
   i5.append(mat[i][4])
   i6.append(mat[i][5])
   i7.append(mat[i][6])
   i8.append(mat[i][7])
   i9.append(mat[i][8])
   i10.append(mat[i][9])
   i11.append(mat[i][10])
   i12.append(mat[i][11])
   i13.append(mat[i][12])
   i14.append(mat[i][13])

plt.xlabel('Iteration')
plt.ylabel('Value')
#plt.axis([0,1000,0,1000])
plt.title('Value-State Plot')
plt.plot(range(len(i1)), i1, color="blue")
plt.plot(range(len(i2)), i2, color="red")
plt.plot(range(len(i3)), i3, color="orange")
plt.plot(range(len(i4)), i4, color="yellow")
plt.plot(range(len(i5)), i5, color="green")
plt.plot(range(len(i6)), i6, color="black")
plt.plot(range(len(i7)), i7, color="purple")
plt.plot(range(len(i8)), i8, color="blue")
plt.plot(range(len(i9)), i9, color="blue")
plt.plot(range(len(i10)), i10, color="blue")
plt.plot(range(len(i11)), i11, color="blue")
plt.plot(range(len(i12)), i12, color="blue")
plt.plot(range(len(i13)), i13, color="blue")
plt.plot(range(len(i14)), i14, color="blue")
plt.show()
