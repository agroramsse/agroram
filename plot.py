
import matplotlib.pyplot as plt
import json
import csv
import numpy as np

from termcolor import colored


print(colored('-'*100+' \033[1mPLOT FIGURES\033[0m', 'green'))
# Loading the amazon book 1D dataset for the 1D segment tree

print( colored('\033[1mAmazon book 1D dataset\033[0m', color= 'green'))

data = []
with open('datasets/amazon-books.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(int(row[0]))

data.sort()
# measuring the data sparcity
print('size of the dataset:', len(data))
print('min value:', min(data))
print('max value:', max(data))
print('unique values:', len(set(data)))

max = max(data)
min = min(data)
# dataset = [1 if i in data else 0 for i in range(max-min+1)]

sparsity = (((max-min+1)-len(set(data)))/(max-min+1)) 
density = (1 - sparsity) 

print(f"Sparsity: {sparsity*100:.4f}")
print(f"Density: {density*100:.4f}")
#  plot the data
# Create a 1D scatter plot (strip plot)
plt.plot(data, np.zeros_like(data) + 0.5, 'o', color='blue')

# Add titles and labels
plt.title('1D amazon-books dataset')
plt.xlabel('Data values')
plt.yticks([])  # Hide the y-axis ticks
# plt.show()
# Set more granular X-axis ticks
# min_val, max_val = min(data), max(data)
# plt.xticks(np.linspace(min_val, max_val, num=20))  # 20 evenly spaced ticks
# plot data and save the figure to a file
# plt.scatter(range(len(data)), data)
plt.savefig('plots/amazon_books.pdf')

print(colored('\033[1mplot: Done\033[0m', 'green'))
print(colored('-'*100, 'green'))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# *Loading 2D_1024x1024 spitz dataset for the 2D segment tree


print(colored('\033[1mSpitz 2d dataset\033[0m', 'green'))

with open('datasets/spitz-1024x1024.csv') as fp:
    pts = json.load(fp)

num_dims = len(pts[0])
pts = list(set(map(tuple, pts))) # Remove duplicates

# clean the prev plt object and plot the 2d points and save the figure to a file
plt.clf()
# plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts])
# Create a 2D scatter plot
plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], color='blue', marker='o')

# Add titles and labels
plt.title('Spitz 2D dataset-1024x1024')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.savefig('plots/spitz-1024x1024.pdf')
# fig = plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts])
# fig.savefig('plots/spitz-1024x1024.png')
# plt.show()
print(colored('\033[1mplot: Done\033[0m', 'green'))
print(colored('-'*100, 'green'))
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# *loading 3D nh_64 dataset
import json
from segtree import SegmentTree3D

print(colored('\033[1mnh 3d dataset\033[0m', 'green'))

with open('datasets/nh_64.txt') as fp:
    pts = json.load(fp)

num_dims = len(pts[0])
pts = list(set(map(tuple, pts))) # Remove duplicates


# print('size of the dataset:', len(pts))

# sorting 3d points
pts = sorted(pts, key=lambda x: (x[0], x[1], x[2]))

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], [pt[2] for pt in pts], color='blue', marker='o')

# Add titles and labels
ax.set_title('nh_64 3D dataset')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Save the plot as an image file
plt.savefig('plots/nh_64.pdf')

# Show the plot
# plt.show()

# plot the pts in 3d
# fig = plt.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], [pt[2] for pt in pts])
# fig.savefig('plots/nh_64.png')
# plt.show()

print(colored('\033[1mplot: Done\033[0m', 'green'))
print(colored('-'*100, 'green'))


