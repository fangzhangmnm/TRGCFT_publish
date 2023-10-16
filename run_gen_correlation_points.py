
import pickle

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os


# =================================
# verify and save the data

def dist(coords1,coords2):
    return ((coords1[0]-coords2[0])**2+(coords1[1]-coords2[1])**2)**0.5

def validate_and_save(filename,coordsss,lattice_size,override=False):
    if not override and os.path.exists(filename):
        print('file already exists:',filename)
        return
    # remove the duplicated ones
    coordsss=list(sorted(set(coordsss)))
    print('filename:',filename)
    for coordss in coordsss:
        # print coords and distance
        for coords in coordss:
            print(coords,end=' ')
        print()
        for i in range(len(coordss)):
            for j in range(i+1,len(coordss)):
                print(dist(coordss[i],coordss[j]),end=' ')
        print()
        # check if they are the same point
        for i in range(len(coordss)):
            for j in range(i+1,len(coordss)):
                assert coordss[i]!=coordss[j]
        # check if they are in the range of the lattice
        for coords in coordss:
            assert coords[0]>=0 and coords[0]<lattice_size[0]
            assert coords[1]>=0 and coords[1]<lattice_size[1]

    pickle.dump(coordsss,open(filename,'wb'))
    print('total correlators:',len(coordsss))
    print('saved to',filename)


def generate_2pt_correlation_points(lattice_size,data_count=100,fixed_x0=None,fixed_y0=None):
    rmin,rmax=1,min(lattice_size)
    coordsss=[]
    while len(coordsss)<data_count:
        th=np.random.uniform(0,np.pi/2)
        r=np.exp(np.random.uniform(np.log(rmin),np.log(rmax)))
        x,y=int(np.abs(r*np.cos(th))),int(np.abs(r*np.sin(th)))
        if x==0 and y==0:
            x,y=(1,0) if np.random.uniform()<0.5 else (0,1)
        x0,y0=np.random.randint(0,lattice_size[0]-x),np.random.randint(0,lattice_size[1]-y)
        x0=fixed_x0 if fixed_x0 is not None else x0
        y0=fixed_y0 if fixed_y0 is not None else y0
        x1,y1=x0+x,y0+y
        coordsss.append(((x0,y0),(x1,y1)))
    return coordsss

lattice_size=(2**30,2**30)
coordsss=generate_2pt_correlation_points(lattice_size,data_count=100)
validate_and_save('data/2pt_correlation_points_30.pkl',coordsss,lattice_size)
coordsss=generate_2pt_correlation_points(lattice_size,data_count=900)
validate_and_save('data/2pt_correlation_points_30_appended.pkl',coordsss,lattice_size)
coordsss=generate_2pt_correlation_points(lattice_size,data_count=100,fixed_x0=0,fixed_y0=0)
validate_and_save('data/2pt_correlation_points_30_00.pkl',coordsss,lattice_size)

def generate_4pt_correlation_points(lattice_size,data_count=100):
    coordsss=[]
    while len(coordsss)<data_count:
        # choose a center point randomly in the lattice
        # for each point, determine the distance to the center point randomly in geometric distribution
        # then determine the angle randomly
        # choose r from [1,max_block_size] in geometric distribution
        max_block_size=min(lattice_size[0],lattice_size[1])//3
        r1=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
        r2=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
        r3=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
        r4=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
        block_size=max(r1,r2,r3,r4)
        # choose a center point randomly in the lattice
        cx0,cy0=np.random.randint(block_size,lattice_size[0]-block_size),np.random.randint(block_size,lattice_size[1]-block_size)

        theta1=np.random.uniform(0,2*np.pi)
        theta2=np.random.uniform(0,2*np.pi)
        theta3=np.random.uniform(0,2*np.pi)
        theta4=np.random.uniform(0,2*np.pi)
        x0,y0=cx0+int(r1*np.cos(theta1)),cy0+int(r1*np.sin(theta1))
        x1,y1=cx0+int(r2*np.cos(theta2)),cy0+int(r2*np.sin(theta2))
        x2,y2=cx0+int(r3*np.cos(theta3)),cy0+int(r3*np.sin(theta3))
        x3,y3=cx0+int(r4*np.cos(theta4)),cy0+int(r4*np.sin(theta4))
        # confine them into the lattice
        x0,y0=max(0,min(lattice_size[0]-1,x0)),max(0,min(lattice_size[1]-1,y0))
        x1,y1=max(0,min(lattice_size[0]-1,x1)),max(0,min(lattice_size[1]-1,y1))
        x2,y2=max(0,min(lattice_size[0]-1,x2)),max(0,min(lattice_size[1]-1,y2))
        x3,y3=max(0,min(lattice_size[0]-1,x3)),max(0,min(lattice_size[1]-1,y3))

        # check if they are the same point
        if x0==x1 and y0==y1: continue
        if x0==x2 and y0==y2: continue
        if x0==x3 and y0==y3: continue
        if x1==x2 and y1==y2: continue
        if x1==x3 and y1==y3: continue
        if x2==x3 and y2==y3: continue
        # save the data
        coordsss.append(((x0,y0),(x1,y1),(x2,y2),(x3,y3)))
    return coordsss
        
lattice_size=(2**10,2**10)
coordsss=generate_4pt_correlation_points(lattice_size,data_count=1000)
validate_and_save('data/4pt_correlation_points_10.pkl',coordsss,lattice_size)
lattice_size=(2**30,2**30)
coordsss=generate_4pt_correlation_points(lattice_size,data_count=1000)
validate_and_save('data/4pt_correlation_points.pkl',coordsss,lattice_size)

def random_rel_to_corner(lx,ly):
    diagonal_length=np.sqrt(lx**2+ly**2)
    r=np.exp(np.random.uniform(np.log(1),np.log(diagonal_length)))
    theta=np.random.uniform(0,2*np.pi)
    x,y=int(r*np.cos(theta)),int(r*np.sin(theta))
    if x<0: x+=lx
    if y<0: y+=ly
    # confine them into the bigger block
    x,y=max(0,min(lx-1,x)),max(0,min(ly-1,y))
    assert x>=0 and x<lx
    assert y>=0 and y<ly
    return x,y
def generate_smearing_corner_points(lattice_size,data_count=128):
    coordsss=[]
    for l in range(0,20):
        for i in range(data_count):
            lx=2**(l//2) if l%2==0 else 2**(l//2+1)
            ly=2**(l//2)
            if l%2==0:
                lx=2**(l//2)
                ly=2**(l//2)
                BX0=np.random.randint(0,(lattice_size[0]//lx)-1)
                BX1=BX0+1
                BY0=np.random.randint(0,lattice_size[1]//ly)
                BY1=BY0
            else:
                lx=2**(l//2+1)
                ly=2**(l//2)
                BX0=np.random.randint(0,lattice_size[0]//lx)
                BX1=BX0
                BY0=np.random.randint(0,(lattice_size[1]//ly)-1)
                BY1=BY0+1

            x0,y0=random_rel_to_corner(lx,ly)
            x0,y0=BX0*lx+x0,BY0*ly+y0
            x1,y1=random_rel_to_corner(lx,ly)
            x1,y1=BX1*lx+x1,BY1*ly+y1
            #print(lx,ly,BX0,BX1,BY0,BY1,x0,y0,x1,y1)
            assert x0!=x1 or y0!=y1
            assert x0>=0 and x0<lattice_size[0]
            assert x1>=0 and x1<lattice_size[0]
            assert y0>=0 and y0<lattice_size[1]
            assert y1>=0 and y1<lattice_size[1]
            coordsss.append(((x0,y0),(x1,y1)))
    return coordsss
lattice_size=(2**10,2**10)
coordsss=generate_smearing_corner_points(lattice_size,data_count=1024)
validate_and_save('data/smearing_corner_10.pkl',coordsss,lattice_size)


def generate_torus_correlation_points(log2Size,data_count_axis=30,fixed_y=None):
    coordsss=[]
    lattice_size=(2**log2Size,2**log2Size)
    xAxis=list(range(1,5))+list(np.geomspace(5,2**log2Size-1,max(2,data_count_axis//2-4)).astype(int))
    xAxis=[int(x) for x in xAxis]
    xAxis=sorted(set(xAxis+[lattice_size[0]-x for x in xAxis]))
    yAxis=[0]+xAxis if fixed_y is None else [fixed_y]
    for x in xAxis:
        for y in yAxis:
            x0,y0=0,y
            x1,y1=x,y
            coordsss.append(((x0,y0),(x1,y1)))
    return coordsss

log2Size=10
data_count_axis=30
coordsss=generate_torus_correlation_points(log2Size,data_count_axis)
validate_and_save('data/torus_correlation_points_y_10.pkl',coordsss,(2**log2Size,2**log2Size))

log2Size=30
data_count_axis=100
coordsss=generate_torus_correlation_points(log2Size,data_count_axis,fixed_y=0)
validate_and_save('data/torus_correlation_points_30_00.pkl',coordsss,(2**log2Size,2**log2Size))

log2Size=10
data_count_axis=30
coordsss=generate_torus_correlation_points(log2Size,data_count_axis,fixed_y=2**(log2Size-1))
validate_and_save('data/torus_correlation_points_y_mid_10.pkl',coordsss,(2**log2Size,2**log2Size))

log2Size=30
data_count_axis=100
coordsss=generate_torus_correlation_points(log2Size,data_count_axis,fixed_y=2**(log2Size-1))
validate_and_save('data/torus_correlation_points_30_mid.pkl',coordsss,(2**log2Size,2**log2Size))

# smearing between edge
# filename='data/smearing_between_edge_10.pkl'
# lattice_size=(1024,1024)
# for dist in [1,2,4,8,16,32,64,128,256,512]:
#     for start in [1,2,4,8,16,32,64,128,256,512]:
#         x0,y0=start-1,start-1
#         x1,y1=start-1+dist,start-1+dist
#         coordsss.append(((x0,y0),(x1,y1)))


# smearing between edge3 Not good
# filename='data/smearing_between_edge_10_3.pkl'
# lattice_size=(1024,1024)
# for blockSize in [1,2,4,8,16,32,64,128,256,512]:
#     for i in range(250):
#         bBlockX=np.random.randint(0,lattice_size[0]//blockSize//2)*blockSize*2
#         bBlockY=np.random.randint(0,lattice_size[1]//blockSize//2)*blockSize*2
#         # find the center of the bigger block
#         cx0,cy0=bBlockX+blockSize-1,bBlockY+blockSize-1
#         # choose a point on the horizontal or vertical center line of the bigger block, in geometric scale
#         r=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         # up down left right
#         dir=np.random.randint(0,4)
#         if dir==0:
#             cx0,cy0=cx0,cy0+int(r)
#         elif dir==1:
#             cx0,cy0=cx0,cy0-int(r)
#         elif dir==2:
#             cx0,cy0=cx0+int(r),cy0
#         elif dir==3:
#             cx0,cy0=cx0-int(r),cy0

#         # choose r from [1,blockSize-1] in geometric
#         r1=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         r2=np.exp(np.random.uniform(np.log(1),np.log(blockSize)))
#         theta1=np.random.uniform(0,2*np.pi)
#         theta2=np.random.uniform(0,2*np.pi)
#         x0,y0=cx0+int(r1*np.cos(theta1)),cy0+int(r1*np.sin(theta1))
#         x1,y1=cx0+int(r2*np.cos(theta2)),cy0+int(r2*np.sin(theta2))
#         # confine them into the bigger block
#         x0,y0=max(bBlockX,min(bBlockX+blockSize*2-1,x0)),max(bBlockY,min(bBlockY+blockSize*2-1,y0))
#         x1,y1=max(bBlockX,min(bBlockX+blockSize*2-1,x1)),max(bBlockY,min(bBlockY+blockSize*2-1,y1))
#         # check if they are the same point
#         if x0==x1 and y0==y1:
#             continue
#         coordsss.append(((x0,y0),(x1,y1)))




# sigma sigma epsilon correlation
# filename='data/sigma_sigma_epsilon_correlation_points.pkl'
# lattice_size=(1024,1024)
# coordsss=[]
# N=1000
# while len(coordsss)<N:
#     # choose a center point randomly in the lattice
#     # for each point, determine the distance to the center point randomly in geometric distribution
#     # then determine the angle randomly
#     # choose r from [1,max_block_size] in geometric distribution
#     max_block_size=min(lattice_size[0],lattice_size[1])//3
#     r1=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
#     r2=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
#     r3=np.exp(np.random.uniform(np.log(2),np.log(max_block_size)))
#     block_size=max(r1,r2,r3)
#     # choose a center point randomly in the lattice
#     cx0,cy0=np.random.randint(block_size,lattice_size[0]-block_size),np.random.randint(block_size,lattice_size[1]-block_size)

#     theta1=np.random.uniform(0,2*np.pi)
#     theta2=np.random.uniform(0,2*np.pi)
#     theta3=np.random.uniform(0,2*np.pi)
#     x0,y0=cx0+int(r1*np.cos(theta1)),cy0+int(r1*np.sin(theta1))
#     x1,y1=cx0+int(r2*np.cos(theta2)),cy0+int(r2*np.sin(theta2))
#     x2,y2=cx0+int(r3*np.cos(theta3)),cy0+int(r3*np.sin(theta3))
#     x3,y3=x2+1,y2
#     # confine them into the lattice
#     x0,y0=max(0,min(lattice_size[0]-1,x0)),max(0,min(lattice_size[1]-1,y0))
#     x1,y1=max(0,min(lattice_size[0]-1,x1)),max(0,min(lattice_size[1]-1,y1))
#     x2,y2=max(0,min(lattice_size[0]-1,x2)),max(0,min(lattice_size[1]-1,y2))
#     x3,y3=max(0,min(lattice_size[0]-1,x3)),max(0,min(lattice_size[1]-1,y3))

#     # check if they are the same point
#     if x0==x1 and y0==y1: continue
#     if x0==x2 and y0==y2: continue
#     if x0==x3 and y0==y3: continue
#     if x1==x2 and y1==y2: continue
#     if x1==x3 and y1==y3: continue
#     if x2==x3 and y2==y3: continue
#     # save the data
#     coordsss.append(((x0,y0),(x1,y1),(x2,y2),(x3,y3)))


