from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as Img
import multiprocessing


class string_picture:
    def __init__(self,img,thickness,threads = 4) -> None:
        x = np.arange(img.shape[0])
        y = np.arange(img.shape[1])
        self.xx,self.yy = np.meshgrid(x,y, indexing="ij")
        self.thickness = thickness
        self.orig = img
        self.fake = np.zeros(img.shape)
        self.threads = 4
    def draw_line(self,start:np.ndarray,end:np.ndarray,image:np.ndarray):
        direction = end - start
        normal = np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        additive = -np.dot(normal,start.T)
        grid_calc = np.abs(normal[0]*self.xx + normal[1]*self.yy + additive)
        line = np.where(grid_calc < self.thickness,0.8*(1 - grid_calc/self.thickness),0)

        #image = np.where(grid_calc < self.thickness,1,image)
        return np.clip(image + line,0,1)


    def generate_points(self, n_o_points, radius):
        points = []
        offset = np.array(self.orig.shape)//2
        for i in range(n_o_points):
            point = np.array([radius,0]).T
            alpha = i*2*np.pi/n_o_points
            rot_arr = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
            point = np.dot(rot_arr,point).T + offset
            points.append(point)
            #self.fake[point[0]][point[1]] = 10
        self.points = points
        self.t_points = []
        for i in range(4):
            self.t_points.append(self.points[i:(i+1)*len(points)//4])

    def greedy_step(self,startpoint):
        points = self.t_points
        pool = multiprocessing.Pool(processes=4)
        inputs = [(startpoint,points[0]),(startpoint,points[1]),(startpoint,points[2]),(startpoint,points[3])]
        
        output = pool.map(self.greedy_thread,inputs)
        t1,t2,t3,t4 = output
        val = min(t1[0],t2[0],t3[0],t4[0])
        if val == t1[0]:
            return t1
        if val == t2[0]:
            return t2
        if val == t3[0]:
            return t3
        if val == t4[0]:
            return t4
        return np.array([np.nan,np.nan])


    def greedy_thread(self,startpoint_and_points):
        startpoint = startpoint_and_points[0]
        points = startpoint_and_points[1]
        min_cost = np.inf
        min_cost_next_point = np.array([np.nan,np.nan])
        for point in points:
            if (point == startpoint).all():
                continue
            tested = self.draw_line(startpoint,point,self.fake)
            cost = criterion(self.orig,tested)
            if cost < min_cost:
                min_cost = cost
                min_cost_next_point = point
        return min_cost,min_cost_next_point
    


def criterion(original:np.ndarray,fake:np.ndarray):
    testbench = np.where(original>0,1,0)
    testbench = np.multiply(testbench,original-fake)
    #testbench = original-fake
    return np.sum(np.abs(testbench))

if __name__ == "__main__":
    img_real = Img.open("mario.png")
    background = Img.new('RGBA', img_real.size, (255, 255, 255))
    img_real = Img.alpha_composite(background, img_real)
    img_real = img_real.convert('L')
    img_arr = 1 - np.array(img_real)/255
    #img_arr = np.where(img_arr<1,0,1)
    #img_arr = np.ones((100,100))
    #x = np.arange(img_arr.shape[0])
    #y = np.arange(img_arr.shape[1])
    #xx,yy = np.meshgrid(x,y, indexing="ij")
    #img_arr = np.where((xx - 50)**2 + (yy - 50)**2 < 10**2,0,1)
    plt.imshow(img_arr,cmap='gray')
    plt.show()
    pic = string_picture(img_arr,6.)
    pic.generate_points(90,max(img_arr.shape)/2)
    startpoint = pic.points[10]
    prev_cost = np.inf
    i = 0
    while i < 5000:
        cost,next_startpoint = pic.greedy_step(startpoint)
        print(startpoint,next_startpoint)
        if np.isnan(next_startpoint).any():
            print("finished")
            break
        if prev_cost <= cost:
            print("done!")
            break
        prev_cost = cost
        pic.fake = pic.draw_line(startpoint,next_startpoint,pic.fake)
        startpoint = next_startpoint
        print(i)
        if i%20 == 0:
            plt.imshow( 1 -pic.fake,cmap="gray")
            plt.draw() 
            plt.pause(0.000001)      
        i+=1
    plt.imshow(1 - pic.fake,cmap="gray")
    plt.show()
