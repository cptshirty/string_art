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
        self.find_weigth_matrix()

    def find_weigth_matrix(self):
        img = self.orig
        non_white = np.sum(np.where(img>0,1,0))
        total = img.shape[0]*img.shape[1]
        percentage_non_white = non_white/total
        self.weigth = np.where(img>0,1/percentage_non_white,1)
        print(np.max(self.weigth))
        #bb_tl = np.array([min(xx),min(yy)])
        #bb_br = np.array([max(xx),max(yy)])
        #self.bb = np.zeros(img.shape)
        #self.bb[bb_tl[0]:bb_br[0]][bb_tl[1]:bb_br[1]] = 



    def draw_line(self,start:np.ndarray,end:np.ndarray,image:np.ndarray):
        direction = end - start
        normal = np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        additive = -np.dot(normal,start.T)
        grid_calc = np.abs(normal[0]*self.xx + normal[1]*self.yy + additive)
        line = np.where(grid_calc < self.thickness,1*(1 - grid_calc/self.thickness),0)

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
            cost = self.criterion(self.orig,tested)
            if cost < min_cost:
                min_cost = cost
                min_cost_next_point = point
        return min_cost,min_cost_next_point
    


    def criterion(self,original:np.ndarray,fake:np.ndarray):
        testbench = np.multiply(self.weigth,original-fake)
        #testbench = original-fake
        return np.sum(np.square(testbench))

if __name__ == "__main__":
    img_real = Img.open("robot.png")
    #background = Img.new('RGBA', img_real.size, (255, 255, 255))
    #background.paste(img_real,(0,0),img_real)
    img_real = img_real.convert('L')
    img_arr = 1- np.array(img_real)/255
    plt.imshow(img_arr,cmap='gray')
    plt.show()
    pic = string_picture(img_arr,2)
    pic.generate_points(360*2,1.4*max(img_arr.shape)/2)
    startpoint = pic.points[10]
    prev_cost = np.inf
    i = 0
    while i < 5000:
        cost,next_startpoint = pic.greedy_step(startpoint)
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
            plt.imshow( 1-pic.fake,cmap="gray")
            plt.draw() 
            plt.pause(0.000001)      
        i+=1

    to_save = Img.fromarray(255*pic.fake)
    to_save = to_save.convert('RGBA')
    to_save.show()
    to_save.save("robot_string.png")
    
    plt.imshow(1 - pic.fake,cmap="gray")
    plt.show()
