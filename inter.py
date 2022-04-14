from tkinter import *
import numpy as np



#Height and width setup
height = 1000
width  = 1000
percent_margin = 0.1
root   = Tk()
root.title('Some Titles Are Best Left Unread')
root.geometry(f'{width}x{height}')
c_width=int(width*(1-percent_margin))
c_height=int(height*(1-percent_margin))
can = Canvas(root, width=c_width, height=c_height, bg='Black')
can.pack(pady=10)
#can.create_rectangle(1,1,20,20, fill='blue')
root.update_idletasks()
root.update()

'''
things i need a grid to do:
draw the path of the agent traveling through the grid
draw the boarders/ squares of the grid
-------------------------
#   0 = Free Square     #
#   1 = Blocked Sqare   #
#   2 = Agent Start     #
#   3 = Goal Square     #
-------------------------
'''
grid_len_x = 10
grid_len_y = 10
grid = np.zeros((10,10))
grid[3,6] = 1
grid[3,2] = 2
grid[3,1] = 3
grid[3,4] = 4
def drawgrid(grid):
    top_left_x=1
    top_left_y=1
    #bottom_right_x=1
    #bottom_right_y=1
    sq_size = min(c_height/grid_len_y,c_width/grid_len_x)
    for col in grid:
        for row in col:
            if row==0:#   0 = Free Square     #
                can.create_rectangle(top_left_x, top_left_y, (top_left_x+sq_size), (top_left_y+sq_size), fill='blue')
            elif row == 1: #   1 = Blocked Sqare   #
                can.create_rectangle(top_left_x, top_left_y, (top_left_x + sq_size), (top_left_y + sq_size), fill='black')
            elif row == 2:#   2 = Agent Start     #
                can.create_rectangle(top_left_x, top_left_y, (top_left_x + sq_size), (top_left_y + sq_size), fill='red')
            elif row == 3:#   3 = Goal Square     #
                can.create_rectangle(top_left_x, top_left_y, (top_left_x + sq_size), (top_left_y + sq_size), fill='green')
            else:
                can.create_rectangle(top_left_x, top_left_y, (top_left_x + sq_size), (top_left_y + sq_size), fill='magenta')
            top_left_x = (top_left_x+sq_size)
        top_left_x = 1
        top_left_y=(top_left_y+sq_size)
    root.update_idletasks()
    root.update()

drawgrid(grid)
print('done')
root.mainloop()







