from tkinter import *
from tkinter import colorchooser
from tkinter import ttk
from time import sleep
import cv2
import numpy as np
from functools import partial # allows us to pass arguments in callback

root = Tk()
root.wm_attributes('-fullscreen','true')
root.option_add("*tearOff", False) # needed for drop-down menu
root.geometry('850x700')
root.state("zoomed")
root.title('Drawing Tool')
root.resizable(False,False)

toolsDebounce = True

circleNum = 0 #number of canvas shapes used for drawing, not including the erasing ones

tool = 'draw' #tool currently in use - default can be set here

fillColor = '#000000' #color for drawing - default can be set here

backgroundColor = '#FFFFFF' #background color of drawing canvas - default can be set here

old_x = None
old_y = None

scaleVal = DoubleVar()
brush_scale = Scale(root, orient = HORIZONTAL, length = 850, variable = scaleVal, from_ = 0.0, to = 100.0)
brush_scale.set(10.0)
brush_scale.pack()

styleQ = ttk.Style()
quitButton = ttk.Button(root,text='Quit',command = lambda: root.destroy())
styleQ.configure('Quit.TButton',foreground='red',font=('Philosopher',15,'bold'))
quitButton.config(style = 'Quit.TButton')

styleL = ttk.Style()
windowDown = ttk.Button(root,text='Leave',command=lambda: root.iconify())
styleL.configure('Leave.TButton',foreground='yellow',font=('Philosopher',15,'bold'))
windowDown.config(style = 'Leave.TButton')

windowDown.pack()
quitButton.pack()

drawings = [
    #[(x1,y1),(x2,y2),(255,55,255)]
]

def draw(event):
    global circleNum

    if old_x and old_y:
        if tool == 'draw':
            drawCanvas.create_line(old_x,old_y,event.x,event.y,width = brush_scale.get(),capstyle = ROUND, smooth = True,
                fill = fillColor, tag = "drawPiece")
            circleNum = circleNum + 1
            brush_color = fillColor.lstrip('#')
            brush_color = list(int(brush_color[i:i+2], 16) for i in (0, 2, 4))
            brush_color.reverse() # reverse color to make BGR instead of RGB
            drawings.append([(old_x,old_y), (event.x,event.y), brush_color, brush_scale.get()])

        elif tool == 'erase':
            drawCanvas.create_line(old_x,old_y,event.x,event.y,width = brush_scale.get(),capstyle = ROUND, smooth = True,
                fill = backgroundColor, tag = "erasePiece")

    setOlds(event)

def setOlds(event):
    global old_x
    old_x = event.x
    global old_y
    old_y = event.y

def toolsDebounceCheck():
    if toolsDebounce:
        return True
        
# Shortcut Actions

def change_brush_color():
    color = ask_color()
    if color:
        global fillColor
        fillColor = color

def change_background_color():
        color = ask_color()
        if color != False:
            global backgroundColor
            backgroundColor = color
            drawCanvas.config(background=backgroundColor)
            drawCanvas.itemconfig('erasePiece', fill = backgroundColor)

def clear_all():
    drawCanvas.delete('all')
            
#Event-Listeners

def canvas_click(event):
    setOlds(event)
    draw(event)

def canvas_motion(event):
    draw(event)

def key_binds(event):
    if event.char == "c":
        clear_all()
            
    elif event.char == "r":
        change_brush_color()
            
    elif event.char == "d":
        select_tool("draw")
            
    elif event.char == "e":
        select_tool("erase")

    elif event.char == "b":
        change_background_color()
    
    elif event.char == "s":
        pass

def ask_color():
    asking = colorchooser.askcolor()
    color = asking[1]
    if color:
        if color[:1] == '#':
            return color
        else:
            return False

def select_tool(choice_tool):
    global tool
    tool = choice_tool

def save_as(transparent = False):
    
    width = drawCanvas.winfo_width()
    height = drawCanvas.winfo_height()

    if not transparent:
        image = np.zeros((height, width, 3), dtype = np.uint8) # initialize image for saving preferences
        # converting HEX value background color to BGR color
        backgroundColorBGR = backgroundColor.lstrip('#')
        backgroundColorBGR = list(int(backgroundColorBGR[i:i+2], 16) for i in (0, 2, 4))
        backgroundColorBGR.reverse() # reverse color to make BGR instead of RGB
        image[:] = backgroundColorBGR # change background color

        for drawing in drawings: # loop through each drawing
            cv2.line(image, drawing[0], drawing[1], drawing[2], thickness = drawing[3], lineType=cv2.LINE_AA) # create line that user drawwed
        cv2.imwrite("yourimage.png", image)
    else:
        imageTransparent = np.zeros((height, width, 4), dtype = np.uint8) # 4 channels instead of 3, for transparent images
        
        for drawing in drawings: # loop through each drawing
            BGR = drawing[2]
            cv2.line(imageTransparent, drawing[0], drawing[1], color = (BGR[0], BGR[1], BGR[2], 255), thickness = drawing[3])
        cv2.imwrite("yourimage.png", imageTransparent)

def main():
    global drawCanvas

    # drop-down menus (file, menu, preferences etc)
    menubar = Menu(root)
    root.config(menu = menubar)
    file = Menu(menubar)
    edit = Menu(menubar)
    menubar.add_cascade(menu = file, label = "File")
    menubar.add_cascade(menu = edit, label = "Edit")
    save_transparent = partial(save_as, True)
    file.add_command(label = "Save", command = save_as)
    file.add_command(label = "Save Transparent", command = save_transparent)

    myframe = Frame(root)
    myframe.pack(fill=BOTH, expand=YES)
    drawCanvas = Canvas(myframe,width = 850, height = 700, background = backgroundColor, highlightthickness=5)
    drawCanvas.pack(fill=BOTH, expand=YES)

    frameBack = ttk.Style()
    frameBack.configure('Gray.TFrame', background = '#cbcdd1')

    root.bind('<KeyPress>', key_binds)
    drawCanvas.bind('<ButtonPress>', canvas_click)
    drawCanvas.bind('<B1-Motion>', canvas_motion)

    root.mainloop()

main()