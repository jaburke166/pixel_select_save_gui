"""
Created on Wed 12 May 12:24:19 2021

@author: Jamie Burke
@email s1522100@ed.ac.uk

This script allows a high-level GUI to selectg pixels from an image and save them as part of an image processing pipeline

"""
import os
import pandas as pd
import cv2
import numpy as np
import sys

default_dir = sys.path[0]
class select_pixels(object):
    
    def __init__(self, image_path, output_fname, image_number, img_or_array='img', 
                 no_of_pixels=4, which_group='Donors', scale=800, second_mon=True, save_path=default_dir):
        
        self.img_or_array = img_or_array
        self.save_path = save_path
        self.second_mon = second_mon
        self.no_of_pixels = no_of_pixels
        if self.img_or_array == 'img':
            self.image_path = image_path
            self.img = cv2.imread(self.image_path, 1)
        else:
            self.image_path = image_path
            self.img = self.image_path
            
        if self.img.ndim == 2:
            self.img = cv2.cvtColor(self.img.astype(np.float32), code=cv2.COLOR_GRAY2BGR)
            
        self.shape = np.array([self.img.shape[0], self.img.shape[1]])
        self.shape_ar = self.shape / self.shape.max()
        self.scale = scale
        self.ar_size = (int(scale*self.shape_ar[0]), int(scale*self.shape_ar[1]))
        
        self.output_fname = output_fname
        self.image_number = image_number
        self.which_group = which_group
        self.zoom_txt = 'zoom_xy.txt'
        self.selected_txt = 'selected_pixels.txt'
        
        self.zoom_txt_path = os.path.join(self.save_path, self.zoom_txt)
        self.select_pixel_path = os.path.join(self.save_path, self.selected_txt)

    def show_img(self, img, image_name='original_image'):
        cv2.namedWindow(image_name, flags=cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(image_name, self.ar_size[1], self.ar_size[0])
        cv2.imshow(image_name, img)
        
        # If displaying on second monitor ensure the images are displayed on the second monitor.
        if self.second_mon:
            offset = 2000
        else:
            offset = 0
            
        # If the window shows the original image then left-align image on screen. Otherwise, display
        # zoomed image to the right of so it's not occluding the original image
        if image_name == 'original_image':
            cv2.moveWindow(image_name, offset, 0)
        else:
            cv2.moveWindow(image_name, offset+self.ar_size[1], 0)

    def plot_coord(self, img, x, y, image_name='original_image', zoomed=False):

        # Plot coordinates on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        if zoomed:
            radius = 0
            scale = 0.25
        else:
            radius = 3
            scale = 1
            offset = 175
            if x < self.shape[1] - offset:
                position = (x, y)
            else:
                position = (x - offset, y)
            cv2.putText(img, '('+str(x)+', '+str(y)+')', position, font, scale, (255, 255, 255), 2)
        cv2.circle(img, (x, y), radius=radius, thickness=-1, color=(255, 255, 255))
        cv2.imshow(image_name, img)
    
    def show_coords(self, img, image_name='original_image', zoom=False):

        # Load in selected_pixels.txt and plot coordinates in them
        with open(self.select_pixel_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line != '\n':
                    z_x, z_y = line.split(',')
                    if zoom:
                        s_x = np.max([0, int(z_x) - 50])
                        s_y = np.max([0, int(z_y.split('\n')[0]) - 50])
                    else:
                        s_x = int(z_x)
                        s_y = int(z_y.split('\n')[0])

                    # displaying the coordinates on the shell
                    if zoom:
                        p_x = x+s_x
                        p_y = y+s_y
                    else:
                        p_x = s_x
                        p_y = s_y

                    self.plot_coord(img, p_x, p_y, image_name)


    def save_coords(self, RPEChor=False, fovea_pit=False):
        # Read coordinates
        with open(self.select_pixel_path, 'r') as file:
            lines = file.readlines()
            if len(lines) != self.no_of_pixels:
                print(f'Selected less or more than {self.no_of_pixels} pixels. Try Again.')
                return
            
            coord_LHS = []
            coord_RHS = []
            for i, line in enumerate(lines):
                if line != '\n':
                    z_x, z_y = np.array(line.split(',')).astype(int)
                    if i % 2 == 0:
                        coord_LHS.append((z_x, z_y))
                    else:
                        coord_RHS.append((z_x, z_y))
                   
        # Create DataFrame of coordinates
        RPEChor_path = os.path.join(self.save_path, f'RPEChor_ImgNum{self.image_number}.csv')
        if fovea_pit and RPEChor is None:
            result_df = pd.DataFrame({'Patient_Group':self.which_group, 'Patient_Num':self.image_number, 'Coord':coord_LHS})
            
        elif RPEChor:
            output_df = pd.DataFrame({'Edge':'RPEChor', 'Img_Num':self.image_number, 'LHS':coord_LHS, 'RHS':coord_RHS})
            output_df.to_csv(RPEChor_path, mode='w', index_label=False)
            return 
        else:
            output_df = pd.DataFrame({'Edge':'ChorSclera', 'Img_Num':self.image_number, 'LHS':coord_LHS, 'RHS':coord_RHS})
            result_df = pd.read_csv(RPEChor_path).append(output_df)
            os.remove(RPEChor_path)
            
       
        # Save coordinates to csv file
        output_path = os.path.join(self.save_path, f'{self.output_fname}.csv')
        if os.path.exists(output_path):
            endpoint_df = pd.read_csv(output_path)
            endpoint_df = pd.concat([endpoint_df, result_df], sort=False)
            endpoint_df.to_csv(output_path, mode='w', index_label=False)
        else:
            result_df.to_csv(output_path, mode='w', index_label=False)
            
            
    def click_event_main(self, event, x, y, flags, params):
        '''
        Function to display the coordinates of the points clicked on the image
        '''
        # Load in image and extra parameters on whether image is zoomed in or not
        img_zoom, zoomed = params
        
        if self.img_or_array == 'img':
            img = cv2.imread(self.image_path, 1)
        else:
            img = self.image_path.copy()
        
        if self.img.ndim == 2:
            self.img = cv2.cvtColor(self.img.astype(np.float32), code=cv2.COLOR_GRAY2BGR)
        
        # If zoomed then change the window name
        if zoomed:
            image_name = 'zoomed'
        else:
            image_name = 'original_image'


        # If the mouse is moving then plot the coordinate onto the original window
        if event == cv2.EVENT_MOUSEMOVE:

            # Plot coordinates on image
            self.plot_coord(img, x, y, zoomed=zoomed)

            # Show current selected pixel coordinates 
            self.show_coords(img)

        # Check for SHIFT key pressed
        if event == cv2.EVENT_MBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            
            # Save selected points into csv file
            self.save_coords(fovea_pit=True, RPEChor=None)
            
            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')

            # Reset txt file, setting image_zoomed=False
            with open(self.zoom_txt_path, 'w') as file:
                file.write(f'image_zoomed=False')
            

        # Check for CTRL key pressed
        if event == cv2.EVENT_RBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            
            # Save selected points into csv file
            self.save_coords(RPEChor=True)

            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')

            # Reset txt file, setting image_zoomed=False
            with open(self.zoom_txt_path, 'w') as file:
                file.write(f'image_zoomed=False')
            
        if event == cv2.EVENT_LBUTTONDOWN and (flags & cv2.EVENT_FLAG_CTRLKEY):
            
            # Save selected points into csv file
            self.save_coords(RPEChor=False)

            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')

            # Reset txt file, setting image_zoomed=False
            with open(self.zoom_txt_path, 'w') as file:
                file.write(f'image_zoomed=False')


        # Check for right mouse clicks 
        if event == cv2.EVENT_RBUTTONDOWN:

            # Reset selected pixels by clearning txt file
            with open(self.select_pixel_path, 'w') as file:
                file.write('')


        # Check for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN and not (flags & cv2.EVENT_FLAG_CTRLKEY):

            # Show current selected pixel coordinates 
            self.show_coords(img)

            # Load in text file and check if scroll has occurred
            with open(self.zoom_txt_path, 'r') as file:

                # Read in lines and extract whether image has been zoomes
                lines = file.readlines()
                image_zoomed = lines[0].split('=')[1]

                # If zoom has occured (via scroll) then plot coordinate on original image and fill in chosen pixel
                # in red on zoomed image
                if image_zoomed == 'True\n':
                    zoom_xy = lines[1]
                    z_x, z_y = zoom_xy.split(',')
                    s_x = np.max([0, int(z_x) - 50])
                    s_y = np.max([0, int(z_y) - 50])

                    # displaying the coordinates on the shell
                    p_x = x+s_x
                    p_y = y+s_y

                    # Plot coordinates by filling pixel with red on zoomed image
                    self.plot_coord(img_zoom, x, y, 'zoomed', True)

                # If zoom has not occurred then left click corresponds to clicking pixel on original image
                else:
                    p_x = x
                    p_y = y

            # Plot coordinate on original image by default
            self.plot_coord(img, p_x, p_y, 'original_image', False)

            # Write pixel coordinate selected to file
            with open(self.select_pixel_path, 'a') as file:
                file.write(f'{p_x},{p_y}\n')                

            
        # Checking for mousewheel scrolls
        if event == cv2.EVENT_MOUSEWHEEL:

            # Extract size of image
            M, N = self.shape

            # Flags > 0 means scrolling forward (zoom in)
            if flags > 0:

                # Store coordinate that was selected during mouse scroll
                zoom_xy = (x, y)

                # Write to txt file that image was zoomed and save pixel coordinate selected during
                # mouse scroll
                with open(self.zoom_txt_path, 'w+') as file:
                    file.write(f'image_zoomed=True')
                    file.write(f'\n{x},{y}')

                # Extract region of interest of 100 x 100 centred at pixel which zoom occured at
                zoom_img = img[np.max([y-50, 0]):np.min([y+50, M]), 
                              np.max([x-50, 0]):np.min([x+50, N])]

                # Show zoomed image
                self.show_img(zoom_img, 'zoomed')

                # Setting mouse handler for the image and calling to click_main_event and sending zoomed img 
                # and zoomed=True as parameters to this callback function
                cv2.setMouseCallback('zoomed', self.click_event_main, (zoom_img, True))


            # Flags == 0 means scrolling backward (Reset zoom)
            else:
                
                # Load in text file and check if zoom has occurred    
                with open(self.zoom_txt_path, 'r') as file:

                    # Read in lines and extract whether image has been zoomed
                    lines = file.readlines()
                    image_zoomed = lines[0].split('=')[1]

                # If zoom has occured (via scroll) the backward scroll can be used to destroy the zoomed
                # window 
                if image_zoomed == 'True\n':

                    # Reset txt file, setting image_zoomed=False
                    with open(self.zoom_txt_path, 'w') as file:
                        file.write(f'image_zoomed=False')

                    # Destory zoomed window
                    cv2.destroyWindow('zoomed')

        
    def __call__(self, image_path, image_number, which_group='Donors', img_or_array='img'):
        # Plot image
        self._img_or_array = img_or_array
        if self.img_or_array == 'img':
            self.image_path = image_path
            self.img = cv2.imread(self.image_path, 1)
        else:
            self.image_path = image_path
            self.img = self.image_path
            
        if self.img.ndim == 2:
            self.img = cv2.cvtColor(self.img.astype(np.float32), code=cv2.COLOR_GRAY2BGR)
        self.image_number = image_number
        self.which_group = which_group
        self.show_img(self.img)
        params = (self.img, False)

        # Rewrite .txt file to show image_zoomed=False
        with open(self.zoom_txt_path, 'w') as file:
            file.write(f'image_zoomed=False')

        # Remove selected pixels
        with open(self.select_pixel_path, 'w') as file:
            file.write('')

        # Setting mouse handler for the image and calling the click_event() function
        cv2.setMouseCallback('original_image', self.click_event_main, params)

        # wait for Enter key to sbe pressed to exit
        k = cv2.waitKey(0)
        if k == 13:         

            # Once finished processing image delete the txt files
            os.remove(self.zoom_txt_path)
            os.remove(self.select_pixel_path)

            # close the window
            cv2.destroyAllWindows()