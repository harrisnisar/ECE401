import numpy as np
from PIL import Image
import submitted
import h5py

###############################################################################
# Here is the code that gets called if you type python mp3.py on command line.
# For each step of processing:
#  1. do the step
#  2. pop up a window to show the result  in a figure.
#  3. wait for you to close the window, then continue.
# Almost all of the code below is just creating the figures to show you.
# None of the code below will get run by the autograder;
# if you want to see what the autograder does, type python run_tests.py
# on the command line.
if __name__=="__main__":
    import tkinter as tk
    from matplotlib.backend_bases import key_press_handler
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.figure, argparse
    parser = argparse.ArgumentParser(
        description='''Run MP1 using provided image file, and show popup windows of results.''',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s','--steps',nargs='*',
                        help='''
                        Perform only the specified steps (should be a space-separated list).
                        Results of any preceding steps are read from solutions.hdf5.
                        This is intended so that you can focus debugging on the step that you specify.
                        ''')
    parser.add_argument('-i','--imagefile',default='image.jpg',nargs='?',
                        help='''Process the specified image file.''')
    parser.add_argument('-n','--nopictures',action='store_true',
                        help='''
                        Don't show pictures, just do the computations.
                        This may be useful if you want to run your code until it crashes,
                        in order to quickly find bugs.
                        ''')
    parser.add_argument('-o','--outputfile',nargs='?',default=False,
                        help='''
                        If this option is provided, it should specify an HDF5 filename 
                        to which results will be saved.
                        ''')
    args = parser.parse_args()
    if args.steps is not None:
        solutions = h5py.File('solutions.hdf5','r')

    class PlotWindow(object):
        '''
        Pop up a window containing a matplotlib Figure.
        The NavigationToolbar2TK allows you to save the figure in a file.
        The key_press_handler  permits standard key events as described at
        https://matplotlib.org/3.3.0/users/interactive.html#key-event-handling
        '''
        def __init__(self, fig):
            import tkinter
            self.root = tkinter.Tk()
            self.canvas = FigureCanvasTkAgg(fig, master=self.root)
            toolbar = NavigationToolbar2Tk(self.canvas, self.root)
            self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
            self.canvas.mpl_connect("key_press_event", lambda e: key_press_handler(e,self.canvas,toolbar))
            button=tkinter.Button(master=self.root, text="Close and continue", command=self.quit)
            button.pack(side=tkinter.BOTTOM)
            self.root.mainloop()
        def quit(self):
            self.root.quit()     # stops mainloop
            self.root.destroy()  # this is necessary on Windows

    ##############################################################
    # Step 0: load the data, and show the original image
    original = np.asarray(Image.open(args.imagefile)).astype('float64')
    if args.steps is None or '0' in args.steps:
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            ax = fig.subplots()
            ax.imshow(original.astype(int))
            ax.set_title('Here is the original image!')
            PlotWindow(fig)

    ##############################################################
    # Step 1: create a rectangular filter, and show it in a plot
    if args.steps is None or '1' in args.steps:
        (n_rect, h_rect) = submitted.todo_rectangular_filter()
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(6, 4))
            axs = fig.subplots(3,1,sharex=True)
            axs[0].stem(n_rect, h_rect, markerfmt='D',use_line_collection=True)
            axs[0].set_title('Rectangular smoothing filter')
            (n_step, x_step) = submitted.unit_step()
            axs[1].stem(n_step, x_step, markerfmt='D',use_line_collection=True)
            axs[1].set_title('Unit step')
            y_step = np.convolve(x_step, h_rect, mode='same')
            axs[2].stem(n_step, y_step, markerfmt='D',use_line_collection=True)
            axs[2].set_title('Rectangular-smoothed Unit step')
            PlotWindow(fig)
    else:
        n_rect = solutions['n_rect']
        h_rect = solutions['h_rect']

    ##############################################################
    # Step 2: Use h_rect to smooth the rows of the image
    if args.steps is None or '2' in args.steps:
        smoothed_rows = submitted.todo_convolve_rows(original, h_rect)
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            ax = fig.subplots()
            ax.clear()
            ax.imshow(smoothed_rows.astype(int))
            ax.set_title('Image with rows smoothed')
            PlotWindow(fig)
    else:
        smoothed_rows = solutions['smoothed_rows']

    ###############################################################
    # Step 3: Use h_rect to smooth the columns of the image
    if args.steps is None or '3' in args.steps:
        smoothed_image = submitted.todo_convolve_columns(smoothed_rows, h_rect)
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            ax = fig.subplots()
            ax.clear()
            ax.imshow(smoothed_image.astype(int))
            ax.set_title('Image with both rows and columns smoothed')
            PlotWindow(fig)
    else:
        smoothed_image = solutions['smoothed_image']

    ##############################################################
    # Step 4: create a backward-difference filter, and show it in a plot
    if args.steps is None or '4' in args.steps:
        (n_diff, h_diff) = submitted.todo_backward_difference()
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            axs = fig.subplots(3,1,sharex=True)
            axs[0].stem(n_diff, h_diff, markerfmt='D',use_line_collection=True)
            axs[0].set_title('Backward-Difference filter')
            (n_step, x_step) = submitted.unit_step()
            axs[1].stem(n_step, x_step, markerfmt='D',use_line_collection=True)
            axs[1].set_title('Unit step')
            y_step = np.convolve(x_step, h_diff, mode='same')
            axs[2].stem(n_step, y_step, markerfmt='D',use_line_collection=True)
            axs[2].set_title('Differenced Unit step')
            PlotWindow(fig)
    else:
        n_diff = solutions['n_diff']
        h_diff = solutions['h_diff']

    ##############################################################
    # Step 5: create a Gaussian smoothing filter, and show it in a plot
    if args.steps is None or '5' in args.steps:
        (n_gauss, h_gauss) = submitted.todo_gaussian_smoother()
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            axs = fig.subplots(3,1,sharex=True)
            axs[0].stem(n_gauss, h_gauss, markerfmt='D',use_line_collection=True)
            axs[0].set_title('Gaussian smoothing filter')
            (n_step, x_step) = submitted.unit_step()
            axs[1].stem(n_step, x_step, markerfmt='D',use_line_collection=True)
            axs[1].set_title('Unit step')
            y_step = np.convolve(x_step, h_gauss, mode='same')
            axs[2].stem(n_step, y_step, markerfmt='D',use_line_collection=True)
            axs[2].set_title('Gaussian-smoothed Unit step')
            PlotWindow(fig)
    else:
        n_gauss = solutions['n_gauss']
        h_gauss = solutions['h_gauss']


    ##############################################################
    # Step 6: create a difference-of-Gaussians filter, and show it in a plot
    if args.steps is None or '6' in args.steps:
        (n_dog, h_dog) = submitted.todo_difference_of_gaussians()
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            axs = fig.subplots(3,1,sharex=True)
            axs[0].stem(n_dog, h_dog, markerfmt='D',use_line_collection=True)
            axs[0].set_title('Difference-of-Gaussians filter')
            (n_step, x_step) = submitted.unit_step()
            axs[1].stem(n_step, x_step, markerfmt='D',use_line_collection=True)
            axs[1].set_title('Unit step')
            y_step = np.convolve(x_step, h_dog, mode='same')
            axs[2].stem(n_step, y_step, markerfmt='D',use_line_collection=True)
            axs[2].set_title('DoG-filtered Unit step')
            PlotWindow(fig)
    else:
        n_dog = solutions['n_dog']
        h_dog = solutions['h_dog']

    ###############################################################
    # Step 7: Use DoG filter to compute Gx and Gy, then normalize
    # within each color plane
    if args.steps is None or '7' in args.steps:
        tmp = submitted.todo_convolve_rows(original, h_dog)
        hgrad = submitted.todo_normalize_colors(np.abs(tmp))
        tmp = submitted.todo_convolve_columns(original, h_dog)
        vgrad = submitted.todo_normalize_colors(np.abs(tmp))
        if not args.nopictures:
            fig = matplotlib.figure.Figure(figsize=(10, 4))
            axs = fig.subplots(1,2)
            axs[0].imshow(hgrad)
            axs[0].set_title('Horizontal grad magnitude')
            axs[1].imshow(vgrad)
            axs[1].set_title('Vertical grad magnitude')
            PlotWindow(fig)
    else:
        hgrad = solutions['hgrad']
        vgrad = solutions['vgrad']

    ###############################################################
    # Step 8: Gradient magnitude
    if args.steps is None or '8' in args.steps:
        tmp = submitted.todo_gradient_magnitude(hgrad, vgrad)        
        gradient_magnitude = submitted.todo_normalize_colors(tmp)
        if not args.nopictures:        
            fig = matplotlib.figure.Figure(figsize=(5, 4))
            ax = fig.subplots()
            ax.imshow(gradient_magnitude)
            ax.set_title('Gradient magnitude')
            PlotWindow(fig)
    else:
        gradient_magnitude = solutions['gradient_magnitude']

    ###############################################################
    # Now create an hdf5 file with your results
    if args.outputfile:
        with h5py.File(args.outputfile, 'w') as f:
            f.create_dataset('n_rect', data=n_rect)
            f.create_dataset('h_rect', data=h_rect)
            f.create_dataset('smoothed_rows', data=smoothed_rows)
            f.create_dataset('smoothed_image', data=smoothed_image)
            f.create_dataset('n_diff', data=n_diff)
            f.create_dataset('h_diff', data=h_diff)
            f.create_dataset('n_gauss', data=n_gauss)
            f.create_dataset('h_gauss', data=h_gauss)
            f.create_dataset('n_dog', data=n_dog)
            f.create_dataset('h_dog', data=h_dog)
            f.create_dataset('hgrad', data=hgrad)
            f.create_dataset('vgrad', data=vgrad)
            f.create_dataset('gradient_magnitude',data=gradient_magnitude)

    print('Done!  Now try python grade.py.')
