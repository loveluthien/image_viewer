import customtkinter as ctk
import tkinter as tk
import numpy as np
import scipy as sp
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import ticker
import skimage
from fast_histogram import histogram1d
import warnings


class App(ctk.CTk):
	def __init__(self):
		super().__init__()
		
		# self.title('write title to App.title')
		self.geo_xy = np.array([1400, 1080])
		self.geometry(f'{self.geo_xy[0]}x{self.geo_xy[1]}')
		self.configure(bg='#343434')

		self.canvas00_size = np.array([800, 600])
		self.zoom_level = 1
		# self.dpi = self.winfo_fpixels('1i')*2

		self.i_ch = 0 # default channel
		self.dpi = 200 # default dpi
		# self.dpi = plt.rcParams['figure.dpi']*2

		self.run()

		
	def run(self):
		
		self.open_file()
		self.load_fits()
		self.set_labels()
		self.default_img()
		self.init_img()
		self.init_histogram()
		self.set_profs_dict()
		self.init_profs(axis='x')
		self.init_profs(axis='y')
		self.init_profs(axis='z')


	def open_file(self):

		filetypes = (('fits files', '*.fits'),)	
		self.filename = tk.filedialog.askopenfilename(
			title='Open a file',
			initialdir='../alma_data',
			filetypes=filetypes)


	def load_fits(self):
		## load file ##
		fitsfile = fits.open(self.filename)
		self.header = fitsfile[0].header
		self.image_cube = fitsfile[0].data
		fitsfile.close()

		self.raw_img_size = self.image_cube[0,0,:,:].shape

		self.freqs = self.header['CRVAL3'] + np.arange(self.header['NAXIS3']) * self.header['CDELT3']
		self.freqs = self.freqs[::-1]/ 1e9


	def load_wcs(self):

		self.wcs_info = WCS(naxis=2)
		self.wcs_info.wcs.crpix = [self.header['CRPIX1']*self.zoom_level, 
									self.header['CRPIX2']*self.zoom_level]
		self.wcs_info.wcs.crval = [self.header['CRVAL1'], self.header['CRVAL2']]
		self.wcs_info.wcs.cunit = [self.header['CUNIT1'], self.header['CUNIT2']]
		self.wcs_info.wcs.ctype = [self.header['CTYPE1'], self.header['CTYPE2']]
		self.wcs_info.wcs.cdelt = [self.header['CDELT1']/self.zoom_level, 
									self.header['CDELT2']/self.zoom_level]
		self.wcs_info.array_shape = [self.header['NAXIS1']*self.zoom_level, 
										self.header['NAXIS2']*self.zoom_level]


	def set_labels(self):

		## open a file ##
		self.open_file_bottom = ctk.CTkButton(master=self, text='File', command=self.run, 
											text_color='yellow', fg_color='#343434')
		self.open_file_bottom.grid(row=0, column=0)

		## showing the image pixel and its value ##
		self.position_label = ctk.CTkLabel(master=self, text=f"Image: ; Value: ", 
											text_color='yellow', fg_color='#343434', 
											width=350, corner_radius = 5)
		self.position_label.grid(row=0, column=1)

		## showing zoom level ##
		self.zoom_level_label = ctk.CTkLabel(master=self, text=f'Zoom level: {self.zoom_level}', 
											text_color='yellow', fg_color='#343434', 
											corner_radius = 5)
		self.zoom_level_label.grid(row=0, column=2)

		## reset image to the window center ##
		self.reset_buttom = ctk.CTkButton(master=self, text = 'Reset center', 
										command=self.reset_center, 
										text_color='yellow', fg_color='#343434')
		self.reset_buttom.grid(row=0, column=3)

		## channel selection bar ##
		# self.slider = ctk.CTkSlider(master=self, from_=0, to=int(self.header['NAXIS3']-1))
		# self.slider.bind("<ButtonRelease-1>", command=self.slider_event)
		# self.slider.bind("<B1-Motion>", command=self.slider_event)
		self.slider = tk.Scale(master=self, from_=0, to=int(self.header['NAXIS3']-1), 
			tickinterval=int(self.header['NAXIS3']/5), orient='horizontal', length=400)
		self.slider.set(self.i_ch)
		self.slider.bind("<ButtonRelease-1>", self.slider_event)
		self.slider.bind("<B1-Motion>", self.slider_event)
		self.slider.grid(row=4, column=1, columnspan=2)

		## showing the channel ##
		self.ch_i_label = ctk.CTkLabel(master=self, text=f'channel: {int(self.slider.get())}', 
									corner_radius = 5)
		self.ch_i_label.grid(row=4, column=0)

		## set the image to its original size ##
		self.zoom1to1_buttom = ctk.CTkButton(master=self, text = '1:1', 
										command=self.zoom_1to1, width=50,
										text_color='yellow', fg_color='#343434')
		self.zoom1to1_buttom.grid(row=0, column=4)

		## set the image to a size which fits the window size ##
		self.zoomdefualt_buttom = ctk.CTkButton(master=self, text = 'fit window', 
										command=self.zoom_default, width=100,
										text_color='yellow', fg_color='#343434')
		self.zoomdefualt_buttom.grid(row=0, column=5)


	def set_clip_entry(self):

		self.c_min_label = ctk.CTkLabel(master=self, text=f'Clip min:', text_color='yellow', 
										fg_color='#343434', corner_radius = 5)
		self.c_min_label.grid(row=4, column=4)
		self.c_min = ctk.CTkEntry(master=self, placeholder_text=f"{self.vmin:.3f}")
		self.c_min.grid(row=4, column=5, sticky='W')
		self.c_min.bind('<Return>', command=self.update_clip_min)

		self.c_max_label = ctk.CTkLabel(master=self, text=f'Clip max:', text_color='yellow', 
										fg_color='#343434', corner_radius = 5)
		self.c_max_label.grid(row=4, column=6)
		self.c_max = ctk.CTkEntry(master=self, placeholder_text=f"{self.vmax:.3f}")
		self.c_max.grid(row=4, column=7, sticky='W')
		self.c_max.bind('<Return>', command=self.update_clip_max)


	def update_clip_max(self, value):

		self.vmax = float(self.c_max.get())

		self.img_obj.set_clim(self.vmin, self.vmax)
		# self.canvas00.draw()
		self.canvas00.draw_idle()

		self.hist_vmax.set_data([self.vmax, self.vmax], [0, self.max_hist*self.y_fac])
		self.canvas.draw_idle()


	def update_clip_min(self, value):

		self.vmin = float(self.c_min.get())

		self.img_obj.set_clim(self.vmin, self.vmax)
		# self.canvas00.draw()
		self.canvas00.draw_idle()

		self.hist_vmin.set_data([self.vmin, self.vmin], [0, self.max_hist*self.y_fac])
		self.canvas.draw_idle()


	def slider_event(self, value):

		self.i_ch = int(self.slider.get())
		self.ch_i_label.configure(text=f'channel: {self.i_ch}', fg_color='#343434')
		
		self.resize_img()
		self.update_img()
		self.make_hist()
		self.update_hist()
		self.update_prof(axis='x')
		self.update_prof(axis='y')


	def reset_center(self):

		# self.disp_img_center = self.canvas00_size[0]/2, self.canvas00_size[1]/2
		self.disp_img_center = self.img_size[0]/2, self.img_size[1]/2
		# self.canvas00.create_image(self.disp_img_center[0], self.disp_img_center[1], 
		# 							anchor=tk.CENTER, image=self.tk_image00)

		self.ax_img.set_xlim( self.disp_img_center[0] - self.canvas00_size[0]/2 , 
				self.disp_img_center[0] + self.canvas00_size[0]/2)
		self.ax_img.set_ylim( self.disp_img_center[1] - self.canvas00_size[1]/2 , 
				self.disp_img_center[1] + self.canvas00_size[1]/2)
		
		self.canvas00.draw_idle()


	def resize_img(self):

		self.img = self.image_cube[0,self.i_ch,:,:]
		down_factor = int(1 / self.zoom_level)
		
		## if the image size greater than the canvas size, downsampling the image 
		if down_factor > 1:
			self.img = skimage.measure.block_reduce(self.img, 
								(down_factor, down_factor), np.nanmean)
		
		elif down_factor < 1:
			# self.img = skimage.transform.resize(self.img, 
			# np.array(self.img_size)*self.zoom_level)
			self.img = sp.ndimage.zoom(self.img, self.zoom_level, order=0)
		
		self.img_size = self.img.shape


	def default_img(self):
		
		## determine the initial image size ##
		self.img_size = np.array(self.raw_img_size, dtype=int)
		
		## if the image size greater than the canvas size ## 
		dim = 2
		for i in range(dim):
			while self.img_size[i] > self.canvas00_size[i]:
				self.img_size = self.img_size / 2
				self.zoom_level = self.zoom_level / 2

		## if the image size smaller than the canvas size ## 
		while (self.canvas00_size[0]/self.img_size[0] > 2) & \
			(self.canvas00_size[1]/self.img_size[1] > 2):
			self.img_size = self.img_size * 2
			self.zoom_level = self.zoom_level * 2

		self.default_zoom_level = self.zoom_level

		self.resize_img()


	def init_img(self):

		self.load_wcs()
		
		self.disp_img_center = self.img_size[0]/2, self.img_size[1]/2

		# figsize = (int(self.raw_img_size[0]/dpi), int(self.raw_img_size[1]/dpi))
		# figsize = (int(self.canvas00_size[0]/self.dpi), int(self.canvas00_size[1]/self.dpi))
		figsize = (self.canvas00_size[0]/self.dpi, self.canvas00_size[1]/self.dpi)
		
		## using wcs coordinates ##
		self.fig_img, self.ax_img = plt.subplots(figsize=figsize, 
												subplot_kw={'projection': self.wcs_info})
		self.fig_img.set_facecolor("#343434")
		self.fig_img.subplots_adjust(left=0.1, right=0.99, bottom=0.08, top=0.99, 
									wspace=0, hspace=0)

		self.ax_img.set_facecolor("#343434")
		self.ax_img.tick_params(direction='in', length=2, width=1, 
					colors='cyan', labelsize=5,
					grid_color='cyan', grid_alpha=0.8, 
					grid_linestyle = '--', grid_linewidth=0.5,
					pad=1.2)
		self.ax_img.tick_params(axis='y', labelrotation=90, pad=0.1)
		self.ax_img.spines['bottom'].set_color('cyan')
		self.ax_img.spines['left'].set_color('cyan')
		self.ax_img.spines['right'].set_color('cyan')
		self.ax_img.spines['top'].set_color('cyan')
		self.ax_img.grid()
		self.ax_img.set_xlabel('R.A.', fontsize = 5, color='cyan')
		self.ax_img.set_ylabel('Dec.', fontsize = 5, color='cyan')
		self.ax_img.set_xlim( self.disp_img_center[0] - self.canvas00_size[0]/2 , 
				self.disp_img_center[0] + self.canvas00_size[0]/2)
		self.ax_img.set_ylim( self.disp_img_center[1] - self.canvas00_size[1]/2 , 
				self.disp_img_center[1] + self.canvas00_size[1]/2)


		self.data_percent = 99 # %
		self.vmax = np.nanpercentile(self.image_cube[0,self.i_ch,:,:].flat[:], 
										50 + self.data_percent/2 )
		self.vmin = np.nanpercentile(self.image_cube[0,self.i_ch,:,:].flat[:], 
										50 - self.data_percent/2 )

		self.img_cmap = matplotlib.cm.inferno
		self.img_cmap.set_bad(color='grey')
		self.img_obj = self.ax_img.imshow(self.img, origin='lower', 
						vmin=self.vmin, vmax=self.vmax, cmap = self.img_cmap)

		self.canvas00 = FigureCanvasTkAgg(self.fig_img, master=self)
		self.canvas00.get_tk_widget().grid(row=1, column=0, columnspan=4, rowspan=2)
		# self.canvas00.get_tk_widget().bind('<Motion>', self.show_cursor_position)
		self.canvas00.mpl_connect('motion_notify_event', self.show_cursor_position)
		self.canvas00.get_tk_widget().bind('<MouseWheel>', self.on_mousewheel)
		self.canvas00.get_tk_widget().bind('<Button-1>', self.apply_new_center)
		
		self.zoom_level_label.configure(text=f"Zoom Level: {self.zoom_level}")


	def update_img(self):

		self.img_obj.set_data(self.img)
		# self.data_percent = 99 # default 99%
		self.vmax = np.nanpercentile(self.image_cube[0,self.i_ch,:,:].flat[:], 
										50 + self.data_percent/2 )
		self.vmin = np.nanpercentile(self.image_cube[0,self.i_ch,:,:].flat[:], 
										50 - self.data_percent/2 )
		self.img_obj.set_clim(self.vmin, self.vmax)
		# self.canvas00.draw()
		self.canvas00.draw_idle()

	def update_img_profs(self):

		self.resize_img()
		self.zoom_level_label.configure(text=f"Zoom Level: {self.zoom_level}")
		self.init_img()
		self.update_prof(axis='x')
		self.update_prof(axis='y')


	def zoom_in(self):

		## the maximum zoom level is 4 ##
		self.zoom_level = min(self.zoom_level*2, 4.0)
		self.update_img_profs()

	def zoom_out(self):

		## the minimum zoom level is 1/8 ##
		self.zoom_level = max(self.zoom_level/2, 1/8)
		self.update_img_profs()


	def zoom_default(self):

		## set the image to the default size ##
		self.zoom_level = self.default_zoom_level
		self.update_img_profs()


	def zoom_1to1(self):
		
		## set the image to the original size ##
		self.zoom_level = 1
		self.update_img_profs()


	def apply_new_center(self, event):

		# self.disp_img_center = (event.x, event.y)
		dx = self.canvas00_size[0]/2 - event.x
		dy = self.canvas00_size[1]/2 - event.y
		
		new_center_x = self.disp_img_center[0] - dx
		new_center_y = self.disp_img_center[1] + dy

		self.disp_img_center = (new_center_x, new_center_y)


		self.ax_img.set_xlim( self.disp_img_center[0] - self.canvas00_size[0]/2 , 
				self.disp_img_center[0] + self.canvas00_size[0]/2)
		self.ax_img.set_ylim( self.disp_img_center[1] - self.canvas00_size[1]/2 , 
				self.disp_img_center[1] + self.canvas00_size[1]/2)


		self.canvas00.draw_idle()
		

	def init_histogram(self):

		self.y_fac = 1.1 ## additional space for y axis

		# figsize = (int(self.canvas00_size[0]/self.dpi), int(self.canvas00_size[1]/self.dpi/2))
		figsize = (self.canvas00_size[0]/self.dpi, self.canvas00_size[1]/self.dpi/2)

		self.fig, self.ax = plt.subplots(figsize=figsize)
		self.fig.subplots_adjust(left=0.1, right=0.99, bottom=0.25, top=0.99, 
									wspace=0, hspace=0)
		self.fig.set_facecolor(color='#343434')

		self.ax.set_facecolor(color='#343434')
		self.ax.tick_params(axis='both', which='both', colors='cyan', labelsize=5, 
							length=0, pad=0.2)
		self.ax.spines['bottom'].set_color(None)
		self.ax.spines['left'].set_color(None)
		self.ax.spines['right'].set_color(None)
		self.ax.spines['top'].set_color(None)

		self.make_hist()
		self.hist_plot, = self.ax.plot(self.bins, self.hist_data, drawstyle='steps', 
										color='w', lw=0.5)
		self.max_hist = np.nanmax(self.hist_data)
		self.hist_vmin, = self.ax.plot([self.vmin, self.vmin], [0, self.max_hist], lw=0.5, c='y')
		self.hist_vmax, = self.ax.plot([self.vmax, self.vmax], [0, self.max_hist], lw=0.5, c='y')
		self.ax.grid(lw=0.5, ls='--')
		self.ax.set_xlabel(f"Value ({self.header['BUNIT']})", fontsize = 5, color='cyan')
		self.ax.set_ylabel(r'log$_{10}$(Counts)', fontsize = 5, color='cyan')
		# self.canvas.draw()
		
		self.canvas = FigureCanvasTkAgg(self.fig, master=self)
		self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=4)
		self.canvas.draw_idle()

		self.set_clip_entry()


	def make_hist(self):

		self.hist_bin_num = 1000
		ch_slice = self.image_cube[0,self.i_ch,:,:].flat[:]
		self.bins = np.linspace(np.nanmin(ch_slice), np.nanmax(ch_slice), self.hist_bin_num+1)[:-1]
		self.hist_data = histogram1d(ch_slice, range=[np.nanmin(ch_slice), np.nanmax(ch_slice)], 
									bins=self.hist_bin_num)
		self.hist_data = np.log10(self.hist_data)


	def update_hist(self):
		
		self.hist_plot.set_data(self.bins, self.hist_data)
		self.max_hist = np.nanmax(self.hist_data)
		
		self.hist_vmin.set_data([self.vmin, self.vmin], [0, self.max_hist*self.y_fac])
		self.hist_vmax.set_data([self.vmax, self.vmax], [0, self.max_hist*self.y_fac])
		
		if len(np.where(np.isfinite(self.hist_data))[0]) > 0:
			self.ax.set_ylim(0, self.max_hist*self.y_fac)
		self.ax.set_xlim(np.min(self.bins), np.max(self.bins))

		self.canvas.draw_idle()

	# def prepare_hist(self):
	# 	self.hist_bin_num = 1000
	# 	self.bins = np.zeros((self.hist_bin_num, self.header['NAXIS3']))
	# 	self.hist_data = np.zeros((self.hist_bin_num, self.header['NAXIS3']))
	# 	for i in range(self.header['NAXIS3']):
	# 		ch_slice = self.image_cube[0,i,:,:].flat[:]
	# 		self.bins[:,i] = np.linspace(np.nanmin(ch_slice), np.nanmax(ch_slice), self.hist_bin_num+1)[:-1]
	# 		self.hist_data[:,i] = histogram1d(ch_slice, range=[np.nanmin(ch_slice), np.nanmax(ch_slice)], bins=self.hist_bin_num)
		
	# 	self.hist_data = np.log10(self.hist_data)


	def set_profs_dict(self):

		## Dictionaries for storing the profile figures and parameters ##
		self.fig_profs = {}
		self.ax_profs = {}
		self.canvas_profs = {}
		self.profs = {}
		self.xpix, self.ypix = 0, 0
		self.raw_xpix, self.raw_ypix = 0, 0

		
	def init_profs(self, axis='x'):

		grid_pos = {'x': 1, 'y': 2, 'z': 3}
		xlabels = {'x': 'x (pixel)', 'y': 'x (pixel)', 'z': 'Frequency (GHz)'}
		
		# figsize = (2.5, int(self.canvas00_size[1]/self.dpi/2))
		# figsize = (int(self.canvas00_size[0]/self.dpi * 0.8), int(self.canvas00_size[1]/self.dpi/2))
		figsize = (self.canvas00_size[0]/self.dpi * 0.6, self.canvas00_size[1]/self.dpi/2)
		self.fig_profs[axis], self.ax_profs[axis] = plt.subplots(figsize=figsize)
		self.fig_profs[axis].subplots_adjust(left=0.2, right=0.95, bottom=0.25, top=0.99, 
											wspace=0, hspace=0)
		self.fig_profs[axis].set_facecolor(color='#343434')
		self.ax_profs[axis].set_facecolor(color='#343434')
		self.ax_profs[axis].tick_params(axis='both', which='both', colors='cyan', 
										labelsize=5, length=0, pad=0.3)
		self.ax_profs[axis].spines['bottom'].set_color(None)
		self.ax_profs[axis].spines['left'].set_color(None)
		self.ax_profs[axis].spines['right'].set_color(None)
		self.ax_profs[axis].spines['top'].set_color(None)
		self.ax_profs[axis].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2e}"))


		# self.profs[axis], = self.ax_profs[axis].plot(range(len(profile[axis])), profile[axis], drawstyle='steps')
		self.profs[axis], = self.ax_profs[axis].plot([], [], drawstyle='steps', 
													lw=0.5, color='w')
		# self.profs[axis], = self.ax_profs[axis].plot([], [], marker='.', lw=0.5, color='w')
		self.ax_profs[axis].grid(lw=0.5, ls='--')
		self.ax_profs[axis].set_xlabel(xlabels[axis], fontsize = 5, color='cyan')
		self.ax_profs[axis].set_ylabel(f"Value ({self.header['BUNIT']})", 
										fontsize = 5, color='cyan')
		
		self.canvas_profs[axis] =FigureCanvasTkAgg(self.fig_profs[axis], master=self)
		self.canvas_profs[axis].get_tk_widget().grid(row=grid_pos[axis], 
													column=4, columnspan=4)
		self.canvas_profs[axis].draw_idle()


	def update_prof(self, axis='x'):

		profile = {}
		profile['x'] = self.image_cube[0, self.i_ch, self.raw_ypix, :] # x-profile
		profile['y'] = self.image_cube[0, self.i_ch, :, self.raw_xpix] # y-profile
		
		# if axis == 'z':
		# 	if self.zoom_level < 1:
		# 		st_pix = int(self.xpix * self.zoom_level)
		# 		end_pix = st_pix + int(1/self.zoom_level)
		# 		profile['z'] = np.nanmean(np.nanmean(
		# 						self.image_cube[0, :, st_pix: end_pix, st_pix: end_pix], 
		# 						axis=2), axis=1) # x-profile
		# 	else:
		# 		profile['z'] = self.image_cube[0, :, self.ypix, self.xpix]

		# 	profile['z'] = profile['z'][::-1]

		profile['z'] = self.image_cube[0, ::-1, self.raw_ypix, self.raw_xpix]

		x_axis = {'x': range(len(profile[axis])), 
				'y': range(len(profile[axis])), 
				'z': self.freqs }

		self.profs[axis].set_data(x_axis[axis], profile[axis])
		
		x_bound = min(self.img_size[0], self.canvas00_size[0])
		y_bound = min(self.img_size[1], self.canvas00_size[1])
		prof_limxy = {'x': ( (self.disp_img_center[0] - x_bound/2)/self.zoom_level , 
							(self.disp_img_center[0] + x_bound/2)/self.zoom_level), 
					'y': ( (self.disp_img_center[1] - y_bound/2)/self.zoom_level , 
							(self.disp_img_center[1] + y_bound/2)/self.zoom_level),
					'z': (np.min(self.freqs), np.max(self.freqs))}

		self.ax_profs[axis].set_xlim(prof_limxy[axis])

		if len(np.where(np.isfinite(profile[axis]))[0]) > 0:
			self.ax_profs[axis].set_ylim(np.nanmin(profile[axis])*self.y_fac, 
											np.nanmax(profile[axis])*self.y_fac)
		
		self.canvas_profs[axis].draw_idle()


	## cursur motion ##
	def show_cursor_position(self, event):
		# x = event.x
		# y = event.y

		if event.inaxes and (
			(event.xdata >= 0) & (event.xdata < self.img_size[0]) 
			& (event.ydata >= 0) & (event.ydata < self.img_size[1])):
			self.xpix = int(event.xdata)
			self.ypix = int(event.ydata)
			# print(event.inaxes)
			# self.position_label.configure(text=f"Image: ({x}, {y}); ({xdata}, {ydata})")
			intensity = self.img[self.ypix, self.xpix]

			self.raw_xpix = int(self.xpix/ self.zoom_level)
			self.raw_ypix = int(self.ypix/ self.zoom_level)
			self.position_label.configure(
				text=f"Image: ({self.raw_xpix}, {self.raw_ypix}); " +
					f"Value: {intensity:0.3e} ({self.header['BUNIT']})")
			
		else:
			self.position_label.configure(text=f"Image: ({np.nan}, {np.nan})" + 
											f"; Value: {np.nan} ")

		self.update_prof(axis='x')
		self.update_prof(axis='y')
		self.update_prof(axis='z')


	def on_mousewheel(self, event):
		# Determine the direction of the mouse wheel scroll
		if event.delta > 0:
			self.zoom_in()
		else:
			self.zoom_out()


if __name__ == '__main__':

	warnings.filterwarnings("ignore")

	app = App()
	app.title('Image viewer')

	app.mainloop()
	



