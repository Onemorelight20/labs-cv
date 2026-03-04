import cv2
import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.patches import Patch

IMAGE_PATH = "gradient.png"

def main():
    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayscale_input1_no_shadow.png", img)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    H, W = img.shape
    
    # Sobel Gradients
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # довжина (модуль) вектора градієнта
    magnitude = np.sqrt(Ix**2 + Iy**2)
    mag_safe = np.where(magnitude == 0, 1, magnitude) # prevent division by zero

    # normalized components of the gradient vector
    nx = Ix / mag_safe
    ny = Iy / mag_safe
    
    # Separation 
    edge_threshold = 45
    is_edge = magnitude > edge_threshold

    # Use the color thresholds to separate the shadows from the objects
    bg_color = 180
    color_tolerance = 30
    is_similar_to_bg = np.abs(img.astype(float) - float(bg_color)) < color_tolerance
    # cube_max_brightness = 155
    # is_similar_to_bg = img > cube_max_brightness

    ground_mask = (magnitude < edge_threshold) & is_similar_to_bg
    
    sin_15 = np.sin(np.radians(15))
    is_vertical_edge = is_edge & (np.abs(ny) <= sin_15) & (~ground_mask)
    is_horizontal_edge = is_edge & (~is_vertical_edge) & (~ground_mask)
    
    is_face = (~is_edge) & (~ground_mask)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sobel Magnitude
    axs[0, 0].set_title("1. Sobel Magnitude")
    im_mag = axs[0, 0].imshow(magnitude, cmap='hot')
    fig.colorbar(im_mag, ax=axs[0, 0], fraction=0.046, pad=0.04)
    axs[0, 0].axis('off')
    
    # Sobel Direction
    axs[0, 1].set_title("2. Edges orientation")
    dir_img = np.zeros((H, W, 3), dtype=np.uint8)
    dir_img[is_vertical_edge] = [255, 0, 0] 
    dir_img[is_horizontal_edge] = [0, 0, 255]
    axs[0, 1].imshow(dir_img)
    axs[0, 1].axis('off')
    
    # Gradient Vectors
    axs[1, 0].set_title("3. Gradient Vectors")
    axs[1, 0].imshow(img, cmap='gray', alpha=0.5)
    step_q = 5
    y_coords, x_coords = np.mgrid[0:H:step_q, 0:W:step_q]
    u = nx[0:H:step_q, 0:W:step_q]
    v = -ny[0:H:step_q, 0:W:step_q]  # invert for matplotlib Y-axis pointing up
    m_mask = magnitude[0:H:step_q, 0:W:step_q] > edge_threshold
    axs[1, 0].quiver(x_coords[m_mask], y_coords[m_mask], u[m_mask], v[m_mask], color='red', scale=40, headwidth=3, headlength=4)
    axs[1, 0].axis('off')

    # Segmentation
    axs[1, 1].set_title("4. Segmentation")
    seg_img = np.zeros((H, W, 3), dtype=np.uint8)
    seg_img[ground_mask] = [0, 255, 0]        # Green
    seg_img[is_face] = [200, 200, 200]        # Gray
    seg_img[is_vertical_edge] = [0, 0, 255]   # Red
    seg_img[is_horizontal_edge] = [255, 0, 0] # Blue
    axs[1, 1].imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))
    legend_elements = [
        Patch(facecolor='green', label='Ground'),
        Patch(facecolor='lightgray', label='Faces'),
        Patch(facecolor='red', label='Vertical Edges'),
        Patch(facecolor='blue', label='Horizontal Edges')
    ]
    axs[1, 1].legend(handles=legend_elements, loc='upper right')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Construct Linear System 
    N = H * W
    row_ind = []
    col_ind = []
    data = []
    b = np.zeros(N)
    
    theta = np.radians(45) # camera angle
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    tan_t = np.tan(theta)
    
    def add_coef(r, c, val):
        if 0 <= r < N and 0 <= c < N:
            row_ind.append(r)
            col_ind.append(c)
            data.append(val)
            
    print("Building constraint matrix...")
    for i in range(H):
        for j in range(W):
            k = i * W + j
            
            # Y(x,y) = 0
            if ground_mask[i, j]:
                add_coef(k, k, 1.0)
                b[k] = 0.0

            # 2Y(x) - Y(x+1) - Y(x-1) = 0
            # 4*Y(x,y) - Y(x+1,y) - Y(x-1,y) - Y(x,y+1) - Y(x,y-1) = 0
            # (1)*Up + (1)*Down + (1)*Left + (1)*Right - (4)*Center = 0
            elif is_face[i, j]:
                count = 0
                # 2Y(x,y)−Y(x+1,y)−Y(x−1,y) = 0
                if i > 0: 
                    # Y(x, y-1). W pixels back
                    count+=1
                    add_coef(k, k - W, 1.0)
                if i < H - 1: 
                    # Y(x, y+1). W pixels forward
                    count+=1
                    add_coef(k, k + W, 1.0)
                if j > 0: 
                    # Y(x-1, y). 1 pixel back
                    count+=1
                    add_coef(k, k - 1, 1.0)
                if j < W - 1: 
                    # Y(x+1, y). 1 pixel forward
                    count+=1
                    add_coef(k, k + 1, 1.0)
                
                # Add central pixel with coefficient -count
                add_coef(k, k, -count)
                b[k] = 0.0


            # Y_y = 1 / cos(theta)
            # In image Cartesian Y, Y(y) - Y(y-1) equates to Y[i] - Y[i+1]
            # Y[i]−Y[i+1]=1/cos(θ)
            elif is_vertical_edge[i, j]:
                add_coef(k, k, 1.0)
                add_coef(k, k + W, -1.0)
                b[k] = 1.0 / cos_t
                    
            # Yt = 0
            elif is_horizontal_edge[i, j]:
                t_x = -ny[i, j]
                t_y = nx[i, j]
                
                # Upwind diff with tiny regularization to prevent singularity
                diag = 1e-4

                if t_x > 0: 
                    diag += t_x
                    add_coef(k, k - 1, -t_x)
                else:       
                    diag -= t_x
                    add_coef(k, k + 1, t_x)
                if t_y > 0: 
                    diag += t_y
                    add_coef(k, k - W, -t_y)
                else:       
                    diag -= t_y
                    add_coef(k, k + W, t_y)
                
                add_coef(k, k, diag)
                b[k] = 0.0

    print("Converting to sparse matrix...")
    print(f"{N=}")
    print(f"{len(row_ind)=}")
    print(f"{len(col_ind)=}")
    print(f"{len(data)=}")
    A = sp.coo_matrix((data, (row_ind, col_ind)), shape=(N, N)).tocsr()
    print(f"{A.shape=}")
    print(f"{len(b)=}")
    
    print("Solving sparse system...")
    Y_flat = spla.spsolve(A, b)
    
    Y = Y_flat.reshape((H, W))
    
    print("Computing Z coordinates...")
    Z = np.zeros_like(Y)
    
    # Find Z from projection equation: y = Y(x,y)*cos(theta) - Z(x,y)*sin(theta)
    # Z(x,y) = (Y(x,y)*cos(theta) - y) / sin(theta)
    for i in range(H):
        y_cart = H - 1 - i
        for j in range(W):
            Z[i, j] = (Y[i, j] * cos_t - y_cart) / sin_t
        
    print("Plotting 3D result")
    step = 5
    x_plot = np.arange(0, W, step)
    y_plot = np.arange(0, H, step)
    X_grid, Y_grid = np.meshgrid(x_plot, y_plot)
    Z_plot = Z[0:H:step, 0:W:step]

    # Map the real image colors onto the surface
    color_surface_rgb = img_rgb[0:H:step, 0:W:step]
    
    # Quantize colors to ~512 max
    color_surface_rgb = (color_surface_rgb // 32) * 32 + 16
    sh_h, sh_w, _ = color_surface_rgb.shape
    
    # Create a custom discrete colorscale from the unique colors in the image
    unique_colors, inverse_indices = np.unique(color_surface_rgb.reshape(-1, 3), axis=0, return_inverse=True)
    color_indices_2d = inverse_indices.reshape((sh_h, sh_w))
    
    custom_colorscale = []
    color_count = len(unique_colors)
    if color_count <= 1:
        c = unique_colors[0] if color_count == 1 else [200, 200, 200]
        hex_c = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
        custom_colorscale = [(0.0, hex_c), (1.0, hex_c)]
        cmax_val = 1
    else:
        max_idx = color_count - 1
        for i, c in enumerate(unique_colors):
            hex_color = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
            custom_colorscale.append((i / max_idx, hex_color))
        cmax_val = max_idx

    Y_plot = Y[0:H:step, 0:W:step]

    fig = go.Figure(data=[go.Surface(
        x=X_grid,       # World X = Image X
        y=Z_plot,       # World Z = Depth into the screen
        z=Y_plot,       # World Y = Height (starts at 0 for ground)
        surfacecolor=color_indices_2d,
        colorscale=custom_colorscale,
        cmin=0,
        cmax=cmax_val,
        showscale=False
    )])
    
    fig.update_layout(title='3D Reconstructed', autosize=True,
                      scene=dict(xaxis_title='X', yaxis_title='y', zaxis_title='Z'),
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.update_scenes(yaxis_autorange="reversed")
    fig.show()

if __name__ == "__main__":
    main()
