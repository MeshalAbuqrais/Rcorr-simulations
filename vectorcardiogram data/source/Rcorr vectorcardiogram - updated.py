import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geomstats.backend as gs
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import scipy.special as sp
import csv
import pandas as pd
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean

np.set_printoptions(suppress=True, precision=5, formatter={'float_kind': '{:f}'.format})
np.random.seed(2)

sphere = Hypersphere(dim=2)
so3 = SpecialOrthogonal(n=3)

def normalize_data(data):
    """Normalize data vectors to lie on the unit sphere."""
    norm = np.linalg.norm(data, axis=-1, keepdims=True)
    norm[norm == 0] = 1  # Prevent division by zero by setting zero norms to 1
    return data / norm

def rotation_matrix(axis, theta):
    """Compute rotation matrix for rotating around a given axis by theta radians."""
    axis = normalize_data(axis[np.newaxis])[0]  # Ensure axis is a unit vector
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rotation_matrix

def rotate_and_perturbate(data, axis, angle, magnitude):
    """Rotate data points around an axis by a given angle and add random perturbation."""
    rot_mat = rotation_matrix(axis, angle)
    rotated_data = np.dot(data, rot_mat)
    perturbation = np.random.normal(0, magnitude, rotated_data.shape)
    perturbed_data = rotated_data + perturbation
    perturbed_data = normalize_data(perturbed_data)
    return perturbed_data

def sample_von_mises_fisher(mu, kappa, num_samples):
    """Sample from the von Mises-Fisher distribution on the sphere and rotate to align with mu."""
    dim = len(mu)
    r = np.random.rand(num_samples)
    w = 1 + (np.log(r) + np.log(1 + (1 - r) * np.exp(-2 * kappa))) / kappa
    v = np.random.randn(num_samples, dim - 1)
    v = normalize_data(v)
    x = np.sqrt(1 - w ** 2)[:, np.newaxis] * v
    samples = np.hstack((x, w[:, np.newaxis]))

    # Rotation to align the mean direction with mu
    mu = np.array(mu) / np.linalg.norm(mu)  # Ensure mu is a unit vector
    base_mu = np.array([0, 0, 1])  # Initial mean direction assumed for samples
    axis = np.cross(base_mu, mu)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(base_mu, mu), -1.0, 1.0))

    # Rotate samples to align with mu
    rot_mat = rotation_matrix(axis, angle)
    rotated_samples = np.dot(samples, rot_mat)

    return rotated_samples

def compute_frechet_mean(data: object, sphere: object) -> object:
    normalized_data = normalize_data(data)
    frechet_mean = FrechetMean(space=sphere)
    mean = frechet_mean.fit(normalized_data).estimate_
    return mean

def logarithmic_map_at_base_point(point, base_point):
    """Compute the logarithmic map of a point at a base point."""
    log_map_result = sphere.metric.log(base_point=base_point, point=point)
    return log_map_result

def covariance_on_manifold(X, Y, base_point):
    """
    Compute the covariance matrix on a spherical manifold at a given base point.

    Args:
    X (np.array): Data set X (N x dim), points on the sphere.
    Y (np.array): Data set Y (N x dim), points on the sphere.
    base_point (np.array): The base point on the sphere for the logarithmic maps.
    sphere (geomstats.geometry.hypersphere.Hypersphere): The sphere object.

    Returns:
    np.array: The covariance matrix computed at the base point.
    """
    # Compute log maps for each point to the tangent space at the base point
    log_maps_X = np.array([sphere.metric.log(point=x, base_point=base_point).reshape(-1) for x in X])
    log_maps_Y = np.array([sphere.metric.log(point=y, base_point=base_point).reshape(-1) for y in Y])

    # Compute the average log maps
    mean_log_map_X = np.mean(log_maps_X, axis=0)
    mean_log_map_Y = np.mean(log_maps_Y, axis=0)

    # Center the log maps by subtracting the mean log maps
    centered_log_maps_X = log_maps_X - mean_log_map_X
    centered_log_maps_Y = log_maps_Y - mean_log_map_Y

    # Compute the covariance using matrix multiplication
    covariance = np.dot(centered_log_maps_X.T, centered_log_maps_Y) / (X.shape[0] - 1)

    return covariance

def Rcov_about_point(data_1, data_2, base_point):
    cross_covariance_matrix = covariance_on_manifold(data_1, data_2, base_point)
    return np.trace(cross_covariance_matrix)

def cross_correlation_matrix_about_a_point(data_1, data_2, base_point):
    C = covariance_on_manifold(data_1, data_2, base_point)  # Cross-covariance between X and Y
    A = covariance_on_manifold(data_1, data_1, base_point)  # Covariance of X
    B = covariance_on_manifold(data_2, data_2, base_point)  # Covariance of Y

    # Calculate traces of A and B
    trace_A = np.sqrt(np.trace(A))
    trace_B = np.sqrt(np.trace(B))

    # Normalize the cross-covariance matrix C by trace(A) * trace(B)
    normalized_cross_corr_matrix = C / (trace_A * trace_B)

    return normalized_cross_corr_matrix

def Rcorr(data_1, data_2, base_point):
    cross_corr_matrix = cross_correlation_matrix_about_a_point(data_1, data_2, base_point)
    return np.trace(cross_corr_matrix)

def compute_geodesic_on_sphere(initial_point, final_point, t):
    if not 0 <= t <= 1:
        raise ValueError("Parameter t must be within the range [0, 1].")
    initial_point_normalized = normalize_data(initial_point)
    final_point_normalized = normalize_data(final_point)
    geodesic_function = sphere.metric.geodesic(initial_point=initial_point_normalized, end_point=final_point_normalized)
    geodesic_point = np.squeeze(geodesic_function(t))
    return geodesic_point

def distance_correlation_dcorr(X, Y):
    """Compute the distance correlation between two datasets X and Y on the unit sphere using geodesic distances."""
    def geodesic_distance_matrix(Z):
        """Compute the pairwise geodesic distance matrix for a dataset on the unit sphere."""
        # Ensure normalization of points on the unit sphere
        Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        # Use arc cosine of dot product to calculate geodesic distances
        dot_product_matrix = np.dot(Z_norm, Z_norm.T)
        # Clip values to avoid numerical errors beyond the interval [-1, 1]
        dot_product_matrix = np.clip(dot_product_matrix, -1.0, 1.0)
        geodesic_distance = np.arccos(dot_product_matrix)
        return geodesic_distance

    def double_centered_matrix(D):
        """Double center the distance matrix D."""
        n = D.shape[0]
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        total_mean = D.mean()
        return D - row_mean - col_mean + total_mean

    def distance_covariance(A, B):
        """Compute the distance covariance between two double-centered distance matrices A and B."""
        n = A.shape[0]
        return np.sqrt(np.sum(A * B) / (n ** 2))

    # Compute geodesic distance matrices for X and Y
    A = geodesic_distance_matrix(X)
    B = geodesic_distance_matrix(Y)

    # Double center the distance matrices
    A_centered = double_centered_matrix(A)
    B_centered = double_centered_matrix(B)

    # Compute distance covariance and variances
    dcov_AB = distance_covariance(A_centered, B_centered)
    dcov_AA = distance_covariance(A_centered, A_centered)
    dcov_BB = distance_covariance(B_centered, B_centered)

    # Compute distance correlation
    dcor = dcov_AB / np.sqrt(dcov_AA * dcov_BB)

    return dcor
data_1_BF = np.array([[5363, 5716, 6209],
                  [5610, 6623, 4966],
                  [-1665, 4812, 8606],
                  [2652, 8051, 5306],
                  [4701, 6902, 5501],
                  [1010, 4040, 9092],
                  [2555, 7348, 6283],
                  [6585, 5958, 4599],
                  [7675, 5855, 2611],
                  [4554, 4921, 7419],
                  [5553, 4264, 7140],
                  [5505, 6900, 4698],
                  [3704, 5870, 7198],
                  [6978, 7063, 1191],
                  [9064, 3591, 2223],
                  [4965, 8504, -1738],
                  [8289, 4844, 2798],
                  [4587, 7427, 4878],
                  [8746, 811, 4779],
                  [1090, 1575, 9815],
                  [5851, 5972, 5485],
                  [1409, 8086, 5712],
                  [8001, 5819, 1454],
                  [1885, 7881, 5860],
                  [-223, 1116, 9935],
                  [762, 8482, 5242],
                  [4258, 7343, 5287],
                  [5519, 5625, 6155]])

data_2_BM=np.array([[1820, 7221, 6675],
                [7261, 5691, 3859],
                [7485, 6581, 822],
                [4986, 8430, -2018],
                [5881, 3667, 7209],
                [4320, 8118, 3928],
                [5380, 8116, 2279],
                [8643, 4707, 1774],
                [7184, 6940, -487],
                [7697, 5114, 3822],
                [7550, 3576, 5497],
                [7119, 7017, 305],
                [6232, 5428, 5629],
                [8092, 5864, 351],
                [9029, 2670, -3369],
                [5086, 5805, -6358],
                [6814, 4440, 5819],
                [5086, 8279, 2365],
                [8493, 3602, 3860],
                [6572, 7097, -2540],
                [-1499, 272, 9883],
                [4645, 8479, 2555],
                [9997, 256, 0],
                [4567, 8719, 1764],
                [1415, 2437, 9595],
                [5397, 8224, 1799],
                [5573, 4698, 6847],
                [7455, 4852, 4586]])

data_3_GF=np.array([[4716, 4892, 7337],
                [4526, 7174, 5295],
               [7008, 5745, 4230],
               [2859, 6219, 7291],
               [1080, 8513, 5135],
                [2993, 7128, 6343],
                [5349, 7200, 4422],
                [3953, 6424, 6565],
                [6218, 6933, 3644],
                [2535, 9673, 0],
                [5637, 5840, 5840],
                [4330, 6900, 5799],
                [7269, 5469, 4154],
                [4468, 6906, 5687],
                [5007, 7510, 4305],
                [4707, 6276, 6201],
                [1752, 7993, 5749],
                [2892, 8942, 3418],
                [2949, 6851, 6661],
                [5933, 4638, 6580],
                [4468, 7649, 4640],
                [4923, 6105, 6203],
                [3541, 8026, 4800],
                [-5684, 3038, 7646],
                [8111, 2703, 5187]])

data_4_GM=np.array([[6160, 4909, 6160],
                [5622, 7693, 3033],
                [8330, 5355, 1388],
                [6371, 5597, 5299],
                [6759, 7364, 302],
                [2918, 7958, 5305],
                [7336, 6633, 1482],
                [7216, 6549, 2243],
                [7781, 5893, 2174],
                [6740, 6283, -3884],
                [7222, 5778, 3801],
                [5828, 7935, 1755],
                [7874, 5366, 3033],
                [5691, 5464, 6146],
                [6803, 5798, 4484],
                [6931, 7071, 1400],
                [5250, 7656, 3718],
                [4250, 8040, 4158],
                [7011, 6322, 3298],
                [6847, 3869, 6177],
                [4537, 7491, 4826],
                [6718, 6151, 4127],
                [4578, 6758, 5777],
                [-3211, 3600, 8759],
                [9278, 2741, 2530]])


def compute_and_rcorr_dcorr_for_datasets(data_X, data_Y, t):
    frechet_mean_X = compute_frechet_mean(data_X, sphere)
    frechet_mean_Y = compute_frechet_mean(data_Y, sphere)
    point_on_geodesic = compute_geodesic_on_sphere(frechet_mean_X, frechet_mean_Y, t)
    dcorr = distance_correlation_dcorr(data_X, data_Y)
    rcorr_at_geodesic_pt = Rcorr(data_X, data_Y, point_on_geodesic)

    return f"Rcorr is {rcorr_at_geodesic_pt:.5f}\ndcorr is {dcorr:.5f}"



print(compute_and_rcorr_dcorr_for_datasets(data_3_GF,data_4_GM,0.5))


def plot_datasets_on_sphere_with_means(data_X, data_Y, save_path=None, file_format='pdf', dpi=300,
                                       sphere_grid_color='black', zoom_level=1.0):
    """
    Plots datasets on the unit sphere with Fréchet means and optionally saves the plot.

    Parameters:
    - data_X, data_Y: Datasets to plot.
    - save_path: File path to save the plot. If None, the plot is shown interactively.
    - file_format: The format to save the file in (e.g., 'png', 'pdf', 'svg').
    - dpi: The resolution of the saved plot.
    - sphere_grid_color: The color of the grid lines on the sphere.
    - zoom_level: A multiplier to control the zoom level. Values > 1 zoom out, values < 1 zoom in.
    """
    # Normalize data to lie on the unit sphere
    data_X = normalize_data(data_X)
    data_Y = normalize_data(data_Y)

    # Compute Fréchet means
    frechet_mean_X = compute_frechet_mean(data_X, sphere)
    frechet_mean_Y = compute_frechet_mean(data_Y, sphere)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Remove the 3D axes, grid, and background
    ax.set_axis_off()

    # Plot data points with distinct markers and colors
    ax.scatter(data_X[:, 0], data_X[:, 1], data_X[:, 2], color='royalblue', marker='^', label='F-system', alpha=0.7, s=40)
    ax.scatter(data_Y[:, 0], data_Y[:, 1], data_Y[:, 2], color='limegreen', marker='v', label='MP-system', alpha=0.7, s=40)

    # Plot Fréchet means with bold markers and distinct colors
    ax.scatter(frechet_mean_X[0], frechet_mean_X[1], frechet_mean_X[2], color='blue', s=150,
               label='Fréchet Mean F-system', edgecolors='black', marker='X')
    ax.scatter(frechet_mean_Y[0], frechet_mean_Y[1], frechet_mean_Y[2], color='green', s=150,
               label='Fréchet Mean MP-system', edgecolors='black', marker='P')

    # Plot sphere surface with grid control
    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightgrey', alpha=0.1,
                    rstride=2, cstride=2, edgecolor=sphere_grid_color, linewidth=0.4)

    # Control the zoom level by adjusting the axis limits
    ax.set_xlim([-zoom_level, zoom_level])
    ax.set_ylim([-zoom_level, zoom_level])
    ax.set_zlim([-zoom_level, zoom_level])

    # Add a legend
    ax.legend(loc='upper left', fontsize=10)

    # Adjust the viewing angle for better perspective
    ax.view_init(elev=20, azim=30)

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format=file_format)
        print(f"Plot saved as '{save_path}' in {file_format} format.")
    else:
        plt.show()


# Example usage with zoom control
plot_datasets_on_sphere_with_means(data_1_BF, data_2_BM, save_path='vectorcardiogram data for girls using both systems.pdf', file_format='pdf',
                                   sphere_grid_color='black', zoom_level=0.8)

