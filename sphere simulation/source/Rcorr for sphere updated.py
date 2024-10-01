import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    norm = np.linalg.norm(data, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    return data / norm

def rotation_matrix(axis, theta):

    axis = normalize_data(axis[np.newaxis])[0] 
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rotation_matrix = np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                                [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                                [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
    return rotation_matrix

def rotate_and_perturbate(data, axis, angle, magnitude):

    rot_mat = rotation_matrix(axis, angle)
    rotated_data = np.dot(data, rot_mat)
    perturbation = np.random.normal(0, magnitude, rotated_data.shape)
    perturbed_data = rotated_data + perturbation
    perturbed_data = normalize_data(perturbed_data)
    return perturbed_data

def sample_von_mises_fisher(mu, kappa, num_samples):
    dim = len(mu)
    r = np.random.rand(num_samples)
    w = 1 + (np.log(r) + np.log(1 + (1 - r) * np.exp(-2 * kappa))) / kappa
    v = np.random.randn(num_samples, dim - 1)
    v = normalize_data(v)
    x = np.sqrt(1 - w ** 2)[:, np.newaxis] * v
    samples = np.hstack((x, w[:, np.newaxis]))

    mu = np.array(mu) / np.linalg.norm(mu) 
    base_mu = np.array([0, 0, 1]) 
    axis = np.cross(base_mu, mu)
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(base_mu, mu), -1.0, 1.0))

    rot_mat = rotation_matrix(axis, angle)
    rotated_samples = np.dot(samples, rot_mat)

    return rotated_samples

def compute_frechet_mean(data: object, sphere: object) -> object:
    normalized_data = normalize_data(data)
    frechet_mean = FrechetMean(space=sphere)
    mean = frechet_mean.fit(normalized_data).estimate_
    return mean

def logarithmic_map_at_base_point(point, base_point):
    log_map_result = sphere.metric.log(base_point=base_point, point=point)
    return log_map_result

def covariance_on_manifold(X, Y, base_point):

    log_maps_X = np.array([sphere.metric.log(point=x, base_point=base_point).reshape(-1) for x in X])
    log_maps_Y = np.array([sphere.metric.log(point=y, base_point=base_point).reshape(-1) for y in Y])

    mean_log_map_X = np.mean(log_maps_X, axis=0)
    mean_log_map_Y = np.mean(log_maps_Y, axis=0)

    centered_log_maps_X = log_maps_X - mean_log_map_X
    centered_log_maps_Y = log_maps_Y - mean_log_map_Y

    covariance = np.dot(centered_log_maps_X.T, centered_log_maps_Y) / (X.shape[0] - 1)

    return covariance

def Rcov_about_point(data_1, data_2, base_point):
    cross_covariance_matrix = covariance_on_manifold(data_1, data_2, base_point)
    return np.trace(cross_covariance_matrix)

def cross_correlation_matrix_about_a_point(data_1, data_2, base_point):
    C = covariance_on_manifold(data_1, data_2, base_point) 
    A = covariance_on_manifold(data_1, data_1, base_point) 
    B = covariance_on_manifold(data_2, data_2, base_point)
    trace_A = np.sqrt(np.trace(A))
    trace_B = np.sqrt(np.trace(B))
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

    def geodesic_distance_matrix(Z):
        Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        dot_product_matrix = np.dot(Z_norm, Z_norm.T)
        dot_product_matrix = np.clip(dot_product_matrix, -1.0, 1.0)
        geodesic_distance = np.arccos(dot_product_matrix)
        return geodesic_distance

    def double_centered_matrix(D):
        n = D.shape[0]
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        total_mean = D.mean()
        return D - row_mean - col_mean + total_mean

    def distance_covariance(A, B):
        n = A.shape[0]
        return np.sqrt(np.sum(A * B) / (n ** 2))
    A = geodesic_distance_matrix(X)
    B = geodesic_distance_matrix(Y)

    A_centered = double_centered_matrix(A)
    B_centered = double_centered_matrix(B)

    dcov_AB = distance_covariance(A_centered, B_centered)
    dcov_AA = distance_covariance(A_centered, A_centered)
    dcov_BB = distance_covariance(B_centered, B_centered)

    dcor = dcov_AB / np.sqrt(dcov_AA * dcov_BB)

    return dcor

def compute_and_save_rcorr_dcorr_noise_sphere(sample_size, mu, kappa, axis_of_rotation, angle_of_rotation, t,
                                              filename="sphere_data.csv"):
    noise_levels = np.linspace(0, 2, num=41)
    rcorr_geodesic = []
    dcorr_values = []

    axis_normalized = normalize_data(axis_of_rotation.reshape(1, -1))[0]

    for epsilon in noise_levels:
        data_X = sample_von_mises_fisher(mu, kappa, sample_size)
        data_Y = rotate_and_perturbate(data_X, axis_normalized, angle_of_rotation, epsilon)

        frechet_mean_X = compute_frechet_mean(data_X, sphere)
        frechet_mean_Y = compute_frechet_mean(data_Y, sphere)

        point_on_geodesic = compute_geodesic_on_sphere(frechet_mean_X, frechet_mean_Y, t)

        rcorr_mid_geodesic = Rcorr(data_X, data_Y, point_on_geodesic)
        rcorr_geodesic.append(rcorr_mid_geodesic)

        dcorr_value = distance_correlation_dcorr(data_X, data_Y)
        dcorr_values.append(dcorr_value)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Noise Level", "Rcorr", "dcorr"])
        for noise_level, rcorr_value, dcorr_value in zip(noise_levels, rcorr_geodesic, dcorr_values):
            writer.writerow([noise_level, rcorr_value, dcorr_value])

    return noise_levels, rcorr_geodesic, dcorr_values


def plot_rcorr_and_dcorr_vs_noise_sphere_from_csv(filename, save=False, output_filename=None):
    data = pd.read_csv(filename)
    noise_levels = data["Noise Level"]
    rcorr_geodesic = data["Rcorr"]
    dcorr_values = data["dcorr"]

    plt.figure(figsize=(10, 6))

    plt.plot(noise_levels, rcorr_geodesic, linestyle='-', marker='o', label='Rcorr at Midpoint of Geodesic',
             color='#555555')  # Darker gray with solid line and circular markers

    plt.plot(noise_levels, dcorr_values, linestyle='--', marker='s', label='dcorr', color='#888888',
             dashes=(5, 5))  # Lighter gray with dashed line and square markers

    plt.xlabel('Noise Level')
    plt.ylabel('Correlation')
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    if save and output_filename:
        plt.savefig(output_filename, format='pdf')

    plt.show()


def compute_and_save_rcorr_dcorr_vs_sample_size_sphere_independent_case(max_sample_size, mu1, mu2, kappa1, kappa2,
                                                                        t=0.5, filename="sample_size_data.csv"):
    sample_sizes = np.arange(10, max_sample_size + 1, 10)
    rcorr_values_geodesic = []
    dcorr_values = []

    for sample_size in sample_sizes:
        data_X = sample_von_mises_fisher(mu1, kappa1, sample_size)
        data_Y = sample_von_mises_fisher(mu2, kappa2, sample_size)

        frechet_mean_X = compute_frechet_mean(data_X, sphere)
        frechet_mean_Y = compute_frechet_mean(data_Y, sphere)

        midpoint_geodesic = compute_geodesic_on_sphere(frechet_mean_X, frechet_mean_Y, t)

        rcorr_mid_geodesic = Rcorr(data_X, data_Y, midpoint_geodesic)
        rcorr_values_geodesic.append(rcorr_mid_geodesic)

        dcorr_value = distance_correlation_dcorr(data_X, data_Y)
        dcorr_values.append(dcorr_value)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sample Size", "Rcorr", "dcorr"])
        for sample_size, rcorr_value, dcorr_value in zip(sample_sizes, rcorr_values_geodesic, dcorr_values):
            writer.writerow([sample_size, rcorr_value, dcorr_value])

    return sample_sizes, rcorr_values_geodesic, dcorr_values


def plot_rcorr_and_dcorr_vs_sample_size_sphere_independent_case_from_csv(filename, save=False, output_filename=None):
    data = pd.read_csv(filename)
    sample_sizes = data["Sample Size"]
    rcorr_values_geodesic = data["Rcorr"]
    dcorr_values = data["dcorr"]

    plt.figure(figsize=(10, 6))

    plt.plot(sample_sizes, rcorr_values_geodesic, linestyle='-', marker='o', label='Rcorr Geodesic at t=0.5',
             color='#555555') 

    plt.plot(sample_sizes, dcorr_values, linestyle='--', marker='s', label='Distance Correlation (dcorr)',
             color='#888888', dashes=(5, 5))

    plt.xlabel('Sample Size')
    plt.ylabel('Correlation')

    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save and output_filename:
        plt.savefig(output_filename, format='pdf')

    plt.show()


sample_size = 100
mu = np.array([0, 0, 1])
kappa = 9
t = 0.5

axis_of_rotation1 = np.array([0, 1, 0])
angle1=0

axis_of_rotation2=np.array([0,1,0])
angle2=np.pi/6

axis_of_rotation3=np.array([1,1,1])
angle3=np.pi



compute_and_save_rcorr_dcorr_noise_sphere(sample_size,mu,kappa,axis_of_rotation1,angle1,0.5,"same_mean_rot0.csv")
compute_and_save_rcorr_dcorr_noise_sphere(sample_size,mu,kappa,axis_of_rotation2,angle2,0.5,"diff_mean_axisrot010_rotpi6.csv")
compute_and_save_rcorr_dcorr_noise_sphere(sample_size,mu,kappa,axis_of_rotation3,angle3,0.5,"diff_mean_axisrot111_rotpi.csv")

plot_rcorr_and_dcorr_vs_noise_sphere_from_csv("same_mean_rot0.csv", save=True, output_filename="grayscale_same_mean_rot0_plot.pdf")
plot_rcorr_and_dcorr_vs_noise_sphere_from_csv("diff_mean_axisrot010_rotpi6.csv", save=True, output_filename="grayscale_diff_mean_axisrot010_rotpi6_plot.pdf")
plot_rcorr_and_dcorr_vs_noise_sphere_from_csv("diff_mean_axisrot111_rotpi.csv", save=True, output_filename="grayscale_diff_mean_axisrot111_rotpi_plot.pdf")


# Define parameters for the sample size independent case function
max_sample_size = 200
mu1 = np.array([0, 0, 1])
mu2 = np.array([0, 1, 1])
kappa1 = 4
kappa2 = 5

compute_and_save_rcorr_dcorr_vs_sample_size_sphere_independent_case(max_sample_size, mu1, mu2, kappa1, kappa2, t=0.5, filename="independentcase_diffmean_mu1_001_kappa1_4_mu2_011_kappa2_5.csv")

plot_rcorr_and_dcorr_vs_sample_size_sphere_independent_case_from_csv("independentcase_diffmean_mu1_001_kappa1_4_mu2_011_kappa2_5.csv", save=True, output_filename="grayscale_independentcase_diffmean_mu1_001_kappa1_4_mu2_011_kappa2_5_plot.pdf")



def visualize_data_on_sphere_with_perturbation(mu_X, kappa, num_samples, angle, axis, magnitude, save=False,
                                               filename=False, file_type='pdf'):

    data_X = sample_von_mises_fisher(mu_X, kappa, num_samples)

    data_Y_transformed = rotate_and_perturbate(data_X, axis, angle, magnitude)


    frechet_mean_X = compute_frechet_mean(data_X, sphere)
    frechet_mean_Y_transformed = compute_frechet_mean(data_Y_transformed, sphere)

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), subplot_kw={'projection': '3d'})

    u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    for ax in axes:
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    axes[0].scatter(data_X[:, 0], data_X[:, 1], data_X[:, 2], color='royalblue', s=5, label='Original Data')


    axes[0].scatter(*frechet_mean_X, color='green', s=99, marker='D', label='Fréchet Mean (Original)')


    axes[1].scatter(data_Y_transformed[:, 0], data_Y_transformed[:, 1], data_Y_transformed[:, 2], color='red', s=5,
                    label='Perturbed Data')


    axes[1].scatter(*frechet_mean_Y_transformed, color='green', s=100, marker='D', label='Fréchet Mean (Transformed)')


    axes[0].set_title('Original dataset')
    axes[1].set_title('Transformed dataset')

    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='Data Points', markerfacecolor='royalblue', markersize=6),
        Line2D([0], [0], marker='D', color='w', label='Fréchet Mean', markerfacecolor='green', markersize=6)]
    custom_legend1 = [Line2D([0], [0], marker='o', color='w', label='Data Points', markerfacecolor='red', markersize=6),
                      Line2D([0], [0], marker='D', color='w', label='Fréchet Mean', markerfacecolor='green',
                             markersize=6)]
    axes[0].legend(handles=custom_legend)
    axes[1].legend(handles=custom_legend1)

    if save:
        full_filename = f"{filename}.{file_type}"
        plt.savefig(full_filename, bbox_inches='tight', pad_inches=0.05)

    plt.show()


# parameters for the plot

mu_X = np.array([-1, 1, 1])
kappa = 15
num_samples = 200
angle = np.pi / 4 
axis = np.array([-1, 0, 1])
magnitude = 0.2

visualize_data_on_sphere_with_perturbation(mu_X, kappa, num_samples, angle, axis, magnitude, save=True,
                                           filename='plot-sphere-dataset_and_dependent_dataset', file_type='pdf')
