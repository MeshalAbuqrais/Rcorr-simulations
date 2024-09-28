import numpy as np
import matplotlib.pyplot as plt
import geomstats.backend as gs
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from scipy.linalg import expm, logm
import csv
import pandas as pd

# Seed random number generators for reproducibility
gs.random.seed(2)
np.random.seed(2)
np.set_printoptions(suppress=True, precision=5, formatter={'float_kind': '{:f}'.format})

# Define the Special Orthogonal group SO(3)
so3 = SpecialOrthogonal(n=3, point_type='matrix')

def compute_frechet_mean(data, so3):
    frechet_mean = FrechetMean(space=so3)
    mean = frechet_mean.fit(data).estimate_
    return mean

def covariance_on_manifold(X, Y, base_point):
    log_maps_X = np.array([so3.metric.log(point=x, base_point=base_point).reshape(-1) for x in X])
    log_maps_Y = np.array([so3.metric.log(point=y, base_point=base_point).reshape(-1) for y in Y])
    mean_log_map_X = np.mean(log_maps_X, axis=0)
    mean_log_map_Y = np.mean(log_maps_Y, axis=0)
    centered_log_maps_X = log_maps_X - mean_log_map_X
    centered_log_maps_Y = log_maps_Y - mean_log_map_Y
    covariance = np.dot(centered_log_maps_X.T, centered_log_maps_Y) / (X.shape[0] - 1)
    return covariance

def Rcorr(data_1, data_2, base_point):
    C = covariance_on_manifold(data_1, data_2, base_point)
    A = covariance_on_manifold(data_1, data_1, base_point)
    B = covariance_on_manifold(data_2, data_2, base_point)
    trace_A = np.sqrt(np.trace(A))
    trace_B = np.sqrt(np.trace(B))
    normalized_cross_corr_matrix = C / (trace_A * trace_B)
    return np.trace(normalized_cross_corr_matrix)

def compute_geodesic_on_so3(initial_rotation, final_rotation, t, so3):
    geodesic_function = so3.metric.geodesic(initial_point=initial_rotation, end_point=final_rotation)
    geodesic_point = np.squeeze(geodesic_function(t))
    return geodesic_point

def distance_correlation_dcorr_so3(X, Y):
    def geodesic_distance_matrix(Z, so3):
        n = Z.shape[0]
        geodesic_distance = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                distance = so3.metric.dist(Z[i], Z[j])
                geodesic_distance[i, j] = distance
                geodesic_distance[j, i] = distance
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

    A = geodesic_distance_matrix(X, so3)
    B = geodesic_distance_matrix(Y, so3)
    A_centered = double_centered_matrix(A)
    B_centered = double_centered_matrix(B)
    dcov_AB = distance_covariance(A_centered, B_centered)
    dcov_AA = distance_covariance(A_centered, A_centered)
    dcov_BB = distance_covariance(B_centered, B_centered)
    dcor = dcov_AB / np.sqrt(dcov_AA * dcov_BB)
    return dcor

def generate_vectors(size, max_norm, seed=None):
    if seed is not None:
        np.random.seed(seed)
    vectors = np.random.randn(size, 3)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    scaling_factors = np.where(norms > max_norm, max_norm / norms, 1)
    scaled_vectors = vectors * scaling_factors
    return scaled_vectors

def rotate_vector_orth(vector, theta):
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude

    # Determine an orthogonal vector
    if unit_vector[0] != 0 or unit_vector[1] != 0:
        orthogonal_vector = np.array([-unit_vector[1], unit_vector[0], 0])
    else:
        orthogonal_vector = np.array([1, 0, 0])

    # Normalize the orthogonal vector
    orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)

    # Compute quaternion components
    a = np.cos(theta / 2.0)
    b, c, d = -orthogonal_vector * np.sin(theta / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    # Construct the rotation matrix from the quaternion
    rotation_matrix = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

    # Rotate the unit vector
    rotated_unit_vector = np.dot(rotation_matrix, unit_vector)

    # Scale back to the original magnitude
    rotated_vector = rotated_unit_vector * magnitude

    return rotated_vector

def rotate_and_perturb(vectors, angle, noise_level):
    rotated_vectors = np.array([rotate_vector_orth(vector, angle) for vector in vectors])
    perturbed_vectors = []
    for vector in rotated_vectors:
        noise = np.random.normal(scale=noise_level, size=vector.shape)
        perturbed_vector = vector + noise
        perturbed_vectors.append(perturbed_vector)
    return np.array(perturbed_vectors)

def hat_operator(vector):
    assert vector.shape == (3,), "Input vector must be 3-dimensional"
    x, y, z = vector
    skew_symmetric_matrix = np.array([[0, -z, y],
                                      [z, 0, -x],
                                      [-y, x, 0]])
    return skew_symmetric_matrix

def map_vectors_to_SO3(vectors):
    assert vectors.ndim == 2 and vectors.shape[1] == 3, "Input must be an array of shape (n_vectors, 3)"
    skew_symmetric_matrices = np.array([hat_operator(vector) for vector in vectors])
    so3_matrices = np.array([expm(matrix) for matrix in skew_symmetric_matrices])
    return so3_matrices




def compute_and_save_rcorr_dcorr_independent_datasets(max_sample_size, max_norm, seed_1, seed_2, base_point_1, base_point_2, t=0.5, filename="raw_data_independent.csv"):
    sample_sizes = np.arange(10, max_sample_size + 1, 10)  # Define sample sizes from 10 to max_sample_size with an increment of 10
    rcorr_values_geodesic = []
    dcorr_values = []

    for sample_size in sample_sizes:
        set_of_rnd_vectors_1 = generate_vectors(sample_size, max_norm,seed_1)
        X_sample = map_vectors_to_SO3(set_of_rnd_vectors_1)
        set_of_rnd_vectors_2 = generate_vectors(sample_size, max_norm,seed_2)
        Y_sample = map_vectors_to_SO3(set_of_rnd_vectors_2)
        frechet_mean_X = compute_frechet_mean(X_sample, so3)
        frechet_mean_Y = compute_frechet_mean(Y_sample, so3)
        midpoint_geodesic = compute_geodesic_on_so3(frechet_mean_X, frechet_mean_Y, t, so3)
        rcorr_mid_geodesic = Rcorr(X_sample, Y_sample, midpoint_geodesic)
        rcorr_values_geodesic.append(rcorr_mid_geodesic)
        dcorr_value = distance_correlation_dcorr_so3(X_sample, Y_sample)
        dcorr_values.append(dcorr_value)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sample Size", "Rcorr Geodesic", "dcorr"])
        for sample_size, rcorr_value, dcorr_value in zip(sample_sizes, rcorr_values_geodesic, dcorr_values):
            writer.writerow([sample_size, rcorr_value, dcorr_value])

    return sample_sizes, rcorr_values_geodesic, dcorr_values


def plot_rcorr_and_dcorr_independent_datasets_from_csv(filename, save=False, output_filename=None):
    data = pd.read_csv(filename)
    sample_sizes = data["Sample Size"]
    rcorr_values_geodesic = data["Rcorr Geodesic"]
    dcorr_values = data["dcorr"]

    plt.figure(figsize=(10, 6))  # Standardized plot size

    # Use different shades of gray, line styles, and markers
    plt.plot(sample_sizes, rcorr_values_geodesic, linestyle='-', marker='o', label='Rcorr Geodesic at t=0.5',
             color='#555555')  # Darker gray with solid line and circle markers

    # Customize the dash pattern for the dcorr plot to make it more distinguishable
    plt.plot(sample_sizes, dcorr_values, linestyle='--', marker='s', label='Distance Correlation (dcorr)',
             color='#888888', dashes=(5, 5))  # Lighter gray with dashed line and square markers

    plt.xlabel('Sample Size')
    plt.ylabel('Correlation')
#    plt.legend()  # Add the legend to distinguish the plots
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)  # Standardized tick label size
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Standardized layout adjustment

    if save and output_filename:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    else:
        plt.show()


def compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size, max_norm, seed ,angle_of_rotation, base_point_2, noise_levels, t=0.5, filename="raw_data.csv"):
    set_of_rnd_vectors = generate_vectors(sample_size, max_norm, seed)
    X_sample = map_vectors_to_SO3(set_of_rnd_vectors)
    Rcorr_values = []
    dcorr_values = []

    for noise_level in noise_levels:
        perturbed_vectors = rotate_and_perturb(set_of_rnd_vectors, angle_of_rotation, noise_level)
        skew_symmetric_matrices = np.array([hat_operator(vector) for vector in perturbed_vectors])
        Y_sample = np.array([base_point_2 @ expm(matrix) for matrix in skew_symmetric_matrices])
        mean_X = compute_frechet_mean(X_sample, so3)
        mean_Y = compute_frechet_mean(Y_sample, so3)
        geodesic_point = compute_geodesic_on_so3(mean_X, mean_Y, t, so3)
        Rcorr_value = Rcorr(X_sample, Y_sample, geodesic_point)
        dcorr_value = distance_correlation_dcorr_so3(X_sample, Y_sample)
        Rcorr_values.append(Rcorr_value)
        dcorr_values.append(dcorr_value)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Noise Level", "Rcorr", "dcorr"])
        for noise_level, Rcorr_value, dcorr_value in zip(noise_levels, Rcorr_values, dcorr_values):
            writer.writerow([noise_level, Rcorr_value, dcorr_value])

    return noise_levels, Rcorr_values, dcorr_values


def plot_Rcorr_vs_dcorr_from_computed_data_multiplicative(filename="raw_data.csv", save=False, output_filename=None):
    # Read the data from the CSV file
    data = pd.read_csv(filename)
    noise_levels = data["Noise Level"]
    Rcorr_values = data["Rcorr"]
    dcorr_values = data["dcorr"]

    # Create the plot
    plt.figure(figsize=(10, 6))  # Standardized plot size

    # Use different shades of gray, line styles, and markers
    plt.plot(noise_levels, Rcorr_values, linestyle='-', marker='o', label='Rcorr at Midpoint of Geodesic',
             color='#555555')  # Darker gray with circle markers

    # Customize the dash pattern to make it "more dashed"
    plt.plot(noise_levels, dcorr_values, linestyle='--', marker='s', label='dcorr', color='#888888',
             dashes=(5, 5))  # Larger gaps between dashes

    plt.xlabel('Noise Level')
    plt.ylabel('Correlation')
#    plt.legend()  # Add the legend to distinguish the plots
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)  # Standardized tick label size
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Standardized layout adjustment

    # Save the plot to a file if requested, otherwise display it
    if save and output_filename:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    else:
        plt.show()


# parameters for the independent datasets functions
max_sample_size = 200
max_norm = 0.6
seed_1 = 1
seed_2 = 2
base_point_1 = np.eye(3)
base_point_2 = np.eye(3)
t = 0.5

#compute_and_save_rcorr_dcorr_independent_datasets(max_sample_size, max_norm, seed_1, seed_2, base_point_1, base_point_2, t, filename="raw_data_independent_samemeanI3_maxnorm06_seeds1_2.csv")
# Plot data from CSV and save to file
#plot_rcorr_and_dcorr_independent_datasets_from_csv("raw_data_independent_samemeanI3_maxnorm06_seeds1_2.csv", save=True, output_filename="rcorr_dcorr_plot_independent_samemean_maxnorm06.pdf")



# parameters for the Rcorr vs dcorr multiplicative function
#sample_size=100
#max_norm=0.6
#seed=1
#noise_levels = np.linspace(0, 2, 41)
#angle1=0
#base_point_2_1=np.eye(3)
#angle2=np.pi/6
#base_point_2_2=expm(np.array([[0, 0, 0],
#                                      [0, 0, -1],
#                                      [0, 1, 0]]))
#E1=np.array([[0, 0, 0],
#            [0, 0, -1],
#             [0, 1, 0]])

#E2=np.array([[0, 0, -1],
#             [0, 0, 0],
#             [1, 0, 0]])
#angle3=np.pi
#base_point_2_3=np.eye(3)



#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle1,base_point_2_1,noise_levels,0.5,"maxnorm06_same_meanI3_rot0.csv")
#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle2,base_point_2_2,noise_levels,0.5,"maxnorm06_diff_mean_I3_expE1_rotpi6.csv")
#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle3,base_point_2_3,noise_levels,0.5,"maxnorm06same_meanI3_rotpi_neg_rcorr.csv")


#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle1,base_point_2_1,noise_levels,0.5,"same_meanI3_rot0.csv")
#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle2,base_point_2_2,noise_levels,0.5,"diff_mean_I3_expE1_rotpi6.csv")
#compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(sample_size,max_norm,seed,angle3,base_point_2_3,noise_levels,0.5,"same_meanI3_rotpi_neg_rcorr.csv")





#plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06_same_meanI3_rot0.csv",True,"plot_maxnorm06_same_meanI3_rot0.pdf")
#plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06_diff_mean_I3_expE1_rotpi6.csv",True,"plot_maxnorm06_diff_mean_I3_expE1_rotpi6.pdf")
#plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06same_meanI3_rotpi_neg_rcorr.csv",True,"plot_maxnorm06_same_meanI3_rotpi_neg_rcorr.pdf")

#print(compute_and_save_Rcorr_dcorr_so3_noise_multiplicaive(15,0.5,seed,angle2,base_point_2_2,noise_levels,0.5,"Test.csv"))



plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06_same_meanI3_rot0.csv",True,"grayscale_plot_maxnorm06_same_meanI3_rot0.pdf")
plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06_diff_mean_I3_expE1_rotpi6.csv",True,"grayscale_plot_maxnorm06_diff_mean_I3_expE1_rotpi6.pdf")
plot_Rcorr_vs_dcorr_from_computed_data_multiplicative("maxnorm06same_meanI3_rotpi_neg_rcorr.csv",True,"grayscale_plot_maxnorm06_same_meanI3_rotpi_neg_rcorr.pdf")

plot_rcorr_and_dcorr_independent_datasets_from_csv("raw_data_independent_samemeanI3_maxnorm06_seeds1_2.csv", save=True, output_filename="grayscale_rcorr_dcorr_plot_independent_samemean_maxnorm06.pdf")