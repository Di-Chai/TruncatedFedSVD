import time
import copy
import logging
import datetime
import argparse

from utils import *
from data_loader import *
from paths import *
from sklearn.decomposition import TruncatedSVD


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pc', '-k', type=int, default=2)
    parser.add_argument('--num_participants', '-p', type=int, default=10)
    parser.add_argument('--num_samples', '-s', type=int, default=1000)
    parser.add_argument('--dataset', '-d', type=str, default='load_synthetic')
    parser.add_argument('--block', '-b', type=int, default=10)
    parser.add_argument('--num_feature', '-f', type=int, default=1000)
    parser.add_argument('--only_time', '-t', type=str, default='False')
    parser.add_argument('--output_pkl', '-o', type=str, default='False')
    parser.add_argument('--log_dir', '-l', type=str, default='')
    args = parser.parse_args()

    # Parameters
    num_participants = args.num_participants
    num_samples = args.num_samples  # per participant

    only_evaluate_time = True if args.only_time == 'True' else False
    save_pickle = True if args.output_pkl == 'True' else False

    # Generate data using Gaussian distribution
    X = np.random.randn(args.num_feature, args.num_participants*args.num_samples)
    label = None
    
    print('PCA mode, subtracting the mean')
    X = X.T
    X -= np.mean(X, axis=0)
    X = X.T

    # Split the data for participants
    Xs = [X[:, e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    if label is not None:
        Ys = [label[e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    else:
        Ys = None

    # Init the efficiency file
    log_file_name = os.path.join(log_dir, args.log_dir,
                                 'FedSVD_' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(levelname)s: %(message)s',
                        filename=log_file_name)
    logging.info(str(args))
    print(str(args))

    ground_truth = np.concatenate(Xs, axis=1)
    m, n = ground_truth.shape

    mape_denominator = copy.deepcopy(ground_truth)
    mape_denominator[np.where(mape_denominator == 0)] = 1e-10

    # Standalone SVD
    start = time.time()
    truncated_svd = TruncatedSVD(n_components=args.num_pc, algorithm='arpack')
    truncated_svd.fit(ground_truth.T)
    normal_components = truncated_svd.components_
    normal_explained_var = truncated_svd.explained_variance_
    normal_explained_var_ratio = truncated_svd.explained_variance_ratio_
    time_standalone_svd = time.time() - start
    logging.info('StandalonePCA time %s' % time_standalone_svd)
    print('StandalonePCA time %s' % time_standalone_svd)
    logging.info('StandalonePCA explained var ratio %s ' % normal_explained_var_ratio)
    print('StandalonePCA explained var ratio %s ' % normal_explained_var_ratio)

    # FedSVD, Start the simulation
    logging.info('Simulation Start!')

    comm_each_data_holder = 0
    
    # Masking Server: Generate random orthogonal matrix P and Q
    start = time.time()
    P = generate_orthogonal_matrix(n=X.shape[0], reuse=False, block_reduce=args.block)
    t1 = time.time()
    Q = generate_orthogonal_matrix(n=np.sum([e.shape[1] for e in Xs]), reuse=False, block_reduce=args.block)
    t2 = time.time()
    Qs = [Q[e * num_samples: e * num_samples + num_samples] for e in range(num_participants)]
    time_generate_orthogonal = t2 - start
    if not only_evaluate_time:
        comm_each_data_holder += get_object_size(P)
        comm_each_data_holder += (get_object_size(Qs) / num_participants)
    logging.info('Generate orthogonal matrix %s done. Using %s seconds.' % (P.shape[0], t1-start))
    logging.info('Generate orthogonal matrix %s done. Using %s seconds.' % (Q.shape[0], t2-start))
    print('Generate orthogonal matrix done. Using %s seconds.' % time_generate_orthogonal)

    # Data Holders & Factorization Server: SecureAggregation to get X'
    start = time.time()
    X_mask_partitions = []
    for i in range(num_participants):
        X_mask_partitions.append(P @ Xs[i] @ Qs[i])
    X_mask, comm_size = secure_aggregation(X_mask_partitions, only_evaluate_time)
    # The time consumption of applying random mask runs in parallel by all the participants
    time_apply_orthogonal = (time.time() - start) / num_participants
    comm_each_data_holder += comm_size
    logging.info('Apply distortion done. Using %s seconds.' % time_apply_orthogonal)
    print('Apply distortion done. Using %s seconds.' % time_apply_orthogonal)

    # Decrypt the distorted_X, and perform the SVD decomposition
    start = time.time()
    truncated_fed_svd = TruncatedSVD(n_components=args.num_pc, algorithm='arpack')
    truncated_fed_svd.fit(X_mask.T)
    fed_components_mask = truncated_fed_svd.components_
    fed_explained_var = truncated_fed_svd.explained_variance_
    fed_explained_var_ratio = truncated_fed_svd.explained_variance_ratio_
    time_svd = time.time() - start
    logging.info('SVD done. Using %s seconds.' % time_svd)
    print('SVD done. Using %s seconds.' % time_svd)
    logging.info('Truncated FedSVD explained var ratio %s ' % fed_explained_var_ratio)
    print('Truncated FedSVD explained var ratio %s ' % fed_explained_var_ratio)

    # Recover the real singular values and vectors
    start = time.time()
    fed_components = (P.T @ fed_components_mask.T).T

    # Evaluation (FedSVD): reconstruct to measure the precision
    mape_to_normal = np.mean(np.abs((np.abs(normal_components) - np.abs(fed_components)) / np.abs(normal_components)))
    logging.info('MAPE to normal PCA %s ' % mape_to_normal)
    print('MAPE to normal PCA %s ' % mape_to_normal)

    # End Simulation
    logging.info('Finished!')
    # Collect the time consumption
    time_consumption = [time_generate_orthogonal, time_apply_orthogonal, time_svd]
    logging.info('Truncated FedSVD totally uses %s seconds' % np.sum(time_consumption))
    print('Truncated FedSVD totally uses %s seconds' % np.sum(time_consumption))

    if save_pickle:
        # Save the results to pickle file
        result = {
            'dataset': args.dataset,
            'num_participants': args.num_participants,
            'num_samples': args.num_samples,
            'Xs': Xs,
            'Ys': Ys,
            'block_reduce': args.block,
            'fed_svd': {
                'var': fed_explained_var,
                'var_ratio': fed_explained_var_ratio,
                'pcs': fed_components
            },
            'standalone_svd': {
                'var': normal_explained_var,
                'var_ratio': normal_explained_var_ratio,
                'pcs': normal_components
            },
            'time_consumption': time_consumption,
            'comm_size': comm_each_data_holder,
            'mape_to_normal': mape_to_normal,
            'log': log_file_name
        }

        with open(os.path.join(results_dir, save_file_name + '.pkl'), 'wb') as f:
            pickle.dump(result, f)