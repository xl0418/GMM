import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation


class GMM2:
    def __init__(self, data, prior_mean, prior_cov, no_iter):
        self.iter = no_iter
        self.output_mean = np.zeros((no_iter, 4))
        self.output_mean[0, :] = prior_mean
        self.output_var = np.zeros((no_iter, 8))
        self.output_var[0, :] = prior_cov
        self.data = data

    def run_GMM2(self):

        for ii in range(no_iter - 1):

            prior_mean = self.output_mean[ii, :].reshape((2, 2))
            prior_mean1 = prior_mean[0]
            prior_mean2 = prior_mean[1]

            prior_var = self.output_var[ii, :].reshape((2, 4))
            prior_var1 = prior_var[0, :].reshape((2, 2))
            prior_var2 = prior_var[1, :].reshape((2, 2))

            data1 = np.array([]).reshape(0, 2)
            data2 = np.array([]).reshape(0, 2)
            no_data = self.data.shape[0]

            for data_ii in range(no_data):
                # The probability of the data point belongs to the 1st cluster
                pro1 = multivariate_normal.pdf(self.data[data_ii],
                                               mean=prior_mean1,
                                               cov=prior_var1)

                # The probability of the data point belongs to the 2nd cluster
                pro2 = multivariate_normal.pdf(self.data[data_ii],
                                               mean=prior_mean2,
                                               cov=prior_var2)
                if pro1 > pro2:
                    # collect the data belonging to the 1st cluster
                    data1 = np.vstack((data1, self.data[data_ii]))
                else:
                    # collect the data belonging to the 2nd cluster
                    data2 = np.vstack((data2, self.data[data_ii]))

            # calculate the new mean for the clusters
            update_mean1 = np.mean(data1, axis=0)
            update_mean2 = np.mean(data2, axis=0)
            # the new cov for the clusters
            update_cov1 = np.cov(data1[:, 0], data1[:, 1])
            update_cov2 = np.cov(data2[:, 0], data2[:, 1])

            self.output_mean[ii + 1, :] = np.concatenate((update_mean1, update_mean2), axis=None)
            self.output_var[ii + 1, :] = np.concatenate((update_cov1.flatten(), update_cov2.flatten()), axis=None)

        return self.output_mean, self.output_var


if __name__ == '__main__':
    # generate the data
    true_mean1 = [10, 12]
    true_cov1 = [[5, 2], [2, 6]]
    x1, y1 = np.random.multivariate_normal(true_mean1, true_cov1, 5000).T

    true_mean2 = [20, 22]
    true_cov2 = [[10, 6], [6, 12]]
    x2, y2 = np.random.multivariate_normal(true_mean2, true_cov2, 5000).T

    data_x = np.concatenate((x1, x2))
    data_y = np.concatenate((y1, y2))

    data = np.stack((data_x, data_y), axis=1)
    data1 = np.stack((x1, y1), axis=1)
    data2 = np.stack((x2, y2), axis=1)

    # the first guess of the mean and cov for the 2 clusters
    prior_mean = np.zeros(4)
    prior_cov = np.zeros(8)
    prior_mean[:] = [20, 15, 22, 30]
    prior_cov[:] = [3, 2, 1, 8, 4, 3, 5, 9]

    # Number of iterations
    no_iter = 100

    # Initialize the class and run GMM2 for 2 clusters
    rungmm2 = GMM2(data, prior_mean, prior_cov, no_iter)
    output_mean, output_var = rungmm2.run_GMM2()

    # Output the means and vars for the 2 clusters
    last_row = 1  # output_mean.shape[0]
    out1_mean = output_mean[last_row - 1, 0:2]
    out2_mean = output_mean[last_row - 1, 2:4]

    out1_cov = np.zeros((2, 2))
    np.fill_diagonal(out1_cov, output_var[last_row - 1, 0:2])
    out2_cov = np.zeros((2, 2))
    np.fill_diagonal(out2_cov, output_var[last_row - 1, 2:4])

    # The static plot
    x, y = np.mgrid[5:30:.01, 5:30:.01]
    pos = np.dstack((x, y))
    rv1 = multivariate_normal(out1_mean, out1_cov)
    rv2 = multivariate_normal(out2_mean, out2_cov)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(x1, y1, 'x', color="blue", alpha=0.3)
    ax2.plot(x2, y2, 'x', color="orange", alpha=0.3)
    ax2.contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=1)
    ax2.contour(x, y, rv2.pdf(pos), cmap='Oranges', alpha=1)
    plt.show()

    ### Animation

    fig, ax = plt.subplots(figsize=(7, 5))


    def animation_func(i, output_mean, output_var, data1, data2):
        ax.clear()
        ax.plot(data1[:, 0], data1[:, 1], 'x', color="blue", alpha=0.3)
        ax.plot(data2[:, 0], data2[:, 1], 'x', color="orange", alpha=0.3)

        # last_row = output_mean.shape[0]
        out1_mean = output_mean[i, 0:2]
        out2_mean = output_mean[i, 2:4]

        output_var = output_var[i, :].reshape((2, 4))
        out1_cov = output_var[0, :].reshape((2, 2))
        out2_cov = output_var[1, :].reshape((2, 2))

        x, y = np.mgrid[5:30:.01, 5:30:.01]
        pos = np.dstack((x, y))
        rv1 = multivariate_normal(out1_mean, out1_cov)
        rv2 = multivariate_normal(out2_mean, out2_cov)

        ax.contour(x, y, rv1.pdf(pos), cmap='Blues', alpha=1)
        ax.contour(x, y, rv2.pdf(pos), cmap='Oranges', alpha=1)


    no_frame = range(no_iter)

    animation = FuncAnimation(fig, animation_func,
                              frames=np.arange(0, no_iter, 1), fargs=(output_mean, output_var, data1, data2,),
                              interval=100)
    plt.show()

    animation.save('test3.gif', writer='imagemagick', fps=10)

