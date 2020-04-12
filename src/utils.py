import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.animation as animation

import pickle

from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, classification_report,
                             confusion_matrix)

import ffmpy


def optimal_threshold_pr(precision, recall, threshs):
    f_score = (2 * precision * recall) / (precision + recall)
    ix = np.argmax(f_score)
    print('Best Threshold=%f, F-Score=%.3f' % (threshs[ix], f_score[ix]))
    print('Best precision=%f, recall=%f' % (precision[ix], recall[ix]))

    return threshs[ix]


def optimal_threshold_roc(tpr, fpr, threshs):
    g_means = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(g_means)
    print('Best Threshold=%f, G-Mean=%.3f' % (threshs[ix], g_means[ix]))
    print('Best tpr=%f, fpr=%f' % (tpr[ix], fpr[ix]))

    return threshs[ix]


def plot_pr(true_label, prediction):
    precision, recall, threshs = precision_recall_curve(true_label, prediction)
    average_precision = average_precision_score(true_label, prediction)

    # compute optimal threshold
    optimal_threshold_pr(precision, recall, threshs)

    # plot
    no_skill_pr = sum(true_label)/len(true_label)

    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')
    plt.plot([0, 1], [no_skill_pr, no_skill_pr], color='k', lw=2,
             linestyle='--', alpha=0.3)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.title('Precision-Recall curve: Average precision = {0:0.2f}'
              .format(average_precision))


def plot_roc(true_label, prediction):
    fpr, tpr, threshs = roc_curve(true_label, prediction)
    auc_res = auc(fpr, tpr)

    # compute optimal threshold
    optimal_threshold_roc(tpr, fpr, threshs)

    # plot
    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.title('Receiver operating characteristic: Area under the curve '
              '= {0:0.2f}'.format(auc_res))
    plt.legend(loc='lower right')
    plt.show()


def convert_mp4_to_gif(mp4_path, gif_path):
    ff = ffmpy.FFmpeg(
        inputs={mp4_path: None},
        outputs={gif_path: None})

    ff.run()


class SubplotAnimation(animation.TimedAnimation):

    def __init__(self, v1, v2, xlim=(-1.5, 1.5), ylim=(-2, 2),
                 threshold=None, interval=50, save_path=None,
                 show_thresholds=False, thin=False):
        '''
        An animation generator - This animation is targeted to a specific
        article regarding ROC and Precision-Recall curves.

        Creates animation and saves last frame as image. The following
        parameters refer only to the last frame that will be used as the
        image output: threshold, save_path and show_thresholds.

        Parameters
        __________
        v1 : list of scores of samples labelled as negative

        v2 : list of scores of samples labelled as positive

        xlim : x-axis limits of the plot with the scores

        ylim : y-axis limits of the plot with the scores

        threshold : to specify a threshold that will be plotted with an
            horizontal line in the last frame and based on which some metrics
            will be computed

        interval : frames speed in milliseconds on the generated animation

        save_path : where to save the as image the last frame

        show_thresholds : whether to show all the thresholds in the last frame

        thin : whether to plot thin lines

        last_frame:
            threshold - if set to a float it will plot the static figure with the defined threshold
            save_path - where to save last frame
            show_thresholds - if set to True it shows the threshold with colors on the last frame of the animation
        '''
        self.threshold = threshold
        self.save_path = save_path
        self.show_thresholds = show_thresholds
        self.thin = thin
        self.v1 = v1
        self.v2 = v2
        
        scores_sorted = np.sort(np.concatenate((self.v1[:, 1], self.v2[:, 1]),
                                               axis=None))

        # extend scores to have a threshold above highest score
        min_y_lim = scores_sorted[0]-0.05
        max_y_lim = scores_sorted[-1]+0.05
        scores_sorted = np.insert(scores_sorted, 0, min_y_lim, axis=0)
        scores_sorted = np.insert(scores_sorted, len(scores_sorted), max_y_lim,
                                  axis=0)
        self.length = len(scores_sorted)
        
        self._init_figure(xlim=xlim, ylim=ylim)
        self._make_plots(scores_sorted)
        
        animation.TimedAnimation.__init__(self, self.fig, interval=interval,
                                          blit=True)

    @staticmethod
    def _check_correctness(data, thresh, cat):
        correct = []
        wrong = []
        for i,j in data:
            if cat == 1 and j >= thresh:
                correct.append([i, j])
            elif cat == 0 and j < thresh:
                correct.append([i, j])
            else:
                wrong.append([i, j])

        return np.array(correct), np.array(wrong)
    
    def _init_figure(self, xlim, ylim, figsize=(11,11)):
        
        self.fig = plt.figure(figsize=figsize)

        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.set_ylabel('prediction score')
        self.ax1.set_xlim(xlim[0], xlim[1])
        self.ax1.set_ylim(ylim[0], ylim[1])
        self.ax1.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off

        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.plot(self.ax2.get_xlim(), self.ax2.get_ylim(), ls="--", c="k",
                      alpha=0.4)
        self.ax2.set_xlabel('1 - Specificity (False Positive Rate)')
        self.ax2.set_ylabel('Sensitivity (True Positive Rate)')
        self.ax2.set_xlim(-0.01, 1.01)
        self.ax2.set_ylim(-0.01, 1.01)     
        self.ax2.set_title('ROC curve')

        self.ax3 = self.fig.add_subplot(2, 2, 4)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.spines['top'].set_visible(False)
        no_skill_pr = len(self.v2) / (len(self.v1) + len(self.v2))
        self.ax3.plot([-2, 2], [no_skill_pr, no_skill_pr], ls="--", c="k",
                      alpha=0.4)
        self.ax3.set_xlabel('Recall (Sensitivity)')
        self.ax3.set_ylabel('Precision (PPV)')
        self.ax3.set_xlim(-0.01, 1.01)
        self.ax3.set_ylim(-0.01, 1.01)
        self.ax3.set_title('PR curve')

    def _make_scatters(self, same_marker=None):
        if same_marker:
            marker_correct = same_marker
            marker_wrong = same_marker
        else:
            marker_correct = 'o'
            marker_wrong = 'x'
        
        if self.v1_correct.size != 0:
            self.scatterplot1 = self.ax1.scatter(self.v1_correct[:, 0], 
                                                 self.v1_correct[:, 1],
                                                 alpha=0.6,
                                                 marker=marker_correct,
                                                 c='blue')

        if self.v1_wrong.size != 0:
            self.scatterplot2 = self.ax1.scatter(self.v1_wrong[:, 0], 
                                                 self.v1_wrong[:, 1],
                                                 alpha=0.6,
                                                 marker=marker_wrong,
                                                 c='blue')

        if self.v2_correct.size != 0:
            self.scatterplot3 = self.ax1.scatter(self.v2_correct[:, 0], 
                                                 self.v2_correct[:, 1],
                                                 alpha=0.6,
                                                 marker=marker_correct,
                                                 c='orange')
        if self.v2_wrong.size != 0:
            self.scatterplot4 = self.ax1.scatter(self.v2_wrong[:, 0], 
                                                 self.v2_wrong[:, 1],
                                                 alpha=0.6,
                                                 marker=marker_wrong,
                                                 c='orange')
        
    def _make_plots(self, scores_sorted):

        if self.thin:
            markersize = 3
            linewidth = 1
        else:
            markersize = None
            linewidth = 2

        self.rocX = []
        self.rocY = []

        self.prX = []
        self.prY = []

        self.v1_correct, self.v1_wrong = np.array([]), np.array([])
        self.v2_correct, self.v2_wrong = np.array([]), np.array([])

        self.x = np.linspace(5, 5, self.length)
        self.y = scores_sorted

        self.v1_correct, self.v1_wrong = self._check_correctness(self.v1,
                                                                 self.y[0],
                                                                 1)
        self.v2_correct, self.v2_wrong = self._check_correctness(self.v2,
                                                                 self.y[0],
                                                                 0)

        self._make_scatters('H')

        # init fill_between
        self.fill_bottom = self.ax1.fill_between([-20, 20], [0, 0], [0, 0],
                                                 alpha=0.08, color='blue')
        self.fill_top = self.ax1.fill_between([-20, 20], [0, 0], [0, 0],
                                              alpha=0.08, color='orange')

        # threshold line
        self.line1 = Line2D([], [], color='black', linewidth=1.5)
        self.ax1.add_line(self.line1)
        self.ax1.legend(['Threshold', 'class 0 ({} samples)'
                        .format(len(self.v1)), 'class 1 ({} samples)'
                        .format(len(self.v2))], loc='upper right',
                        fontsize='medium')

        # ROC curve line
        self.line2 = Line2D([], [], color='black', linewidth=linewidth,
                            drawstyle='steps-post', marker='.',
                            markersize=markersize)
        self.ax2.add_line(self.line2)

        # PR curve line
        self.line3 = Line2D([], [], color='black', linewidth=linewidth,
                            drawstyle='steps-post', marker='.',
                            markersize=markersize)
        self.ax3.add_line(self.line3)

    def _clean_subplot_1(self):
        self.fill_bottom.set_visible(False)
        self.fill_top.set_visible(False)
        self.line1.set_data([], [])
        
        if self.v1_correct.size != 0:
            self.scatterplot1.set_visible(False)
        if self.v1_wrong.size != 0:
            self.scatterplot2.set_visible(False)
        if self.v2_correct.size != 0:
            self.scatterplot3.set_visible(False)
        if self.v2_wrong.size != 0:
            self.scatterplot4.set_visible(False)
            
    def _save_figure(self):
        if self.save_path is not None:
            plt.savefig(self.save_path, bbox_inches='tight')
        else:
            pass
        
    def _compute_metrics(self):
        p = len(self.v2)
        n = len(self.v1)

        tp = len(self.v2_correct)
        fp = len(self.v1_wrong)

        # tn = len(self.v1_correct)
        # fn = len(self.v2_wrong)

        tpr = tp / p
        fpr = fp / n

        recall = tpr

        self.rocX.append(fpr)
        self.rocY.append(tpr)

        if tp + fp == 0:   # singularity
            self.prX.append(recall)
            self.prY.append(self.prY[-1])
        else:
            precision = tp / (tp + fp)
            self.prX.append(recall)
            self.prY.append(precision)

    def _compute_fixed_threshold(self):
        if self.threshold is not None:
            self.line1 = Line2D([-20, 20], [self.threshold, self.threshold],
                                color='black', linewidth=1.5)
            self.ax1.add_line(self.line1)

            self.v1_correct, self.v1_wrong = \
                self._check_correctness(self.v1, self.threshold, 1)
            self.v2_correct, self.v2_wrong = \
                self._check_correctness(self.v2, self.threshold, 0)
            
            true_label = np.concatenate((np.zeros(len(self.v1)),
                                         np.ones(len(self.v2))), axis=None)
            prediction = np.concatenate((self.v1[:, 1],
                                         self.v2[:, 1]), axis=None)
            
            # Area Under the Curve
            fpr, tpr, thresholds = roc_curve(true_label, prediction)
            area_under_curve = auc(fpr, tpr)
            print('AUC:', area_under_curve)
            
            # Average precision
            average_precision = average_precision_score(true_label, prediction) 
            print('Average precision:', average_precision)

            prediction[prediction >= self.threshold] = 1
            prediction[prediction < self.threshold] = 0
            
            print(classification_report(true_label, prediction))
            print(confusion_matrix(true_label, prediction))
        else:
            pass

    def _draw_frame(self, framedata):
        i = framedata
        
        # if it is last frame
        if i+1 == self.length+1:
            self._clean_subplot_1()

            self.ax2.fill_between(self.rocX, self.rocY, step='post',
                                  alpha=0.08, color='k')
            self.ax3.fill_between(self.prX, self.prY, step='post',
                                  alpha=0.08, color='k')
            self._make_scatters('o')
            self._compute_fixed_threshold()

            if self.show_thresholds:
                aux = np.linspace(0, 1, self.length)
                cmap = cm.get_cmap('cool')
                colors = [cmap(i) for i in aux]

                for idx, t in enumerate(self.y):
                    self.ax1.add_line(Line2D([-20, 20], [t, t],
                                             color=colors[idx], linewidth=0.5))

                for i in range(1, self.length):
                    self.ax2.add_line(Line2D(self.rocX[i-1:i+1],
                                             self.rocY[i-1:i+1],
                                             color=colors[i-1],
                                             linewidth=2,
                                             drawstyle='steps-post',
                                             marker='.'))
                    self.ax3.add_line(Line2D(self.prX[i-1:i+1],
                                             self.prY[i-1:i+1],
                                             color=colors[i-1],
                                             linewidth=2,
                                             drawstyle='steps-post',
                                             marker='.'))
            else:
                pass

            # save final image as pickle for further editing - used when it
            # takes some time to produce the image
            # with open('tmp_plot.pkl', 'wb') as fid:
            #     pickle.dump(self.fig, fid)

            self._save_figure()

        # if not the last frame - iterate
        else:
            self._clean_subplot_1()

            # Threshold
            self.line1.set_data([-20, 20], [self.y[i], self.y[i]])

            self.fill_bottom = self.ax1.fill_between([-20, 20], [-10, -10],
                                                     [self.y[i], self.y[i]],
                                                     alpha=0.08, color='blue')
            self.fill_top = self.ax1.fill_between([-20, 20],
                                                  [self.y[i], self.y[i]],
                                                  [10, 10], alpha=0.08,
                                                  color='orange')

            self.v1_correct, self.v1_wrong = self._check_correctness(self.v1,
                                                                     self.y[i],
                                                                     0)
            self.v2_correct, self.v2_wrong = self._check_correctness(self.v2,
                                                                     self.y[i],
                                                                     1)

            self._make_scatters()
            self._compute_metrics()

            # ROC curve
            self.line2.set_data(self.rocX, self.rocY)

            # PR curve
            self.line3.set_data(self.prX, self.prY)

            self._drawn_artists = [self.line1, self.line2, self.line3]
            
    def new_frame_seq(self):
        return iter(range(self.length+1))  # controls the i in _draw_frame
    
    def _init_draw(self):
        lines = [self.line1]
        for l in lines:
            l.set_data([], [])
