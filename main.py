import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
import warnings

from sklearn import metrics
from train import Train

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    auc_result, fprs, tprs = Train(path='new_data/test_output_path.txt',
                                   output_file='TF_Target_Disease/output_first', dim=450, window_size=2,
                                   iterations=2, batch_size=32, care_type=0, initial_lr=0.001, min_count=1,
                                   num_workers=10, random_seed=2)
    tpr = []
    mean_fpr = np.linspace(0, 1, 10000)
    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC  (AUC = %.4f)' % ( auc_result[i]),color = 'b')
    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc_result)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')     
    plt.legend(loc='lower right')
    plt.savefig('AUC')
    plt.show()

