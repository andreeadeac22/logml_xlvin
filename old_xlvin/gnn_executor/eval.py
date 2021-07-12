import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_test_losses(dir=''):
    num_states = 100
    idx = 0
    for num_actions in [5,10,20]:
        states = pickle.load(open(dir + "results_states_" + str(num_states) + "_actions_" + str(num_actions) + ".p", 'rb'))
        losses, accs, gt_losses, gt_accs = states['losses'], states['accs'], states['gt_losses'], states['gt_accs']
        iteration_lengths = [len(graph_losses) for graph_losses in losses]
        print("Minimum ", min(iteration_lengths))
        print("Maximum ", max(iteration_lengths))

        chopped_losses = [graph_losses[:min(iteration_lengths)] for graph_losses in losses]
        chopped_gt_losses = [graph_gt_losses[:min(iteration_lengths)] for graph_gt_losses in gt_losses]
        chopped_accs = [graph_accs[:min(iteration_lengths)] for graph_accs in accs]
        chopped_gt_accs = [graph_gt_accs[:min(iteration_lengths)] for graph_gt_accs in gt_accs]

        chopped_losses = np.stack(chopped_losses, axis=0)
        chopped_gt_losses = np.stack(chopped_gt_losses, axis=0)
        chopped_accs = np.stack(chopped_accs, axis=0)
        chopped_gt_accs = np.stack(chopped_gt_accs, axis=0)

        losses_over_iter = np.mean(chopped_losses, axis=0)
        std_losses_over_iter = np.std(chopped_losses, axis=0)
        gt_losses_over_iter = np.mean(chopped_gt_losses, axis=0)

        accs_over_iter = np.mean(chopped_accs, axis=0)
        std_accs_over_iter = np.std(chopped_accs, axis=0)
        gt_accs_over_iter = np.mean(chopped_gt_accs, axis=0)

        sns.set()

        plt.figure(1)
        plt.xlabel('iteration step')
        plt.ylabel('loss')
        plt.title('Loss over iteration steps, |S|=' + str(num_states))

        plt.figure(2)
        plt.xlabel('iteration step')
        plt.ylabel('policy accuracy')
        plt.title('Policy accuracy over iteration steps, |S|=' + str(num_states))

        plt.figure(1)
        plt.yscale("log")
        plt.plot(range(min(iteration_lengths)), losses_over_iter, color='C' + str(idx), label='Predicted, |A|=' + str(num_actions))
        plt.fill_between(range(min(iteration_lengths)), losses_over_iter - std_losses_over_iter,
                         losses_over_iter + std_losses_over_iter, color='C' + str(idx), alpha=0.2)
        plt.plot(range(min(iteration_lengths)), gt_losses_over_iter, '--', label='Ground-truth, |A|=' + str(num_actions))


        plt.figure(2)
        plt.plot(range(min(iteration_lengths)), accs_over_iter, color='C' + str(idx), label='Predicted, |A|=' + str(num_actions))
        plt.fill_between(range(min(iteration_lengths)), accs_over_iter - std_accs_over_iter,
                         accs_over_iter + std_accs_over_iter, color='C' + str(idx), alpha=0.2)
        plt.plot(range(min(iteration_lengths)), gt_accs_over_iter, '--', label='Ground-truth, |A|=' + str(num_actions))

        plt.figure(1)
        plt.legend()
        plt.ylim(0.01, 100.)

        plt.figure(2)
        plt.legend()
        idx +=1

    plt.figure(1)
    plt.savefig(dir + 'loss_states_' + str(num_states) + '_actions_' + str(num_actions) + '.jpg')

    plt.figure(2)
    plt.savefig(dir + 'acc_states_' + str(num_states) + '_actions_' + str(num_actions) + '.jpg')

plot_test_losses(dir='resultsmaze_mpnn1_neighbaggr_sum_hidden_32_1/')
