import csv
import glob
import time
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import matplotlib.pyplot as plt

from utils import find_modality_bin_behavior, read_dataset, get_data_label_for_given_labels, reshape_data_setting1, \
    object_based_5_fold_cross_validation, repeat_trials, time_taken
from model import EncoderDecoderNetwork

from main import *

tf.set_random_seed(1)


def plot_loss_curve(cost, save_path, title_name_end, xlabel, ylabel):
    """
    Plot loss over iterations and save a plot

    :param cost:
    :param save_path:
    :param title_name_end:
    :param xlabel:
    :param ylabel:

    :return:
    """

    plt.plot(range(1, len(cost)+1), cost)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title_name = " ".join([behavior1, modality1, "TO", behavior2, modality2])
    plt.title(title_name)
    title_name_ = "_".join([behavior1, modality1, "TO", behavior2, modality2])+title_name_end
    plt.savefig(save_path+os.sep+title_name_, bbox_inches='tight', dpi=100)
    plt.close()


def save_cost_csv(cost, save_path, csv_name_end):
    """
    Save loss over iterations in a csv file

    :param cost:
    :param save_path:
    :param csv_name_end:

    :return:
    """

    csv_name = "_".join([behavior1, modality1, "TO", behavior2, modality2])+csv_name_end

    with open(save_path+os.sep+csv_name, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["epoch", "Loss"])
        for i in range(1, len(cost)+1):
            writer.writerow([i, cost[i-1]])


# Writing log file for execution time
with open(LOGS_PATH + 'time_log.txt', 'w') as time_log_file:
    time_log_file.write('Time Log\n')
    main_start_time = time.time()

"""
For all the datasets in SOURCE_DATASETS, project to all the datasets in TARGET_DATASETS
Then train classifier for generated and real data and save results
"""
for a_source_dataset in SOURCE_DATASETS:

    modality1, bins1, behavior1 = find_modality_bin_behavior(A_PATH1, a_source_dataset)
    interaction_data_1, category_labels_1, object_labels_1 = read_dataset(A_PATH1, a_source_dataset)
    num_of_features_1 = interaction_data_1.shape[-1]

    print("Source Robot: ", modality1, bins1, behavior1)
    print("Source Robot: ", interaction_data_1.shape, category_labels_1.shape)

    # Writing log file for execution time
    file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
    file.write("\n\nSource Robot: " + behavior1 + " " + modality1)
    file.close()

    for a_target_dataset in TARGET_DATASETS:
        
        modality2, bins2, behavior2 = find_modality_bin_behavior(A_PATH2, a_target_dataset)
        interaction_data_2, category_labels_2, object_labels_2 = read_dataset(A_PATH2, a_target_dataset)
        num_of_features_2 = interaction_data_2.shape[-1]

        # Both behavior cannot be same
        if behavior1 == behavior2:
            continue

        print("Target Robot: ", modality2, bins2, behavior2)
        print("Target Robot: ", interaction_data_2.shape, category_labels_2.shape)
        start_time = time.time()

        a_map_log_path = LOGS_PATH + "_".join([behavior1, modality1, "TO", behavior2, modality2]) + \
                         "_Category_" + CLF_NAME + os.sep

        os.makedirs(a_map_log_path, exist_ok=True)

        with open(a_map_log_path+os.sep+"results.csv", 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["S. No", "Target robot accuracy for only generated features",
                             "Target robot accuracy for real features corresponding to generated features",
                             "Train categories", "Test categories"])

        for a_run in range(1, RUNS+1):

            train_cat, test_cat = train_test_split(range(len(CATEGORY_LABELS)), test_size=TEST_TRAIN_RATIO)

            print("Object Categories used for Training: ", train_cat)
            print("Object Categories used for Testing: ", test_cat)

            interaction_data_1_train, category_labels_1_train = get_data_label_for_given_labels(train_cat, interaction_data_1, category_labels_1)
            interaction_data_2_train, category_labels_2_train = get_data_label_for_given_labels(train_cat, interaction_data_2, category_labels_2)

            interaction_data_1_test, category_labels_1_test = get_data_label_for_given_labels(test_cat, interaction_data_1, category_labels_1)
            interaction_data_2_test, category_labels_2_test = get_data_label_for_given_labels(test_cat, interaction_data_2, category_labels_2)

            a_map_run_log_path = a_map_log_path+os.sep+str(a_run)
            os.makedirs(a_map_run_log_path, exist_ok=True)

            tf.reset_default_graph()
            # Implement the network
            edn = EncoderDecoderNetwork(input_channels=num_of_features_1,
                                        output_channels=num_of_features_2,
                                        hidden_layer_sizes=HIDDEN_LAYER_UNITS,
                                        n_dims_code=CODE_VECTOR,
                                        learning_rate=LEARNING_RATE,
                                        activation_fn=ACTIVATION_FUNCTION)

            # Repeat trials for both robots to map each trial of the source to all trials of the target
            interaction_data_1_train_repeat, interaction_data_2_train_repeat = repeat_trials(interaction_data_1_train, interaction_data_2_train)

            # Train the network
            # cost_log = edn.train_session(interaction_data_1_train, interaction_data_2_train, a_map_run_log_path)
            # cost_log = edn.train_session(interaction_data_1_train, interaction_data_2_train, None)  # if you dont want to save graph and summary
            cost_log = edn.train_session(interaction_data_1_train_repeat, interaction_data_2_train_repeat, None)  # Repeat trials
            plot_loss_curve(cost_log, a_map_run_log_path, title_name_end="_Loss.png", xlabel='Training Iterations', ylabel='Loss')
            save_cost_csv(cost_log, a_map_run_log_path, csv_name_end="_Loss.csv")

            # Generate features using trained network
            generated_dataset = edn.generate(interaction_data_1_test)
            generated_dataset = np.array(generated_dataset)
            generated_dataset = reshape_data_setting1(NUM_OF_CATEGORY_FOR_TESTING, generated_dataset)

            # Test data loss
            test_loss = edn.rmse_loss(generated_dataset, interaction_data_2_test)
            with open(a_map_run_log_path + os.sep + "test_loss.csv", 'w') as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["Test Loss", test_loss])

            # Training on generated data and testing on read data
            generated_acc = object_based_5_fold_cross_validation(clf=CLF, data_train=generated_dataset,
                                                                 data_test=interaction_data_2_test,
                                                                 labels=category_labels_2_test,
                                                                 num_of_features=num_of_features_2)
            # If the target robot actually interacts
            actual_acc = object_based_5_fold_cross_validation(clf=CLF, data_train=interaction_data_2_test,
                                                              data_test=interaction_data_2_test,
                                                              labels=category_labels_2_test,
                                                              num_of_features=num_of_features_2)

            # Writing results of the run
            with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow([a_run, generated_acc, actual_acc, ' '.join(str(e) for e in train_cat),
                                 ' '.join(str(e) for e in test_cat)])

        print(str(RUNS)+" runs completed :)")

        # Writing log file for execution time
        file = open(LOGS_PATH + 'time_log.txt', 'a')  # append to the file created
        end_time = time.time()
        file.write("\nTarget Robot: " + behavior2+" "+modality2)
        file.write("\nTime: " + time_taken(start_time, end_time))
        file.write("\nTotal Time: " + time_taken(main_start_time, end_time))
        file.close()

        # Writing overall results
        my_data = genfromtxt(a_map_log_path+os.sep+"results.csv", delimiter=',')
        my_data = my_data[1:]
        a_list = []
        b_list = []
        a_list.append("Mean Accuracy")
        b_list.append("Standard Deviation")
        A = my_data[:, 1]
        B = my_data[:, 2]
        a_list.extend([np.mean(A), np.mean(B)])
        b_list.extend([np.std(A), np.std(B)])
        with open(a_map_log_path+os.sep+"results.csv", 'a') as f:  # append to the file created
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(a_list)
            writer.writerow(b_list)

        # Plotting average loss on training data
        all_loss = []
        for a_mapping_folder in glob.iglob(a_map_log_path + '/*/', recursive=True):
            csv_name = "_".join([behavior1, modality1, "TO", behavior2, modality2]) + "_Loss.csv"
            my_data = genfromtxt(a_mapping_folder + os.sep + csv_name, delimiter=',', usecols=(1))
            my_data = my_data[1:]
            all_loss.append(my_data)
        avg_loss = np.mean(all_loss, axis=0)
        plot_loss_curve(avg_loss, a_map_log_path, title_name_end="_Avg_Loss.png", xlabel='Training Iterations',
                        ylabel='Loss')
        save_cost_csv(avg_loss, a_map_log_path, csv_name_end="_Avg_Loss.csv")

        # Computing average loss on test data
        all_loss = []
        for a_mapping_folder in glob.iglob(a_map_log_path + '/*/', recursive=True):
            my_data = genfromtxt(a_mapping_folder + os.sep + 'test_loss.csv', delimiter=',', usecols=(1))
            all_loss.append(my_data)
        avg_loss = np.mean(all_loss, axis=0)
        with open(a_map_log_path + os.sep + "test_loss.csv", 'w') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["Test Loss", avg_loss])

        # Create lists for the plot
        materials = ['Projected Features', 'Ground Truth Features']
        x_pos = np.arange(len(materials))
        means = [np.mean(A), np.mean(B)]
        stds = [np.std(A), np.std(B)]
        title = behavior1+" "+modality1+" to "+behavior2+" "+modality2+" Category Recognition ("+CLF_NAME+")"

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel('% Recognition Accuracy')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(materials)
        ax.set_title(title)
        ax.yaxis.grid(True)

        # Save the figure and show
        plt.tight_layout()
        plt.savefig(a_map_log_path+os.sep+"bar_graph.png", bbox_inches='tight', dpi=100)
        plt.close()
