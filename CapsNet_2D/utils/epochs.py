import tensorflow as tf

def stop(history, nb_epochs, train_colum, threshold):
	if(len(history) >= nb_epochs):
		last_three_accs = [row[train_colum] for row in history][-nb_epochs:]

		acc_diffs = [abs(last_three_accs[i] - last_three_accs[i + 1]) for i in range(len(last_three_accs) - 1)]

		mean_diff_acc = sum(acc_diffs) / len(acc_diffs)

		return threshold > mean_diff_acc
	else:
		return False

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()