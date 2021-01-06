from AI_COVID19.init import *

# loss Accuracy graph
def loss_acc_graph(history_save_path, model_name, metric):
    df_history = pd.read_csv(os.path.join(history_save_path, model_name+'_history.csv'))
    acc = list(df_history[metric[0]])
    val_acc = list(df_history['val_'+metric[0]])
    loss = list(df_history['loss'])
    val_loss = list(df_history['val_loss'])
    lr = list(df_history['lr'])
    epochs = range(len(acc))
    # Accuracy graph
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, acc, 'b', label='Training acc = {}%'.format(np.around(np.max(acc) * 100, decimals=1)))
    plt.plot(epochs, val_acc, 'r', label='Validation acc = {}%'.format(np.around(np.max(val_acc) * 100, decimals=1)))
    plt.title('{} Accuracy (Total Epoch = {})'.format(model_name, len(acc)), fontsize=15, y=1.02)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(history_save_path, model_name+'_acc_fig.png'))
    # Loss graph
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, loss, 'b', label='Training loss = {}'.format(np.around(np.min(loss), decimals=3)))
    plt.plot(epochs, val_loss, 'r', label='Validation loss = {}'.format(np.around(np.min(val_loss), decimals=3)))
    plt.title('{} Loss (Total Epoch = {})'.format(model_name, len(loss)), fontsize=15, y=1.02)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(history_save_path, model_name+'_loss_fig.png'))
    # Learning rate graph
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, lr, 'g', label='Learning Rate = {}'.format(np.around(np.min(lr), decimals=3)))
    plt.title('{} Learning Rate (Total Epoch = {})'.format(model_name, len(lr)), fontsize=15, y=1.02)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.tick_params(axis='both', which='major', length=10, width=1, direction='in')
    plt.legend(fontsize=15)
    plt.savefig(os.path.join(history_save_path, model_name + '_learning_fig.png'))


def train_evaluator(save_model, inputs, targets, batch, train_dict, network_dict, save_path, model_name):
    # Train Result Evaluate
    val_results = save_model.evaluate(inputs['val'], targets['val'], batch_size=batch)
    test_results = save_model.evaluate(inputs['test'], targets['test'], batch_size=batch)
    # Config Save
    config = {}
    config.update({'# train_parameters #': '#####'})
    config.update(train_dict)
    config.update({'# network_parameters #': '#####'})
    config.update(network_dict)
    config.update({'# evaluate_result #': '#####'})
    config.update({'validation_loss': val_results[0], 'validation_acc': val_results[1]})
    config.update({'test_loss': test_results[0], 'test_acc': test_results[1]})
    df_config = pd.DataFrame({'keys': list(config.keys()), 'values': list(config.values())})
    df_config.to_csv(os.path.join(save_path, model_name + '_config.csv'))
    return {'val': val_results, 'test': test_results}


def onehot_class_converter(label_list, mode):
    new_label_list = []
    if mode == '4cls':
        for i, i_ in enumerate(label_list):
            if np.max(i_) == i_[0]:
                new_label_list.append('Covid19')
            elif np.max(i_) == i_[1]:
                new_label_list.append('Bacterial')
            elif np.max(i_) == i_[2]:
                new_label_list.append('Viral')
            elif np.max(i_) == i_[3]:
                new_label_list.append('Normal')
        return new_label_list
    elif mode == '3cls':
        for i, i_ in enumerate(label_list):
            if np.max(i_) == i_[0]:
                new_label_list.append('Covid19')
            elif np.max(i_) == i_[1]:
                new_label_list.append('Pneumonia')
            elif np.max(i_) == i_[2]:
                new_label_list.append('Normal')
        return new_label_list


def train_predictor(save_model, patient_list, inputs, targets, data_type, batch, save_path, model_name, class_mode):
    # Train Result Prediction
    results = save_model.predict(inputs.astype(dtype=np.float16), batch_size=batch)
    # Prediction Save
    data_dic = {'patient_name': patient_list, 'onehot_class': list(targets), 'predict_result': list(results),
                'class': onehot_class_converter(list(targets), class_mode), 'predict_class': onehot_class_converter(list(results), class_mode)}
    df_data = pd.DataFrame(data_dic)
    df_data.to_csv(os.path.join(save_path, model_name + '_' + data_type + '_predict.csv'))
    return results


def result_confusion_matrix(label_list, pred_list, save_path, title, mode):
    if mode == '4cls':
        cls_report = skmet.classification_report(label_list, pred_list, labels=['Covid19', 'Bacterial', 'Viral', 'Normal'])
        con_mat = skmet.confusion_matrix(label_list, pred_list, labels=['Covid19', 'Bacterial', 'Viral', 'Normal'])
        conf_mat = {'X': [], 'Covid19(Truth)': [], 'Bacterial(Truth)': [], 'Viral(Truth)': [], 'Normal(Truth)': []}
        conf_mat['X'] = ['Covid19(Predict)', 'Bacterial(Predict)', 'Viral(Predict)', 'Normal(Predict)', 'Sensitivity(%)', 'Accuracy(%)']
        conf_mat['Covid19(Truth)'] = list(con_mat[0]) + [np.around(100 * (con_mat[0][0] / np.sum(con_mat[0]))),
                                                    np.around(100 * ((con_mat[0][0] + con_mat[1][1] + con_mat[2][2] + con_mat[3][3])
                                                                     / np.sum(con_mat)))]
        conf_mat['Bacterial(Truth)'] = list(con_mat[1]) + [np.around(100 * (con_mat[1][1] / np.sum(con_mat[1]))), None]
        conf_mat['Viral(Truth)'] = list(con_mat[2]) + [np.around(100 * (con_mat[2][2] / np.sum(con_mat[2]))), None]
        conf_mat['Normal(Truth)'] = list(con_mat[3]) + [np.around(100 * (con_mat[3][3] / np.sum(con_mat[3]))), None]
        cm_df = pd.DataFrame(conf_mat, dtype=np.uint8)
        cm_df.to_csv(os.path.join(save_path, title + '_confMatrix.csv'))
        return cls_report, cm_df
    elif mode == '3cls':
        cls_report = skmet.classification_report(label_list, pred_list, labels=['Covid19', 'Pneumonia', 'Normal'])
        con_mat = skmet.confusion_matrix(label_list, pred_list, labels=['Covid19', 'Pneumonia', 'Normal'])
        conf_mat = {'X': [], 'Covid19(Truth)': [], 'Pneumonia(Truth)': [], 'Normal(Truth)': []}
        conf_mat['X'] = ['Covid19(Predict)', 'Pneumonia(Predict)', 'Normal(Predict)',
                         'Sensitivity(%)', 'Accuracy(%)']
        conf_mat['Covid19(Truth)'] = list(con_mat[0]) + [np.around(100 * (con_mat[0][0] / np.sum(con_mat[0]))),
                                                         np.around(100 * ((con_mat[0][0] + con_mat[1][1] + con_mat[2][2])
                                                                          / np.sum(con_mat)))]
        conf_mat['Pneumonia(Truth)'] = list(con_mat[1]) + [np.around(100 * (con_mat[1][1] / np.sum(con_mat[1]))), None]
        conf_mat['Normal(Truth)'] = list(con_mat[2]) + [np.around(100 * (con_mat[2][2] / np.sum(con_mat[2]))), None]
        cm_df = pd.DataFrame(conf_mat, dtype=np.uint8)
        cm_df.to_csv(os.path.join(save_path, title + '_confMatrix.csv'))
        return cls_report, cm_df


def result_roc_auc(label_list, pred_list, pos_id, save_path, title):
    pos_label_list = np.array(label_list)[:, pos_id['idx']]
    pos_pred_list = np.array(pred_list)[:, pos_id['idx']]
    fpr, tpr, thresholds = skmet.roc_curve(pos_label_list, pos_pred_list, pos_label=1)
    roc_auc = skmet.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    lw = 3
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.tick_params(axis='both', which='major', length=10, width=2, direction='in')
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Receiver Operating Characteristic curve ({})'.format(pos_id['name']), fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.savefig(os.path.join(save_path, title + '_ROC_{}.png'.format(pos_id['name'])), format='png')
    return roc_auc

