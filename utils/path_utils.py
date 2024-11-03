import os


def join_and_create(*paths):
    joined_path = os.path.join(*paths)
    os.makedirs(joined_path, exist_ok=True)
    return joined_path


def get_results_path():
    return join_and_create('Experiments')


def get_exp_parent_path(run_name):
    return join_and_create(get_results_path(), run_name)


def get_exp_path(args):
    return join_and_create(get_exp_parent_path(args.run_name), args.version)


def get_checkpts_path(args):
    return join_and_create(get_exp_path(args), 'chckpt')


def get_best_model_path(args):
    checkpts_path = get_checkpts_path(args)
    for ckpt_file in os.listdir(checkpts_path):
        if ckpt_file.startswith("epoch="):
            return os.path.join(checkpts_path, ckpt_file)

    return None


def get_last_checkpts_path(args):
    return join_and_create(get_exp_path(args), 'last_chckpt')


def get_last_model_path(args):
    checkpts_path = get_last_checkpts_path(args)
    last_model_path = os.path.join(checkpts_path, "last.ckpt")
    if os.path.exists(last_model_path):
        return last_model_path
    else:
        return None


def get_visulization_path(args, epoch=0):
    return join_and_create(get_exp_path(args), 'visual', f'epoch_{epoch}')


def get_testset_visulization_path(args, test_data_type, sub_name):
    return join_and_create(get_exp_path(args), 'visual', f'{test_data_type}_set_{sub_name}')


def get_testset_predcitions_path(args):
    return join_and_create(get_exp_path(args), 'test_predictions')


def get_exp_details_path(args):
    return os.path.join(get_exp_path(args), f'exp_details.txt')


def get_model_err_path(args):
    return os.path.join(get_exp_path(args), f'model_err_{args.run_name}_{args.version}.xlsx')


def get_model_test_err_path(args, test_data_type):
    return os.path.join(get_exp_path(args), f'model_{test_data_type}_err_{args.run_name}_{args.version}.xlsx')


def get_eval_results_path(eval_name):
    return join_and_create(get_results_path(), "Evaluation", eval_name)
