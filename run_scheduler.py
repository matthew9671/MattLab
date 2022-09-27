import copy, pprint, traceback

LINE_SEP = "#" * 42

def dict_len(d):
    if (type(d) == list):
        return len(d)
    else:
        return dict_len(d[list(d.keys())[0]])

def dict_map(d, func):
    if type(d) == list:
        return func(d)
    elif type(d) == dict:
        r = copy.deepcopy(d)
        for key in d.keys():
            r[key] = dict_map(r[key], func)
            # Ignore all the Nones
            if r[key] is None:
                r.pop(key)
        if len(r.keys()) == 0:
            # There's no content
            return None
        else:
            return r
    else:
        return None

def dict_product(d1, d2):
    l1, l2 = dict_len(d1), dict_len(d2)
    def expand_list(d):
        result = []
        for item in d:
            result.append(item)
            result.extend([None] * (l2-1))
        return result
    def multiply_list(d):
        return d * l1
    result = dict_map(d1, expand_list)
    additions = dict_map(d2, multiply_list)
    return dict_update(result, additions)

def dict_get(d, id):
    return dict_map(d, lambda l: l[id])

def dict_update(d, u):
    if d is None:
        d = dict()
    for key in u.keys():
        if type(u[key]) == dict:
            d.update({
                key: dict_update(d.get(key), u[key])
            })
        else:
            d.update({key: u[key]})
    return d

# A standardized function that structures and schedules experiments
# Can chain multiple variations of experiment parameters together
def experiment_scheduler(run_params, dataset_getter, model_getter, train_func, 
                         logger_func=None, err_logger_func=None, run_variations=None):
    """
    Arguments:
        run_params: dict{"dataset_params"} 
            A large dictionary containing all relevant parameters to the run 
        dataset_getter: run_params -> dict{"train_data", ["generative_model"]}
            A function that loads/samples a dataset
        model_getter: run_params, data_dict -> model
            A function that creates a model given parameters. Note that the model
            could depend on the specifics of the dataset/generative model as well
        train_func: model, data, run_params -> results
            A function that contains the training loop. 
            TODO: later we might wanna open up this pipeline and customize further!
        (optional) logger_func: results, run_params -> ()
            A function that logs the current run.
        (optional) err_logger_func: message, run_params -> ()
            A function that is called when the run fails.
        (optional) run_variations: dict{}
            A nested dictionary where the leaves are lists of different parameters.
            None means no change from parameters of the last run.
    returns:
        all_results: List<result>
            A list containing results from all runs. Failed runs are indicated
            with a None value.
    """
    num_runs = dict_len(run_variations) if run_variations else 1
    params = copy.deepcopy(run_params)
    print("Total number of runs: {}".format(num_runs))
    print("Base paramerters:")
    pprint(params)

    global data_dict
    all_results = []
    all_models = []

    def _single_run():
        print("Loading dataset!")
        data_dict = dataset_getter(params)
        # Make a new model
        model_dict = model_getter(params, data_dict)
        all_models.append(model_dict)
        results = train_func(model_dict, data_dict, params)
        all_results.append(results)
        if logger_func:
            logger_func(results, params, data_dict)
            
    for run in range(num_runs):
        print(LINE_SEP)
        print("Starting run #{}".format(run))
        print(LINE_SEP)
        curr_variation = dict_get(run_variations, run)
        if curr_variation is None:
            if (run != 0):
                print("Variation #{} is a duplicate, skipping run.".format(run))
                continue
        else:
            print("Current parameter variation:")
            pprint(curr_variation)
            params = dict_update(params, curr_variation)
            if curr_variation.get("dataset_params"):
                reload_data = True
        if (num_runs == 1):
            _single_run()
        else:
            try:
                _single_run()
            except:
                # TODO: call the error logger for more info?
                all_results.append(None)
                print("Run errored out due to some the following reason:")
                traceback.print_exc()
    return all_results, all_models