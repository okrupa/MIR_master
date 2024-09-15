class ModelConfig():

    def __init__(self, model_type, dataset_type, sample_rate = 16000, duration = 1.0, n_way = 3, n_support = 4, n_query = 20, n_unlabeled = 0, n_distractor = 0, n_unlabeled_distractor = 0, 
                                        n_train_episodes = 800, n_val_episodes = 50, num_workers = -1, log_every_n_steps = 1, val_check_interval = 50, dataset_path = None, predict_path= None, classes = [], n_train_dataset = 10, **kwargs):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_unlabeled = n_unlabeled
        self.n_distractor = n_distractor
        self.n_unlabeled_distractor = n_unlabeled_distractor
        self.n_train_episodes = n_train_episodes
        self.n_val_episodes = n_val_episodes
        self.num_workers = num_workers
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.dataset_path = dataset_path
        self.predict_path = predict_path
        self.classes = classes
        self.n_train_dataset = n_train_dataset

